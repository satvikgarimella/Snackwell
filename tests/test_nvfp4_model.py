"""Native linear execution, independent model math, and precision-isolated caches."""

import copy
import csv
import json
import random
import unittest
from dataclasses import replace
from unittest.mock import patch

from snackwell.kv_cache import Generation, KVCache
from snackwell.model_runner import forward
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits
from snackwell.nvfp4_inference import activation_rows, weight_columns
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, reference
from snackwell.reference.native import nvfp4_product
from snackwell.reference.nvfp4 import matmul, quantize
from snackwell.simulator import SimulatorSession, compile_model_runner
from tests.test_model import fixture


class NVFP4ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/nvfp4_model"
        cls.executable = compile_model_runner(cls.output, "verilator", compact=True)

    def test_native_integer_checker_matches_python_oracle(self):
        rng = random.Random(781)
        for k in (16, 48):
            a = quantize([[rng.uniform(-4, 4) for _ in range(k)] for _ in range(3)])
            b = quantize([[rng.uniform(-8, 8) for _ in range(k)] for _ in range(5)])
            initial = [bits(rng.uniform(-3, 3)) for _ in range(15)]
            for apply in (False, True):
                self.assertEqual(
                    nvfp4_product(a, b, initial, apply_tensor_scale=apply),
                    matmul(a, b, initial, apply_tensor_scale=apply),
                )

    def test_padded_shapes_bias_and_multiple_k_chunks(self):
        rng = random.Random(774)
        with SimulatorSession(self.executable) as session:
            runtime = ModelRuntime(self.output / "matrices", session)
            for m, n, k in ((1, 4, 4), (4, 12, 32), (5, 20, 160)):
                a = [bits(rng.uniform(-2, 2)) for _ in range(m * k)]
                b = [bits(rng.uniform(-3, 3)) for _ in range(n * k)]
                bias = [bits(rng.uniform(-0.1, 0.1)) for _ in range(n)]
                left, right = activation_rows(a, m, k), weight_columns(tuple(b), n, k)
                expected = [
                    reference(FADD, w, bias[c])
                    for row in left
                    for c, w in enumerate(matmul(row, right))
                ]
                got = runtime.nvfp4_linear("linear", a, b, m, n, k, bias=bias)
                self.assertEqual(got, expected)
            zero = runtime.nvfp4_linear("zeros", [0] * 16, [0] * 64, 1, 4, 16)
            self.assertEqual(zero, [0] * 4)
            self.assertTrue(
                all(
                    r["instructions"] <= 1024 and r["allocated_words"] <= 1024
                    for r in runtime.reports
                )
            )
            with self.assertRaises(ValueError):
                runtime.nvfp4_linear("nan", [0x7FC00000] * 16, [0] * 64, 1, 4, 16)

    def test_full_model_cached_decode_matches_fresh_and_is_causal(self):
        model = fixture(context=8, width=8, heads=2, layers=2)
        cache = KVCache()
        first = forward(
            model,
            [1, 2, 3],
            self.output / "prefill",
            self.executable,
            precision="nvfp4",
            kv_cache=cache,
            last_only=True,
        )
        decoded = forward(
            model,
            [1, 2, 3, 4],
            self.output / "decode",
            self.executable,
            precision="nvfp4",
            kv_cache=cache,
            last_only=True,
            draw=0.3,
            temperature=0.7,
        )
        full = forward(
            model, [1, 2, 3, 4], self.output / "full", self.executable, precision="nvfp4"
        )
        changed = forward(
            model, [1, 4, 4, 4], self.output / "causal", self.executable, precision="nvfp4"
        )
        self.assertEqual(decoded["probabilities"], full["probabilities"][-5:])
        self.assertEqual(changed["probabilities"][:5], full["probabilities"][:5])
        self.assertTrue(first["verified"] and decoded["verified"])
        self.assertEqual(cache.precision, "nvfp4-linear-row-v1")
        self.assertTrue(any(s["operation"] == "nvfp4_matmul" for s in decoded["stages"]))
        self.assertTrue(
            any(
                s["operation"] == "matmul" and "attention_qk" in s["stage"]
                for s in decoded["stages"]
            )
        )

    def test_precision_switch_rejected_and_failed_run_does_not_commit_cache(self):
        model = fixture()
        cache = KVCache()
        forward(model, [1], self.output / "cache_seed", self.executable, kv_cache=cache)
        before = copy.deepcopy(cache)
        with self.assertRaisesRegex(ValueError, "precision"):
            forward(
                model,
                [1, 2],
                self.output / "wrong_precision",
                self.executable,
                kv_cache=cache,
                precision="nvfp4",
            )
        self.assertEqual(cache, before)
        original = ModelRuntime.nvfp4_linear

        def corrupt(runtime, *args, **kwargs):
            result = original(runtime, *args, **kwargs)
            result[0] ^= 0x00800000
            return result

        empty = KVCache()
        output = self.output / "corrupted"
        output.mkdir(exist_ok=True)
        (output / "result.json").write_text('{"verified":true}')
        with (
            patch.object(ModelRuntime, "nvfp4_linear", corrupt),
            self.assertRaisesRegex(ValueError, "bit mismatch"),
        ):
            forward(model, [1, 2], output, self.executable, kv_cache=empty, precision="nvfp4")
        self.assertEqual(empty, KVCache())
        self.assertFalse((output / "result.json").exists())

    def test_icarus_verilator_parity_and_issued_native_instructions(self):
        model = fixture()
        icarus = compile_model_runner(self.output / "icarus", "icarus", trace=True)
        a = forward(model, [1], self.output / "parity_icarus", icarus, precision="nvfp4")
        b = forward(
            model, [1], self.output / "parity_verilator", self.executable, precision="nvfp4"
        )
        self.assertEqual(a["probabilities"], b["probabilities"])
        self.assertEqual(a["cycles"], b["cycles"])
        launches = list((self.output / "parity_icarus").glob("*/result.json"))
        native = 0
        for path in launches:
            if json.loads(path.read_text())["operation"] != "nvfp4_matmul":
                continue
            with path.with_name("trace.csv").open() as stream:
                issued = [
                    int(r["opcode"])
                    for r in csv.DictReader(stream)
                    if int(r["instruction_accepted"])
                ]
            self.assertNotIn(0x0F, issued)  # Legacy TMUL must not execute in a native launch.
            native += issued.count(0x24) + issued.count(0x25)
        self.assertGreater(native, 0)

    def test_generation_rebuild_keeps_precision_and_model_policy(self):
        model = fixture(context=4)
        generation = Generation(model, "abc", precision="nvfp4", heal=False)
        phases = []
        for step in range(3):
            report = generation.next(self.output / f"rebuild_{step}", self.executable)
            phases.append(report["cache"]["phase"])
            self.assertEqual(report["precision"], "nvfp4")
        self.assertEqual(phases, ["prefill", "decode", "rebuild"])
        with self.assertRaises(ValueError):
            Generation(model, "abc", precision="bad")

    def test_tensor_status_failure_is_reported(self):
        runtime = ModelRuntime(self.output / "bad_status", replace(self.executable, compact=False))
        from snackwell.model_runtime import run_program

        def bad_status(*args, **kwargs):
            after, registers, cycles = run_program(*args, **kwargs)
            registers[14] = 64
            return after, registers, cycles

        with (
            patch("snackwell.model_runtime.run_program", bad_status),
            self.assertRaisesRegex(ValueError, "tensor status"),
        ):
            runtime.nvfp4_linear("bad_status", [bits(1)] * 16, [bits(1)] * 64, 1, 4, 16)
