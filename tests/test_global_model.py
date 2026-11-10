"""All model operands cross the external-memory port; transport cannot change math."""

import copy
import random
import unittest
from dataclasses import replace
from unittest.mock import patch

from snackwell.assembler import assemble
from snackwell.global_memory import GLOBAL_WORDS, run_model_global
from snackwell.kernels.global_model import prepare
from snackwell.kernels.inference import Layout
from snackwell.kv_cache import Generation, KVCache
from snackwell.model_runner import forward
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import SENTINEL, bits
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, compile_model_runner, read_hex
from tests.test_model import fixture


class GlobalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/global_model"
        cls.simulator = compile_model_runner(cls.output, "verilator", compact=True)

    def test_regular_kernels_multiwarp_partial_masks_and_global_guards(self):
        rng = random.Random(704)
        with SimulatorSession(self.simulator) as session:
            shared = ModelRuntime(self.output / "regular_shared", session)
            external = ModelRuntime(
                self.output / "regular_global", session, memory="global", memory_latency=5
            )
            a = [bits(rng.uniform(-1, 1)) for _ in range(8 * 16)]
            b = [bits(rng.uniform(-1, 1)) for _ in range(16 * 16)]
            self.assertEqual(
                shared.matmul("matmul", a, b, 8, 16, 16), external.matmul("matmul", a, b, 8, 16, 16)
            )
            logits = [bits(i / 8) for i in range(8)]
            self.assertEqual(
                shared.rows("softmax", "softmax", logits, 1, 8),
                external.rows("softmax", "softmax", logits, 1, 8),
            )
            self.assertTrue(any(r["warp_mask"] == 15 for r in external.reports))
            self.assertTrue(any(r["active_masks"] == 1 for r in external.reports))
            for report in external.reports:
                metrics = report["memory"]
                self.assertGreater(metrics["reads"], 0)
                self.assertGreater(metrics["writes"], 0)
                self.assertEqual(metrics["copy_reads"], 0)
                self.assertEqual(metrics["reads"] + metrics["writes"], metrics["responses"])

    def test_native_multiple_tiles_k_chunks_and_latency_invariance(self):
        rng = random.Random(705)
        with SimulatorSession(self.simulator) as session:
            for m, n, k, latency in ((1, 4, 4, 0), (4, 12, 32, 3), (5, 20, 160, 11)):
                a = [bits(rng.uniform(-3, 3)) for _ in range(m * k)]
                b = [bits(rng.uniform(-2, 2)) for _ in range(n * k)]
                bias = [bits(rng.uniform(-1, 1)) for _ in range(n)]
                shared = ModelRuntime(self.output / "native_shared", session)
                external = ModelRuntime(
                    self.output / "native_global", session, memory="global", memory_latency=latency
                )
                self.assertEqual(
                    shared.nvfp4_linear("projection", a, b, m, n, k, bias),
                    external.nvfp4_linear("projection", a, b, m, n, k, bias),
                )
                self.assertTrue(
                    all(
                        r["memory"]["nvfp4_macs"] > 0 and r["memory"]["fp32_tensor_multiplies"] == 0
                        for r in external.reports
                    )
                )

    def test_full_model_cache_sampling_rebuild_matches_shared_exactly(self):
        model = fixture(context=4, width=8, heads=2, layers=2)
        for precision in ("fp32", "nvfp4"):
            runs = {
                mode: Generation(model, "abc", heal=False, precision=precision, memory=mode)
                for mode in ("shared", "global")
            }
            for step in range(3):
                reports = {
                    mode: run.next(
                        self.output / f"{precision}_{mode}_{step}",
                        self.simulator,
                        draw=0.4,
                        temperature=0.7,
                    )
                    for mode, run in runs.items()
                }
                self.assertEqual(
                    reports["shared"]["probabilities"], reports["global"]["probabilities"]
                )
                self.assertEqual(reports["shared"]["selection"], reports["global"]["selection"])
                self.assertTrue(
                    all(
                        s["memory"]["transport"] == "simulated_external_memory"
                        for s in reports["global"]["stages"]
                    )
                )
            self.assertEqual(reports["global"]["cache"]["phase"], "rebuild")

    def test_memory_switch_and_corruption_do_not_commit_cache(self):
        model, cache = fixture(), KVCache()
        forward(model, [1], self.output / "cache_seed", self.simulator, kv_cache=cache)
        before = copy.deepcopy(cache)
        with self.assertRaisesRegex(ValueError, "memory"):
            forward(
                model,
                [1, 2],
                self.output / "cache_wrong",
                self.simulator,
                kv_cache=cache,
                memory="global",
            )
        self.assertEqual(cache, before)

        def corrupt(*args, **kwargs):
            shared, registers, memory, metrics = run_model_global(*args, **kwargs)
            memory[-1] ^= 1
            return shared, registers, memory, metrics

        empty = KVCache()
        output = self.output / "corrupt"
        with (
            patch("snackwell.global_memory.run_model_global", corrupt),
            self.assertRaisesRegex(ValueError, "memory guard"),
        ):
            forward(
                model,
                [1],
                output,
                self.simulator,
                kv_cache=empty,
                precision="nvfp4",
                memory="global",
            )
        self.assertEqual(empty, KVCache())
        self.assertFalse((output / "result.json").exists())
        self.assertTrue(next(output.glob("*/failure.json")).exists())
        failed = next(output.glob("*/failure.json")).parent
        self.assertEqual(read_hex(failed / "input.hex", 1024), [SENTINEL] * 1024)
        self.assertEqual(len(read_hex(failed / "global-input.hex", GLOBAL_WORDS)), GLOBAL_WORDS)

    def test_external_fault_timeout_and_restart(self):
        memory = [SENTINEL] * GLOBAL_WORDS
        good = assemble("XOR R0, R0, R0\nADDI R1, R0, 17\nST_GLOBAL [R0], R1\nEXIT")
        bad = assemble("XOR R0, R0, R0\nLD_GLOBAL R1, [R0 + 1]\nEXIT")
        with SimulatorSession(self.simulator) as session:
            _, _, _, metrics = run_model_global(
                session, bad, memory, self.output / "fault", active_masks=15, warp_mask=1, latency=2
            )
            self.assertNotEqual(metrics["status"], 0)
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                run_model_global(
                    session,
                    good,
                    memory,
                    self.output / "timeout",
                    active_masks=15,
                    warp_mask=1,
                    latency=2,
                    max_cycles=1,
                )
            self.assertIsNone(session.process)
            _, _, after, metrics = run_model_global(
                session,
                good,
                memory,
                self.output / "restart",
                active_masks=15,
                warp_mask=1,
                latency=2,
            )
            self.assertEqual(after[0], 17)
            self.assertEqual(metrics["status"], 0)
        with self.assertRaises(ValueError):
            run_model_global(
                self.simulator,
                good,
                memory[:-1],
                self.output,
                active_masks=15,
                warp_mask=1,
                latency=2,
            )

    def test_serial_and_overlapped_native_schedule_match(self):
        from snackwell.reference.nvfp4 import quantize

        q = quantize([[float(k % 7) for k in range(32)]] * 4)
        results = []
        with SimulatorSession(self.simulator) as session:
            for overlap in (False, True):
                layout = Layout()
                packed = [word for g in range(2) for word in q.group_words(0, g)]
                layout.allocate("a", len(packed), packed)
                layout.allocate("b", len(packed), packed)
                layout.allocate("scale_a", 4, [q.tensor_scale] * 4)
                layout.allocate("scale_b", 1, [q.tensor_scale])
                layout.allocate("out", 16)
                params = dict(n=4, k=32, accumulate=False, final=True, bias=False)
                source, memory, wanted = prepare(
                    layout, "", "nvfp4_matmul", params, overlap=overlap
                )
                shared, _, after, metrics = run_model_global(
                    session,
                    assemble(source),
                    memory,
                    self.output,
                    active_masks=15,
                    warp_mask=1,
                    latency=2,
                )
                self.assertEqual(shared, wanted)
                start, count = layout.regions["out"]
                results.append((after[start : start + count], metrics))
        self.assertEqual(results[0][0], results[1][0])
        self.assertLess(results[1][1]["cycles"], results[0][1]["cycles"])
        self.assertGreater(results[1][1]["copy_arithmetic_overlap_cycles"], 0)

    def test_icarus_and_persistent_verilator_match(self):
        model = fixture()
        icarus = compile_model_runner(self.output / "icarus", "icarus", trace=True)
        a = forward(
            model, [1], self.output / "parity_icarus", icarus, precision="nvfp4", memory="global"
        )
        b = forward(
            model,
            [1],
            self.output / "parity_verilator",
            self.simulator,
            precision="nvfp4",
            memory="global",
        )
        self.assertEqual(a["probabilities"], b["probabilities"])
        for left, right in zip(a["stages"], b["stages"], strict=True):
            for metric in (
                "reads",
                "writes",
                "responses",
                "copy_reads",
                "copy_writes",
                "nvfp4_macs",
                "fp32_tensor_multiplies",
            ):
                self.assertEqual(left["memory"][metric], right["memory"][metric])

    def test_traced_verilator_uses_external_memory_and_saves_artifacts(self):
        traced = replace(self.simulator, trace=True, compact=False)
        runtime = ModelRuntime(self.output / "traced", traced, memory="global")
        self.assertEqual(
            runtime.nvfp4_linear("linear", [bits(1)] * 16, [bits(1)] * 64, 1, 4, 16), [bits(16)] * 4
        )
        self.assertTrue((self.output / "traced/000_linear/trace.csv").is_file())
        self.assertEqual(runtime.reports[0]["memory"]["nvfp4_macs"], 256)
        self.assertEqual(
            read_hex(self.output / "traced/000_linear/input.hex", 1024), [SENTINEL] * 1024
        )
