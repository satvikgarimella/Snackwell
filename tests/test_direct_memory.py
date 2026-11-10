"""Direct pointers preserve arithmetic, aliasing and ownership without layout copies."""

import json
import random
import unittest
from dataclasses import replace
from unittest.mock import patch

from snackwell.kernels.resident import lower
from snackwell.kv_cache import Generation
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, compile_model_runner
from tests.test_model import fixture


class DirectMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/direct_memory"
        cls.simulator = compile_model_runner(cls.output, "verilator", compact=True)

    def runtimes(self, a, b, name):
        return (
            ModelRuntime(self.output / name / "copies", a, memory="resident-copy"),
            ModelRuntime(self.output / name / "direct", b, memory="resident"),
        )

    def test_fp32_multiwarp_matvec_tiled_transpose_and_k_carry(self):
        rng = random.Random(4100)
        with SimulatorSession(self.simulator) as a, SimulatorSession(self.simulator) as b:
            old, direct = self.runtimes(a, b, "fp32")
            for m, n, k in ((8, 16, 32), (1, 28, 64), (4, 20, 40)):
                left = [bits(rng.uniform(-1, 1)) for _ in range(m * k)]
                right = [bits(rng.uniform(-1, 1)) for _ in range(k * n)]
                bias = [bits(rng.uniform(-1, 1)) for _ in range(n)]
                for transpose in (False, True):
                    expected = old.matmul("matrix", left, right, m, n, k, transpose, bias)
                    actual = direct.matmul("matrix", left, right, m, n, k, transpose, bias)
                    self.assertEqual(expected, actual)
            self.assertTrue(all(r["memory"]["copy_launches"] == 0 for r in direct.reports))
            self.assertTrue(any(r["warp_mask"] == 15 for r in direct.reports))

    def test_native_maximum_jobs_partial_rows_and_k_chunks(self):
        rng = random.Random(4101)
        with SimulatorSession(self.simulator) as a, SimulatorSession(self.simulator) as b:
            old, direct = self.runtimes(a, b, "nvfp4")
            for m, n, k in ((1, 4, 4), (5, 20, 160), (4, 16, 128)):
                left = [bits(rng.uniform(-3, 3)) for _ in range(m * k)]
                right = [bits(rng.uniform(-2, 2)) for _ in range(k * n)]
                bias = [bits(rng.uniform(-1, 1)) for _ in range(n)]
                self.assertEqual(
                    old.nvfp4_linear("matrix", left, right, m, n, k, bias),
                    direct.nvfp4_linear("matrix", left, right, m, n, k, bias),
                )
            self.assertTrue(all(r["memory"]["device_copy_words"] == 0 for r in direct.reports))
            self.assertTrue(
                any(r["memory"]["copy_arithmetic_overlap_cycles"] for r in direct.reports)
            )
            self.assertEqual(
                sum(r["memory"]["nvfp4_macs"] for r in old.reports),
                sum(r["memory"]["nvfp4_macs"] for r in direct.reports),
            )

    def test_multi_pass_row_kernels_wide_shapes_and_partial_output(self):
        rng = random.Random(4102)
        with SimulatorSession(self.simulator) as a, SimulatorSession(self.simulator) as b:
            old, direct = self.runtimes(a, b, "rows")
            for rows, cols in ((1, 4), (4, 32), (4, 128), (1, 256)):
                data = [bits(rng.uniform(-2, 2)) for _ in range(rows * cols)]
                gamma, beta = [bits(1.1)] * cols, [bits(0.2)] * cols
                for kind, options in (
                    ("layernorm", dict(gamma=gamma, beta=beta)),
                    ("gelu", {}),
                    ("softmax", dict(temperature=0.7)),
                    ("residual_add", dict(other=data)),
                    ("vocabulary_mask", dict(vocabulary=cols - 3)),
                ):
                    self.assertEqual(
                        old.rows(kind, kind, data, rows, cols, **options),
                        direct.rows(kind, kind, data, rows, cols, **options),
                    )
            logits = [bits(-i / 64) for i in range(512)]
            probs = [r.rows("softmax", "softmax", logits, 1, 512) for r in (old, direct)]
            self.assertEqual(*probs)
            for kind, opts in (
                ("min_p", dict(min_p=0.1)),
                ("categorical", dict(draws=[bits(0.4)])),
            ):
                self.assertEqual(
                    old.rows(kind, kind, probs[0], 1, 512, **opts),
                    direct.rows(kind, kind, probs[1], 1, 512, **opts),
                )

    def test_aliases_and_fragmented_views_retain_original_values(self):
        with SimulatorSession(self.simulator) as session:
            runtime = ModelRuntime(self.output / "aliases", session, memory="resident")
            left = runtime.device.constant("left", [bits(i / 8) for i in range(16)])
            right = runtime.device.constant("right", [bits(-i / 16) for i in range(16)])
            view = [word for pair in zip(left, right) for word in pair]
            original = list(map(int, view))
            transformed = runtime.rows("gelu", "gelu", view, 4, 8)
            summed = runtime.rows("residual", "residual_add", view, 4, 8, other=transformed)
            self.assertEqual(list(map(int, view)), original)
            self.assertTrue(
                all(runtime.device.shadow[runtime.device.address(w)] == w for w in view)
            )
            self.assertEqual(len(summed), 32)
            # A mask writes only the padding. Valid entries must still retain
            # their input allocation, including when the input came from host words.
            masked = runtime.rows("mask", "vocabulary_mask", original, 4, 8, vocabulary=5)
            runtime.rows("overwrite_scratch", "gelu", [bits(9)] * 32, 4, 8)
            runtime.rows("consume_mask", "softmax", masked, 4, 8)
            self.assertTrue(
                all(runtime.device.shadow[runtime.device.address(w)] == w for w in masked)
            )
            unchanged = runtime.rows("no_mask", "vocabulary_mask", view, 4, 8, vocabulary=8)
            self.assertEqual(
                [runtime.device.address(w) for w in unchanged],
                [runtime.device.address(w) for w in view],
            )

    def test_causal_masks_token_masks_and_indirect_embedding_indices(self):
        with SimulatorSession(self.simulator) as a, SimulatorSession(self.simulator) as b:
            old, direct = self.runtimes(a, b, "addressing")
            for rows, cols in ((1, 64), (4, 64), (4, 128)):
                data = [bits((i % 17 - 8) / 16) for i in range(rows * cols)]
                self.assertEqual(
                    old.attention_scale(data, rows, 8, cols, cols - rows),
                    direct.attention_scale(data, rows, 8, cols, cols - rows),
                )
            table = [bits(i / 2048) for i in range(257 * 8)]
            ids = [256, 0, 127, 255] * 8
            for count in (1, 32):
                positions = [bits(0.1)] * (count * 8)
                self.assertEqual(
                    old.lookup(ids[:count], table, positions, 8, 257),
                    direct.lookup(ids[:count], table, positions, 8, 257),
                )
            logits = [bits(i / 32) for i in range(64)]
            allowed = [i % 3 == 0 for i in range(64)]
            self.assertEqual(
                old.restrict_tokens("restrict", logits, allowed),
                direct.restrict_tokens("restrict", logits, allowed),
            )

    def test_direct_and_copy_generation_match_decode_rebuild_and_cache_identity(self):
        model = fixture(context=4, width=8, heads=2, layers=2)
        for precision in ("fp32", "nvfp4"):
            with (
                Generation(
                    model, "abc", heal=False, memory="resident-copy", precision=precision
                ) as old,
                Generation(
                    model, "abc", heal=False, memory="resident", precision=precision
                ) as direct,
            ):
                for i in range(3):
                    a, b = [
                        g.next(
                            self.output / f"{precision}_{g.memory}_{i}", self.simulator, draw=0.4
                        )
                        for g in (old, direct)
                    ]
                    self.assertEqual(a["probabilities"], b["probabilities"])
                    self.assertEqual(a["selection"], b["selection"])
                    self.assertLess(b["cycles"], a["cycles"])
                    self.assertEqual(b["launches"], b["model_launches"])
                    self.assertEqual(b["resident"]["constant_upload_bytes"] == 0, i > 0)
                direct.memory = "resident-copy"
                with self.assertRaisesRegex(ValueError, "memory"):
                    direct.next(self.output / "switch", self.simulator)
                self.assertIsNone(direct.session)

    def test_pointer_corruption_fails_and_full_artifacts_describe_physical_uploads(self):
        original = SimulatorSession.run_resident_words

        def corrupt(session, *args, **kwargs):
            shared, registers, values, metrics = original(session, *args, **kwargs)
            shared[0] ^= 4
            return shared, registers, values, metrics

        with Generation(fixture(), "a", heal=False, memory="resident") as generation:
            with (
                patch.object(SimulatorSession, "run_resident_words", corrupt),
                self.assertRaisesRegex(ValueError, "pointer"),
            ):
                generation.next(self.output / "corruption", self.simulator)
            self.assertFalse(generation.cache.ids)
        with SimulatorSession(replace(self.simulator, compact=False)) as session:
            runtime = ModelRuntime(self.output / "artifacts", session, memory="resident")
            values = runtime.device.constant("input", [bits(1)] * 4)
            runtime.rows("gelu", "gelu", values, 1, 4)
            directory = self.output / "artifacts/000_gelu"
            layout = json.loads((directory / "resident-layout.json").read_text())
            self.assertTrue(all(p >= 16384 for p in layout["input_word_addresses"]))
            self.assertTrue(
                any(start == values[0].allocation.start for start, _ in layout["uploads"])
            )
            self.assertTrue((directory / "logical-output.hex").is_file())
            self.assertIn("LD_SHARED", (directory / "program.asm").read_text())

    def test_lowering_rejects_unrelocatable_branches_and_register_exhaustion(self):
        with self.assertRaisesRegex(ValueError, "symbolic"):
            lower("ST_GLOBAL [R0], R1\nBEQ R0, R0, -1\nEXIT")
        source = (
            "ST_GLOBAL [R0], R1\n"
            + "\n".join(f"FADD R{i}, R{i}, R{i}" for i in range(16))
            + "\nEXIT"
        )
        with self.assertRaisesRegex(ValueError, "two dead"):
            lower(source)


if __name__ == "__main__":
    unittest.main()
