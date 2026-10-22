"""Verify wider model launches, including memory/ROM limits and cached execution."""

import random
import unittest

from snackwell.kernels.inference import Layout, layernorm, loop_layernorm
from snackwell.kv_cache import KVCache
from snackwell.model_runner import forward
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, reference
from snackwell.simulator import compile_model_runner
from tests.test_model import fixture


class WideModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/wide_model"
        cls.executable = compile_model_runner(cls.output)

    def test_looped_layernorm_preserves_unrolled_fp32_results(self):
        rng = random.Random(173)
        for rows in (1, 4, 8):
            # Mixed signs and magnitudes distinguish sequential sums from
            # reassociation. A separate numerical reference checks both runs.
            x = [bits(rng.choice((-1, 1)) * 2 ** rng.randrange(-12, 8)) for _ in range(rows * 64)]
            gamma = [bits(rng.uniform(-2, 2)) for _ in range(64)]
            beta = [bits(rng.uniform(-1, 1)) for _ in range(64)]
            outputs = []
            for emitter in (layernorm, loop_layernorm):
                layout = Layout()
                layout.allocate("x", len(x), x)
                layout.allocate("gamma", 64, gamma)
                layout.allocate("beta", 64, beta)
                runtime = ModelRuntime(
                    self.output / f"norm_{rows}_{emitter.__name__}", self.executable
                )
                outputs.append(
                    runtime.launch(
                        "norm",
                        layout,
                        emitter(layout, rows, 64, 1e-5, "x", "x"),
                        "layernorm",
                        ("x", "gamma", "beta"),
                        "x",
                        rows=rows,
                        cols=64,
                        epsilon=1e-5,
                        active_masks=1 if rows == 1 else 0xFFFF,
                    )
                )
                self.assertLessEqual(runtime.reports[-1]["instructions"], 1024)
            self.assertEqual(outputs[0], outputs[1])

    def test_wide_layernorm_handles_zero_variance_and_distinct_rows(self):
        runtime = ModelRuntime(self.output / "wide_norm", self.executable)
        beta = [bits((i - 64) / 16) for i in range(128)]
        gamma = [bits(0.5 + i / 128) for i in range(128)]
        self.assertEqual(
            runtime.rows("constant", "layernorm", [bits(3)] * 128, 1, 128, gamma=gamma, beta=beta),
            beta,
        )
        x = [bits(((i * 13) % 41 - 20) / 8) for i in range(8 * 128)]
        full = runtime.rows("rows", "layernorm", x, 8, 128, gamma=gamma, beta=beta)
        single = [
            word
            for r in range(8)
            for word in runtime.rows(
                "single",
                "layernorm",
                x[r * 128 : (r + 1) * 128],
                1,
                128,
                gamma=gamma,
                beta=beta,
            )
        ]
        self.assertEqual(full, single)
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))

    def test_wide_gelu_and_embedding_keep_flat_order(self):
        runtime = ModelRuntime(self.output / "wide_elementwise", self.executable)
        for columns in (256, 512):
            for rows in (1, 4, 8):
                x = [bits((i % 117 - 58) / 8) for i in range(rows * columns)]
                got = runtime.rows("wide", "gelu", x, rows, columns)
                expected = [
                    word
                    for offset in range(0, len(x), 128)
                    for word in runtime.rows("slice", "gelu", x[offset : offset + 128], 1, 128)
                ]
                self.assertEqual(got, expected)
        for rows in (1, 4):
            ids = [64 - r for r in range(rows)]
            table = [bits(i / 1024) for i in range(65 * 128)]
            position = [bits(-i / 512) for i in range(rows * 128)]
            got = runtime.lookup(ids, table, position, 128, 65)
            expected = [
                reference(FADD, table[token * 128 + col], position[r * 128 + col])
                for r, token in enumerate(ids)
                for col in range(128)
            ]
            self.assertEqual(got, expected)
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))

    def test_wide_cached_decode_matches_full_forward(self):
        model = fixture(context=8, width=128, layers=2, heads=4, ffn_multiplier=4)
        ids = [1, 2, 3, 4, 1]
        cache = KVCache()
        first = forward(
            model,
            ids,
            self.output / "prefill",
            self.executable,
            last_only=True,
            kv_cache=cache,
            draw=0.37,
        )
        ids.append(2)
        cached = forward(
            model,
            ids,
            self.output / "cached",
            self.executable,
            last_only=True,
            kv_cache=cache,
            draw=0.37,
        )
        full = forward(model, ids, self.output / "full", self.executable, draw=0.37)
        self.assertTrue(first["verified"] and cached["verified"] and full["verified"])
        self.assertEqual(cached["probabilities"], full["probabilities"][-5:])
        self.assertEqual(cached["selection"], full["selection"])
        self.assertEqual(cache.ids, tuple(ids))
        self.assertTrue(all(len(k) == len(v) == len(ids) * 128 for k, v in cache.layers))
        self.assertLess(cached["cycles"], full["cycles"])


if __name__ == "__main__":
    unittest.main()
