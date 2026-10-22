"""Exercise the one-token paths against their full-row and numerical references."""

import random
import unittest

from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, reference
from snackwell.reference.inference import tensor_product
from snackwell.simulator import compile_model_runner


class DecodeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/decode_kernels"
        cls.executable = compile_model_runner(cls.output)

    def runtime(self, name):
        return ModelRuntime(self.output / name, self.executable)

    def test_single_row_matmul_column_tails_k_chunks_bias_and_transpose(self):
        rng = random.Random(981)
        runtime = self.runtime("matmul")
        for n, k in ((4, 4), (28, 32), (32, 64), (64, 128), (68, 36)):
            a = [bits(rng.choice((-1, 1)) * 2 ** rng.randrange(-10, 10)) for _ in range(k)]
            b = [bits(rng.uniform(-1, 1)) for _ in range(k * n)]
            bias = [bits(rng.uniform(-2, 2)) for _ in range(n)]
            product = tensor_product(a, b, 1, n, k)
            expected = [reference(FADD, x, y) for x, y in zip(product, bias)]
            for transpose in (False, True):
                right = [b[r * n + c] for c in range(n) for r in range(k)] if transpose else b
                with self.subTest(n=n, k=k, transpose=transpose):
                    got = runtime.matmul("single", a, right, 1, n, k, transpose, bias)
                    self.assertEqual(got, expected)
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))

    def test_row_operations_match_first_row_of_full_tile(self):
        runtime = self.runtime("rows")
        rng = random.Random(919)
        for cols in (4, 64, 128):
            x = [bits(rng.uniform(-4, 4)) for _ in range(cols)]
            gamma = [bits(rng.uniform(0.5, 1.5)) for _ in range(cols)]
            beta = [bits(rng.uniform(-0.2, 0.2)) for _ in range(cols)]
            cases = [
                ("gelu", {}),
                ("softmax", {"temperature": 0.7}),
                ("layernorm", {"gamma": gamma, "beta": beta}),
                ("bias_add", {"other": beta}),
                ("residual_add", {"other": beta}),
                ("vocabulary_mask", {"vocabulary": cols - 1}),
                ("argmax", {}),
            ]
            for kind, kwargs in cases:
                with self.subTest(cols=cols, kind=kind):
                    single = runtime.rows("single", kind, x, 1, cols, **kwargs)
                    if kind == "residual_add":
                        kwargs = dict(other=beta * 4)
                    tiled = runtime.rows("tile", kind, x * 4, 4, cols, **kwargs)
                    self.assertEqual(single, tiled[: len(single)])

    def test_single_row_sampling_boundaries(self):
        runtime = self.runtime("sampling")
        weights = [bits(x) for x in [0, 0.25, 0, 0.5, 0.25] + [0] * 123]
        for draw, expected in ((0, 1), (0.25 - 2**-24, 1), (0.25, 3), (0.75, 4), (1 - 2**-24, 4)):
            self.assertEqual(
                runtime.rows("sample", "categorical", weights, 1, 128, draws=[bits(draw)]),
                [expected],
            )

    def test_mask_loop_preserves_absolute_positions_through_column_127(self):
        runtime = self.runtime("mask")
        for rows, offset in ((1, 0), (1, 64), (1, 127), (4, 60), (4, 124), (128, 0)):
            got = runtime.attention_scale([bits(2)] * (rows * 128), rows, 4, 128, offset)
            expected = [
                bits(1) if col <= offset + row else 0xFF800000
                for row in range(rows)
                for col in range(128)
            ]
            self.assertEqual(got, expected)
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))

    def test_lookup_single_row_capacity_boundary_and_long_context(self):
        runtime = self.runtime("lookup")
        for vocabulary, rows in ((65, 1), (120, 1), (121, 1), (128, 128)):
            width = 8
            ids = [(vocabulary - 1 - i) % vocabulary for i in range(rows)]
            table = [bits(i / 16) for i in range(vocabulary * width)]
            position = [bits(-i / 32) for i in range(rows * width)]
            got = runtime.lookup(ids, table, position, width, vocabulary)
            expected = [
                reference(FADD, table[token * width + c], position[r * width + c])
                for r, token in enumerate(ids)
                for c in range(width)
            ]
            self.assertEqual(got, expected)
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))

    def test_invalid_partial_row_shapes_are_rejected_before_launch(self):
        runtime = self.runtime("invalid")
        for rows in (0, 2, 3, 5):
            with self.assertRaises(ValueError):
                runtime.matmul("invalid", [], [], rows, 4, 4)
            with self.assertRaises(ValueError):
                runtime.rows("invalid", "gelu", [], rows, 4)
        self.assertFalse(runtime.reports)
        with self.assertRaises(ValueError):
            runtime.rows("invalid", "layernorm", [0] * 256, 1, 256)
