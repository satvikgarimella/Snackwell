import random
import unittest
from unittest.mock import patch

from snackwell.reference.fp32 import FADD, FMUL, FSUB, reference
from snackwell.reference.inference import tensor_product as python_product
from snackwell.reference.native import native_library, tensor_product


class NativeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = native_library()
        if cls.library is None:
            raise unittest.SkipTest("requires a C++ compiler for the native integer oracle")

    def test_scalar_oracle_matches_python_edges_random_and_cancellation(self):
        rng = random.Random(0x0FAC1E)
        edges = [
            0,
            0x80000000,
            1,
            0x80000001,
            0x7FFFFF,
            0x800000,
            0x3F800000,
            0xBF800000,
            0x3F800001,
            0x33800000,
            0x33000000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
        ]
        pairs = [(a, b) for a in edges for b in edges]
        pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(20000)]
        # Every normal exponent, adjacent magnitudes, and powers of two test
        # rounding carries, cancellation and the normal/subnormal boundary.
        for exponent in range(1, 255):
            a = exponent << 23
            for delta in (-2, -1, 0, 1, 2):
                pairs.extend([(a, a + delta), (a, (a + delta) ^ 0x80000000)])
        for opcode in (FADD, FSUB, FMUL):
            for a, b in pairs:
                self.assertEqual(
                    self.library.snackwell_fp32(opcode, a, b),
                    reference(opcode, a, b),
                    f"opcode={opcode:x} a={a:08x} b={b:08x}",
                )

    def test_matrix_order_transposition_and_initial_sum(self):
        rng = random.Random(9182)
        for m, n, k in ((4, 4, 4), (4, 8, 64), (8, 16, 32), (4, 4, 128)):
            a = [rng.getrandbits(32) for _ in range(m * k)]
            b = [rng.getrandbits(32) for _ in range(k * n)]
            transposed = [b[r * n + c] for c in range(n) for r in range(k)]
            for initial in (None, [rng.getrandbits(32) for _ in range(m * n)]):
                expected = python_product(a, b, m, n, k, initial=initial)
                self.assertEqual(tensor_product(a, b, m, n, k, initial=initial), expected)
                self.assertEqual(tensor_product(a, transposed, m, n, k, True, initial), expected)


class ReferenceFallbackTests(unittest.TestCase):
    def test_python_fallback_and_invalid_dimensions(self):
        a, b = [0x3F800001] * 16, [0x3F7FFFFE] * 16
        with patch("snackwell.reference.native.native_library", return_value=None):
            self.assertEqual(tensor_product(a, b, 4, 4, 4), python_product(a, b, 4, 4, 4))
        for args in ((a, b, 4, 4, 3), (a[:-1], b, 4, 4, 4), (a, b, -1, 4, 4)):
            with self.assertRaises(ValueError):
                tensor_product(*args)
