"""Numerical checks and memory bounds for streamed pretrained-model kernels."""

import math
import random
import unittest

from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.pretrained_runtime import PretrainedRuntime
from snackwell.reference.fp32 import FADD, FMUL, reference
from snackwell.reference.inference import stable_softmax, verify_approx
from snackwell.simulator import SimulatorSession, compile_model_runner


class PretrainedKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/pretrained_kernels"
        simulator = compile_model_runner(cls.output, compact=True)
        cls.session = SimulatorSession(simulator) if simulator.backend == "verilator" else None
        cls.execution = cls.session.__enter__() if cls.session else simulator

    @classmethod
    def tearDownClass(cls):
        if cls.session:
            cls.session.__exit__(None, None, None)

    def runtime(self, name, executable=None):
        return PretrainedRuntime(self.output / name, executable or self.execution)

    def bounds(self, runtime):
        self.assertTrue(runtime.reports)
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))

    def test_rmsnorm_crosses_chunk_boundaries_without_power_of_two_padding(self):
        rng = random.Random(541)
        runtime = self.runtime("rmsnorm")
        for width in (4, 256, 384, 512, 576):
            for zero in (False, True):
                x = [0 if zero else bits(rng.uniform(-5, 5)) for _ in range(width)]
                gamma = [bits(rng.uniform(0.3, 2)) for _ in x]
                actual = runtime.rmsnorm("norm", x, gamma, 1e-5)
                inv = 1 / math.sqrt(math.fsum(value(a) ** 2 for a in x) / width + 1e-5)
                expected = [value(a) * inv * value(g) for a, g in zip(x, gamma)]
                verify_approx(actual, expected, "layernorm")
        self.bounds(runtime)

    def test_swiglu_extremes_and_partial_vector(self):
        runtime = self.runtime("swiglu")
        x = [bits(a) for a in (-1000, -80, -10, -1, -0.0, 0, 1, 10, 80, 1000)] * 68 + [
            bits(0.001)
        ] * 2
        other = [bits((i % 13 - 6) / 8) for i in range(len(x))]
        actual = runtime.vector("gate", x, "swiglu", other=other)
        expected = []
        for a, b in zip(x, other):
            a, b = value(a), value(b)
            e = math.exp(-abs(a))
            expected.append(a * (1 if a >= 0 else e) / (1 + e) * b)
        verify_approx(actual, expected, "pretrained_swiglu")
        self.bounds(runtime)

    def test_rotary_pairs_and_residual_use_exact_separate_fp32_operations(self):
        rng = random.Random(8)
        runtime = self.runtime("rotary")
        for position in (0, 1, 127, 511):
            x = [bits(rng.uniform(-10, 10)) for _ in range(192)]
            other, cos, sin = [], [], []
            for i in range(len(x)):
                head, feature = divmod(i, 64)
                other.append(x[head * 64 + (feature + 32) % 64])
                angle = position / 10000 ** ((feature % 32) / 32)
                cos.append(bits(math.cos(angle)))
                sin.append(bits(math.sin(angle) * (-1 if feature < 32 else 1)))
            actual = runtime.vector("rope", x, "rope", other=other, cos=cos, sin=sin)
            expected = [
                reference(FADD, reference(FMUL, a, c), reference(FMUL, b, s))
                for a, b, c, s in zip(x, other, cos, sin)
            ]
            self.assertEqual(actual, expected)
            added = runtime.vector("residual", actual, "add", other=x)
            self.assertEqual(added, [reference(FADD, a, b) for a, b in zip(actual, x)])
        self.bounds(runtime)

    def test_large_vocabulary_softmax_and_first_argmax_tie(self):
        runtime = self.runtime("vocabulary")
        for count in (4019, 49152):
            x = [bits(((i * 17) % 97 - 48) / 8) for i in range(count)]
            x[511] = x[-1] = bits(8)
            actual = runtime.softmax("softmax", x, 0.7)
            verify_approx(actual, stable_softmax([value(w) / 0.7 for w in x]), "softmax")
            self.assertEqual(runtime.argmax("argmax", x), 511)
            uniform = runtime.softmax("uniform", [0] * count)
            self.assertTrue(all(w == uniform[0] for w in uniform))
            self.assertAlmostEqual(math.fsum(map(value, uniform)), 1.0, places=5)
        self.bounds(runtime)

    def test_sampling_boundaries_across_pages_and_final_nonzero_fallback(self):
        runtime = self.runtime("sampling")
        weights = [0] * 1031
        weights[0], weights[512], weights[1024] = map(bits, (0.25, 0.5, 0.25))
        for draw, expected in (
            (0, 0),
            (0.25 - 2**-24, 0),
            (0.25, 512),
            (0.75, 1024),
            (1 - 2**-24, 1024),
        ):
            self.assertEqual(runtime.sample("select", weights, draw), expected)
        with self.assertRaises(ValueError):
            runtime.sample("zero", [0] * 4, 0)
        with self.assertRaises(ValueError):
            runtime.sample("one", weights, 1)
        self.bounds(runtime)

    def test_embedding_page_and_feature_tails_preserve_raw_words(self):
        runtime = self.runtime("embedding")
        width, vocabulary = 36, 4019
        table = [bits((i % 193 - 91) / 16) for i in range(width * vocabulary)]
        table[0] = 0x80000000
        for token in (0, 15, 16, 511, 4018):
            self.assertEqual(
                runtime.embedding(token, table, width, vocabulary),
                table[token * width : (token + 1) * width],
            )
        self.bounds(runtime)

    def test_icarus_and_verilator_match_new_nonlinear_kernels(self):
        simulator = compile_model_runner(self.output / "icarus", "icarus")
        first, second = self.runtime("parity_v"), self.runtime("parity_i", simulator)
        x = [bits(x) for x in (-90, -3, -0.0, 0, 0.02, 0.5, 3, 90)]
        for runtime in (first, second):
            runtime.parity = [
                runtime.rmsnorm("norm", x, [bits(1)] * len(x), 1e-6),
                runtime.vector("gate", x, "swiglu", other=[bits(0.5)] * len(x)),
                runtime.softmax("softmax", x),
                runtime.sample("sample", [bits(0.25)] * 4, 0.5),
            ]
            self.bounds(runtime)
        self.assertEqual(first.parity, second.parity)


if __name__ == "__main__":
    unittest.main()
