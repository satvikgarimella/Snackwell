"""Classifier and embedding launches beyond 128 entries, within the same 4 KiB SM."""

import math
import unittest

from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, reference
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.transformer_training_runtime import TransformerTrainingRuntime


class LargeVocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/large_vocabulary"
        cls.simulator = compile_model_runner(cls.output)
        cls.session = None
        if cls.simulator.backend == "verilator":
            cls.session = SimulatorSession(cls.simulator)
            cls.execution = cls.session.__enter__()
        else:
            cls.execution = cls.simulator

    @classmethod
    def tearDownClass(cls):
        if cls.session is not None:
            cls.session.__exit__(None, None, None)

    def bounds(self, runtime):
        self.assertTrue(all(r["allocated_words"] <= 1024 for r in runtime.reports))
        self.assertTrue(all(r["instructions"] <= 1024 for r in runtime.reports))

    def test_high_embedding_ids_and_partial_feature_lanes(self):
        for vocabulary in (256, 512):
            for rows in (1, 4, 32):
                width = 8
                ids = [vocabulary - 1 - r for r in range(rows)]
                table = [bits((i % 971 - 400) / 32) for i in range(vocabulary * width)]
                position = [bits((i - 15) / 128) for i in range(rows * width)]
                runtime = ModelRuntime(self.output / f"lookup_{vocabulary}_{rows}", self.execution)
                actual = runtime.lookup(ids, table, position, width, vocabulary)
                expected = [
                    reference(FADD, table[t * width + c], position[r * width + c])
                    for r, t in enumerate(ids)
                    for c in range(width)
                ]
                self.assertEqual(actual, expected)
                self.bounds(runtime)

    def test_large_softmax_mask_argmax_and_sampling(self):
        for cols in (256, 512):
            runtime = ModelRuntime(self.output / f"classifier_{cols}", self.execution)
            valid = cols - 7
            x = [bits((c - valid) / 32 + r / 8) for r in range(4) for c in range(cols)]
            masked = runtime.rows("mask", "vocabulary_mask", x, 4, cols, vocabulary=valid)
            probabilities = runtime.rows("softmax", "softmax", masked, 4, cols, temperature=0.7)
            selected = runtime.rows("argmax", "argmax", probabilities, 4, cols)
            self.assertEqual(selected, [valid - 1] * 4)
            for row in range(4):
                p = probabilities[row * cols : (row + 1) * cols]
                self.assertEqual(p[valid:], [0] * (cols - valid))
                self.assertAlmostEqual(sum(map(value, p)), 1.0, places=5)
            draws = [bits(v) for v in (0.0, 0.25, 0.5, 1.0 - 2**-24)]
            sampled = runtime.rows("sample", "categorical", probabilities, 4, cols, draws=draws)
            self.assertTrue(all(0 <= token < valid for token in sampled))
            self.assertEqual(sampled[-1], valid - 1)
            self.assertEqual(runtime.rows("ties", "argmax", [bits(1)] * cols, 1, cols), [0])
            self.bounds(runtime)

    def test_cross_entropy_covers_uniform_maximum_sum_and_underflow(self):
        for cols in (256, 512):
            runtime = TransformerTrainingRuntime(self.output / f"loss_{cols}", self.execution)
            loss = runtime.cross_entropy([0] * cols, [cols - 1], 1, cols)
            self.assertAlmostEqual(value(loss[0]), math.log(cols), places=5)
            logits = [bits(1000), bits(-1000)] + [bits(-math.inf)] * (cols - 2)
            loss = runtime.cross_entropy(logits, [1], 1, cols)
            self.assertAlmostEqual(value(loss[0]), 2000.0, places=4)
            p = [bits(1 / cols)] * cols
            derivative = runtime.classifier_backward(p, [cols - 1], cols)
            self.assertEqual(derivative[:-1], p[:-1])
            self.assertEqual(value(derivative[-1]), 1 / cols - 1)
            self.bounds(runtime)
