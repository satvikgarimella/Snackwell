"""Wider transformer features still fit bounded launches on the same SM."""

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from snackwell.kv_cache import Generation
from snackwell.model_runner import forward
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.inference import verify_approx
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.transformer_training import run_step
from snackwell.transformer_training_runtime import TransformerTrainingRuntime
from tests.test_model import fixture


class WideTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / "build/tests/wide_transformer"
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.simulator = compile_model_runner(cls.root, "auto", compact=True)

    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.execution = (
            SimulatorSession(self.simulator)
            if self.simulator.backend == "verilator"
            else self.simulator
        )
        if isinstance(self.execution, SimulatorSession):
            self.addCleanup(self.execution.close)

    def runtime(self, name):
        return TransformerTrainingRuntime(self.path / name, self.execution)

    def assert_bounded(self, reports):
        self.assertTrue(reports)
        self.assertTrue(all(r["verified"] for r in reports))
        self.assertLessEqual(max(r["allocated_words"] for r in reports), 1024)
        self.assertLessEqual(max(r["instructions"] for r in reports), 1024)

    def test_normalization_and_residuals_preserve_rows_and_original_gradients(self):
        rng = random.Random(2727)
        cols, rows = 256, 8
        x = [bits(rng.uniform(-2, 2)) for _ in range(rows * cols)]
        x[:cols] = [bits(1.0)] * cols
        dy = [bits(rng.uniform(-1, 1)) for _ in x]
        gamma = [bits(rng.uniform(0.5, 1.5)) for _ in range(cols)]
        beta = [bits(rng.uniform(-0.2, 0.2)) for _ in range(cols)]
        original = (x[:], dy[:], gamma[:])
        runtime = self.runtime("rows")
        for count in (1, rows):
            a, b = x[: count * cols], dy[: count * cols]
            runtime.rows("residual", "residual_add", a, count, cols, other=b)
            runtime.rows("norm", "layernorm", a, count, cols, gamma=gamma, beta=beta)
        dx, dg, db = runtime.norm_backward("backward", x, dy, gamma, rows, cols, 1e-5)
        self.assertEqual((len(dx), len(dg), len(db)), (len(x), cols, cols))
        self.assertEqual((x, dy, gamma), original)
        self.assert_bounded(runtime.reports)

        if self.simulator.backend == "verilator":
            icarus = compile_model_runner(self.path / "icarus", "icarus")
            other = TransformerTrainingRuntime(self.path / "icarus_rows", icarus)
            expected = other.norm_backward("backward", x[:cols], dy[:cols], gamma, 1, cols, 1e-5)
            actual = runtime.norm_backward("one_row", x[:cols], dy[:cols], gamma, 1, cols, 1e-5)
            self.assertEqual(actual, expected)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional independent autograd")
    def test_wide_training_gradients_updates_and_cached_inference(self):
        import torch
        import torch.nn.functional as F

        from snackwell.torch_model import ReferenceTransformer

        torch.set_num_threads(1)
        model = fixture(context=4, width=256, heads=4, ffn_multiplier=4)
        ids, targets = [1, 2, 1, 3], [2, 1, 3, 4]
        network = ReferenceTransformer(model["config"], model["weights"]).double()
        loss = F.cross_entropy(network(torch.tensor([ids])).reshape(4, 5), torch.tensor(targets))
        loss.backward()
        runtime = self.runtime("training")
        result, updated = run_step(runtime, model, ids, targets, 0.001)
        for name, p in network.parameters_by_name.items():
            words = result["tensors"]["gradients"][name]
            verify_approx(words, p.grad.flatten().tolist(), "transformer_training")
            expected = torch.tensor(model["weights"][name]) - torch.tensor(0.001) * torch.tensor(
                [value(word) for word in words]
            )
            self.assertEqual(
                [bits(x) for x in expected.tolist()], [bits(x) for x in updated["weights"][name]]
            )
        self.assert_bounded(runtime.reports)

        generation = Generation(updated, "abac")
        phases = []
        for step in range(3):
            got = generation.next(self.path / f"cached_{step}", self.execution, draw=0.6)
            full = forward(
                updated,
                got["input_ids"],
                self.path / f"full_{step}",
                self.execution,
                draw=0.6,
                last_only=True,
            )
            self.assertEqual(got["probabilities"][-5:], full["probabilities"][-5:])
            self.assertEqual(got["selection"], full["selection"])
            phases.append(got["cache"]["phase"])
        self.assertEqual(phases, ["prefill", "rebuild", "decode"])


if __name__ == "__main__":
    unittest.main()
