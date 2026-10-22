"""Optional PyTorch checks for training-only behavior and portable exports."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from snackwell.character_model import reference_forward
from snackwell.paths import ROOT
from tests.test_model import fixture


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU training dependency")
class ReferenceTrainingTests(unittest.TestCase):
    def test_feedforward_expansion_preserves_predictions_and_can_learn(self):
        import torch
        import torch.nn.functional as F

        from snackwell.model_expansion import expand_feedforward
        from snackwell.torch_model import ReferenceTransformer

        for layers in (1, 4):
            model = fixture(context=8, width=8, layers=layers, heads=2)
            original = copy.deepcopy(model)
            expanded = expand_feedforward(model, 4, 2171)
            ids = [1, 2, 3, 1, 2, 4, 0, 3]
            self.assertEqual(reference_forward(model, ids), reference_forward(expanded, ids))
            self.assertEqual(model, original)
            self.assertEqual(expanded, expand_feedforward(model, 4, 2171))
            network = ReferenceTransformer(expanded["config"], expanded["weights"])
            prefix = "" if layers == 1 else "block0_"
            optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
            for _ in range(2):
                optimizer.zero_grad()
                loss = F.cross_entropy(
                    network(torch.tensor([ids])).reshape(8, 5), torch.tensor(ids)
                )
                loss.backward()
                optimizer.step()
            p = network.parameters_by_name
            self.assertGreater(float(p[prefix + "w2"][16:].detach().abs().sum()), 0)
            self.assertGreater(float(p[prefix + "w1"].grad[:, 16:].abs().sum()), 0)
            with self.assertRaises(ValueError):
                expand_feedforward(expanded, 4, 2171)

    def test_dropout_is_training_only_and_export_matches_scalar_reference(self):
        import torch

        from snackwell.torch_model import ReferenceTransformer

        torch.set_num_threads(1)
        torch.manual_seed(1847)
        model = fixture(context=8, width=8, layers=2, heads=2)
        network = ReferenceTransformer(model["config"], model["weights"], dropout=0.2)
        ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
        network.train()
        self.assertFalse(torch.equal(network(ids), network(ids)))
        network.eval()
        with torch.no_grad():
            actual = network(ids)
            self.assertTrue(torch.equal(actual, network(ids)))
            portable = ReferenceTransformer(model["config"], model["weights"]).eval()
            self.assertTrue(torch.equal(actual, portable(ids)))
        expected = reference_forward(model, ids[0].tolist())["logits"]
        self.assertLess(max(abs(a - b) for a, b in zip(actual.flatten().tolist(), expected)), 1e-5)

    def test_invalid_training_settings_are_rejected(self):
        from snackwell.torch_model import ReferenceTransformer

        model = fixture()
        for settings in ({"dropout": float("nan")}, {"dropout": 1}, {"initialization": "unknown"}):
            with self.assertRaises(ValueError):
                ReferenceTransformer(model["config"], **settings)
        for multiplier in (True, 0, 3, 8):
            candidate = fixture(width=8, layers=2)
            candidate["config"]["ffn_multiplier"] = multiplier
            from snackwell.character_model import validate_checkpoint

            with self.assertRaises(ValueError):
                validate_checkpoint(candidate)
        self.assertEqual(len(fixture(width=128, ffn_multiplier=4)["weights"]["w1"]), 128 * 512)
        with self.assertRaises(ValueError):
            fixture(width=128, ffn_multiplier=8)

    def test_tail_objective_keeps_causal_prefix_and_weights_target_gradients(self):
        import torch
        import torch.nn.functional as F

        from snackwell.cli.train import training_objective

        logits = torch.tensor(
            [[[0.1, 0.3], [1.2, -0.1], [0.7, 1.1], [-0.8, 0.2]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, 0, 1]])
        all_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        self.assertAlmostEqual(
            float(training_objective(logits, targets, 4, 0.2).detach()), float(all_loss.detach())
        )
        loss = training_objective(logits, targets, 2, 0.2)
        loss.backward()
        expected = logits.detach().softmax(-1)
        expected.scatter_add_(-1, targets.unsqueeze(-1), -torch.ones(1, 4, 1, dtype=torch.float64))
        expected *= torch.tensor([0.05, 0.05, 0.45, 0.45], dtype=torch.float64)[None, :, None]
        self.assertTrue(torch.allclose(logits.grad, expected, atol=1e-12, rtol=0))

        # A tail target must still backpropagate through the earlier input tokens.
        from snackwell.torch_model import ReferenceTransformer

        model = fixture(context=4, width=4)
        network = ReferenceTransformer(model["config"], model["weights"])
        ids = torch.tensor([[0, 1, 2, 3]])
        training_objective(network(ids), torch.tensor([[1, 2, 3, 0]]), 1, 0).backward()
        self.assertGreater(
            float(network.parameters_by_name["token_embedding"].grad[0].abs().sum()), 0
        )

    def test_validation_scores_only_the_next_character_and_enforces_split_boundaries(self):
        import torch
        import torch.nn.functional as F

        from snackwell.cli.train import score_targets
        from snackwell.torch_model import ReferenceTransformer

        model = fixture(context=4, width=4)
        network = ReferenceTransformer(model["config"], model["weights"])
        data = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3])
        positions = [4, 6, 7]
        scored = score_targets(network, data, positions, 4, "cpu")
        expected = sum(
            float(F.cross_entropy(network(data[p - 4 : p][None])[:, -1], data[p : p + 1]))
            for p in positions
        ) / len(positions)
        self.assertAlmostEqual(scored["loss"], expected, places=6)
        self.assertEqual(scored["tokens"], 3)
        self.assertEqual(scored["target_positions"], positions)
        for bad in ([], [3], [8]):
            with self.assertRaises(ValueError):
                score_targets(network, data, bad, 4, "cpu")

    def test_host_trainer_exports_validation_only_and_protects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, out = root / "corpus.txt", root / "run"
            corpus.write_text("abcd abcd abcd\n" * 100)
            command = [
                sys.executable,
                str(ROOT / "scripts/train_reference_model.py"),
                "--data",
                str(corpus),
                "--out",
                str(out),
                "--steps",
                "2",
                "--width",
                "4",
                "--context",
                "4",
                "--layers",
                "1",
                "--heads",
                "1",
                "--batch-size",
                "2",
                "--evaluation-targets",
                "16",
                "--eval-every",
                "1",
                "--loss-tail",
                "2",
            ]
            run = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            checkpoint = (out / "checkpoint.json").read_bytes()
            metadata = json.loads(checkpoint)["training"]
            self.assertIsNone(metadata["test"])
            self.assertEqual(metadata["loss_tail"], 2)
            self.assertEqual(metadata["initial_validation"]["tokens"], 16)
            evaluation = json.loads((out / "evaluation.json").read_text())
            self.assertEqual(evaluation["split"], "validation 90%-95%")
            for example in evaluation["examples"]:
                self.assertGreaterEqual(
                    example["dataset_offset"], int(len(corpus.read_text()) * 0.9)
                )
                self.assertLess(example["dataset_offset"] + 4, int(len(corpus.read_text()) * 0.95))
            rerun = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("new directory", rerun.stderr)
            self.assertEqual(checkpoint, (out / "checkpoint.json").read_bytes())

    def test_invalid_tail_is_rejected_before_training(self):
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train_reference_model.py"), "--loss-tail", "0"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--loss-tail must be", run.stderr)


if __name__ == "__main__":
    unittest.main()
