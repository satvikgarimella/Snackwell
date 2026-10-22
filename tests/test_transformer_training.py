"""Full-model derivatives, simulator parity, safe resume and inference compatibility."""

import copy
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from snackwell.character_model import load_checkpoint, reference_forward
from snackwell.model_runner import forward
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD
from snackwell.reference.fp32 import reference as fp32
from snackwell.reference.inference import stable_softmax, verify_approx
from snackwell.reference.transformer_training import reference
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.transformer_training import run_step
from snackwell.transformer_training_data import Corpus
from snackwell.transformer_training_runtime import TransformerTrainingRuntime
from tests.test_model import fixture


class TransformerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/transformer_training"
        cls.output.mkdir(parents=True, exist_ok=True)
        cls.simulators = {k: compile_model_runner(cls.output / k, k) for k in ("icarus", "auto")}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=self.output)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        simulator = self.simulators["auto"]
        self.session = (
            SimulatorSession(simulator) if simulator.backend == "verilator" else simulator
        )
        if isinstance(self.session, SimulatorSession):
            self.addCleanup(self.session.close)

    def runtime(self, name="rtl", executable=None):
        return TransformerTrainingRuntime(self.path / name, executable or self.session)

    def test_layernorm_backward_zero_variance_and_shared_memory_chunks(self):
        rng = random.Random(113)
        for cols in (4, 32, 128):
            rows = 8
            x = [bits(rng.uniform(-2, 2)) for _ in range(rows * cols)]
            x[:cols] = [bits(1)] * cols
            dy = [bits(rng.uniform(-1, 1)) for _ in x]
            gamma = [bits(rng.uniform(0.5, 1.5)) for _ in range(cols)]
            runtime = self.runtime(f"norm_{cols}")
            dx, dg, db = runtime.norm_backward("norm", x, dy, gamma, rows, cols, 1e-5)
            self.assertEqual((len(dx), len(dg), len(db)), (rows * cols, cols, cols))
            self.assertTrue(
                all(
                    r["verified"] and r["allocated_words"] <= 1024 and r["instructions"] <= 1024
                    for r in runtime.reports
                )
            )

    def test_softmax_backward_masks_future_and_saturated_first_row(self):
        rng = random.Random(83)
        for cols in (8, 128):
            p = [
                bits(v)
                for r in range(8)
                for v in stable_softmax(
                    [rng.uniform(-5, 5) if c <= r else -math.inf for c in range(cols)]
                )
            ]
            dy = [bits(rng.uniform(-1, 1)) for _ in p]
            result = self.runtime(f"softmax_{cols}").softmax_backward(
                "softmax", p, dy, 8, cols, 0.5
            )
            self.assertEqual(list(map(value, result[:cols])), [0.0] * cols)
            for r in range(8):
                self.assertTrue(
                    all(value(x) == 0 for x in result[r * cols + r + 1 : (r + 1) * cols])
                )

    def test_cross_entropy_remains_finite_when_target_probability_underflows(self):
        for cols in (4, 128, 512):
            rows = [
                [1000.0, -1000.0] + [-math.inf] * (cols - 2),
                [0.0] * cols,
                [float(i) / 4 for i in range(cols)],
                [1000 + float(i) / 4 for i in range(cols)],
            ]
            targets = [1, 0, cols - 1, cols - 1]
            runtime = self.runtime(f"ce_{cols}")
            loss = runtime.cross_entropy([bits(x) for row in rows for x in row], targets, 4, cols)
            expected = []
            for row, t in zip(rows, targets):
                maximum = max(row)
                expected.append(
                    maximum - row[t] + math.log(math.fsum(math.exp(x - maximum) for x in row))
                )
            verify_approx(loss, [math.fsum(expected) / 4], "cross_entropy")
            self.assertTrue(math.isfinite(value(loss[0])))
        with self.assertRaises(ValueError):
            self.runtime("invalid_target").cross_entropy([0] * 16, [4] * 4, 4, 4)

    def test_every_parameter_gradient_against_finite_differences(self):
        model, ids, targets = fixture(), [1, 1, 2, 1], [1, 2, 1, 3]
        result, _ = run_step(self.runtime(), model, ids, targets, 0.01)
        perturbed = copy.deepcopy(model)
        epsilon = 1e-5
        for name, weights in perturbed["weights"].items():
            numerical = []
            for i, old in enumerate(weights):
                weights[i] = old + epsilon
                plus = reference(perturbed, ids, targets)["observed"]["loss"][0]
                weights[i] = old - epsilon
                minus = reference(perturbed, ids, targets)["observed"]["loss"][0]
                weights[i] = old
                numerical.append((plus - minus) / (2 * epsilon))
            verify_approx(result["tensors"]["gradients"][name], numerical, "transformer_training")
        dx = result["tensors"]["embedding_gradient"]
        table = result["tensors"]["gradients"]["token_embedding"]
        repeated = [0] * 4
        for row in (0, 1, 3):
            repeated = [fp32(FADD, a, b) for a, b in zip(repeated, dx[row * 4 : (row + 1) * 4])]
        self.assertEqual(table[4:8], repeated)
        self.assertEqual(table[:4] + table[12:], [0] * 12)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional independent autograd oracle")
    def test_multilayer_multihead_gradients_match_pytorch(self):
        import torch
        import torch.nn.functional as F

        from snackwell.torch_model import ReferenceTransformer

        torch.set_num_threads(1)
        for layers, heads, width, context, multiplier in (
            (1, 1, 4, 4, 2),
            (2, 2, 8, 8, 2),
            (4, 4, 16, 4, 2),
            (2, 2, 8, 8, 4),
            (1, 4, 128, 4, 4),
        ):
            model = fixture(
                context=context, width=width, layers=layers, heads=heads, ffn_multiplier=multiplier
            )
            ids, targets = (
                [i % 3 + 1 for i in range(context)],
                [(i + 1) % 5 for i in range(context)],
            )
            result, _ = run_step(
                self.runtime(f"model_{layers}_{multiplier}"), model, ids, targets, 0.01
            )
            expected = reference(model, ids, targets)
            network = ReferenceTransformer(model["config"], model["weights"]).double()
            loss = F.cross_entropy(
                network(torch.tensor([ids])).reshape(context, 5), torch.tensor(targets)
            )
            loss.backward()
            self.assertAlmostEqual(loss.item(), expected["observed"]["loss"][0], places=12)
            for name, parameter in network.parameters_by_name.items():
                gradient = parameter.grad.flatten().tolist()
                self.assertLess(
                    max(abs(a - b) for a, b in zip(gradient, expected["gradients"][name])), 1e-12
                )
                verify_approx(
                    result["tensors"]["gradients"][name], gradient, "transformer_training"
                )

    def test_complete_training_step_matches_both_simulators(self):
        if self.simulators["auto"].backend != "verilator":
            self.skipTest("optional Verilator simulator")
        results = [
            run_step(self.runtime(name, executable), fixture(), [1, 2, 1, 3], [2, 1, 3, 4], 0.01)[0]
            for name, executable in (
                ("icarus", self.simulators["icarus"]),
                ("verilator", self.session),
            )
        ]
        self.assertEqual(results[0]["tensors"], results[1]["tensors"])
        self.assertEqual(results[0]["cycles"], results[1]["cycles"])
        self.assertEqual(results[0]["launches"], results[1]["launches"])

    def test_updated_checkpoint_runs_through_existing_inference(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        ids, targets = [1, 2, 1, 3, 4, 1, 2, 3], [2, 1, 3, 4, 1, 2, 3, 1]
        _, updated = run_step(self.runtime(), model, ids, targets, 0.01)
        actual = forward(updated, ids, self.path / "inference", self.session)
        expected = reference_forward(updated, ids)
        self.assertTrue(actual["verified"])
        self.assertEqual(actual["predictions"], expected["predictions"])

    def test_wrong_embedding_backward_is_rejected_without_mutating_model(self):
        model = fixture()
        original = copy.deepcopy(model)
        runtime = self.runtime()
        run_step(runtime, model, [1, 1, 2, 1], [1, 2, 1, 3], 0.01)
        with patch.object(runtime, "embedding_backward", return_value=[0] * 20):
            with self.assertRaisesRegex(ValueError, "gradient token_embedding"):
                run_step(runtime, model, [1, 1, 2, 1], [1, 2, 1, 3], 0.01)
        self.assertEqual(model, original)
        self.assertFalse((runtime.output / "result.json").exists())

    def test_invalid_inputs_and_cancelled_backward_do_not_commit(self):
        model, ids, targets = fixture(), [1, 1, 2, 1], [1, 2, 1, 3]
        original = copy.deepcopy(model)
        runtime = self.runtime()
        for bad_ids, bad_targets in ((ids[:-1], targets), (ids, [5] * 4), (ids, [True] * 4)):
            with self.assertRaises(ValueError):
                run_step(runtime, model, bad_ids, bad_targets, 0.01)
        invalid = copy.deepcopy(model)
        invalid["weights"]["wq"][0] = float("nan")
        with self.assertRaises(ValueError):
            run_step(runtime, invalid, ids, targets, 0.01)
        self.assertEqual(runtime.reports, [])
        with patch.object(runtime, "norm_backward", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_step(runtime, model, ids, targets, 0.01)
        self.assertEqual(model, original)
        self.assertFalse((runtime.output / "result.json").exists())

    def run_cli(self, name, *args):
        from snackwell.cli.train_transformer import main

        data = self.path / "corpus.txt"
        data.write_text("abcd bcda cdab dabc\n" * 100)
        argv = [
            "train",
            "--data",
            str(data),
            "--evaluation-windows",
            "1",
            "--out",
            str(self.path / name),
            *args,
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            main()
        return self.path / name

    def test_terminal_resume_reproduces_uninterrupted_weights_and_corpus_binding(self):
        options = ("--width", "4", "--context", "4", "--heads", "1", "--layers", "1")
        first = self.run_cli("first", *options, "--steps", "2")
        resumed = self.run_cli(
            "resumed", "--resume", str(first / "checkpoint.json"), "--steps", "1"
        )
        whole = self.run_cli("whole", *options, "--steps", "3")
        actual, expected = (
            load_checkpoint(resumed / "checkpoint.json"),
            load_checkpoint(whole / "checkpoint.json"),
        )
        self.assertEqual(actual["weights"], expected["weights"])
        self.assertEqual(actual["training"], expected["training"])
        report = json.loads((resumed / "result.json").read_text())
        self.assertEqual(
            report["validation_improved"],
            report["final_validation_loss"] < report["initial_validation_loss"],
        )
        self.assertFalse(report["inference_checkpoint_promoted"])
        bad_data = self.path / "other.txt"
        bad_data.write_text("dcba " * 1000)
        from snackwell.transformer_training_data import resume_state

        with self.assertRaisesRegex(ValueError, "corpus does not match"):
            resume_state(actual, Corpus(bad_data, 4, actual["config"]["vocabulary"]))

    def test_terminal_cancellation_preserves_last_successful_step(self):
        from snackwell.cli.train_transformer import run_step as real_step

        count = 0

        def cancel(*args, **kwargs):
            nonlocal count
            if len(args) >= 5 and args[4] is not None:
                count += 1
                if count == 2:
                    raise KeyboardInterrupt
            return real_step(*args, **kwargs)

        with patch("snackwell.cli.train_transformer.run_step", side_effect=cancel):
            with self.assertRaises(SystemExit) as stopped:
                self.run_cli(
                    "cancel", "--width", "4", "--context", "4", "--heads", "1", "--steps", "3"
                )
        self.assertEqual(stopped.exception.code, 130)
        saved = load_checkpoint(self.path / "cancel/checkpoint.json")
        self.assertEqual(saved["training"]["completed_steps"], 1)
        self.assertFalse((self.path / "cancel/result.json").exists())
        self.assertEqual(
            json.loads((self.path / "cancel/failure.json").read_text())["completed_steps"], 1
        )


if __name__ == "__main__":
    unittest.main()
