"""Exercise real RTL gradients, SGD, graph checks, convergence and checkpoint safety."""

import copy
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, FMUL, FSUB, reference
from snackwell.reference.inference import gelu_tanh, verify_approx
from snackwell.reference.training import forward
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.training import (
    Shape,
    batch,
    checkpoint,
    dataset,
    evaluate,
    initialize,
    learning_rate_word,
    load_checkpoint,
    train_step,
    write_json,
)
from snackwell.training_runtime import TrainingRuntime


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/training"
        cls.output.mkdir(parents=True, exist_ok=True)
        cls.simulators = {
            name: compile_model_runner(cls.output / name, name) for name in ("icarus", "auto")
        }

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=self.output)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        simulator = self.simulators["auto"]
        self.session = (
            SimulatorSession(simulator) if simulator.backend == "verilator" else simulator
        )
        if isinstance(self.session, SimulatorSession):
            self.addCleanup(self.session.close)

    def runtime(self, name="kernels", executable=None):
        return TrainingRuntime(self.path / name, executable or self.session)

    def fixture(self):
        shape = Shape()
        x, y = batch(dataset(73)[0], 73, 0, count=4)
        return shape, initialize(shape, 91), x, y

    def test_sgd_separate_rounding_and_vector_chunk_boundary(self):
        rng = random.Random(91)
        x = [bits(rng.uniform(-2, 2)) for _ in range(260)]
        g = [bits(rng.uniform(-2, 2)) for _ in x]
        x[:4], g[:4] = [bits(1), 0x80000000, 1, 0x00800000], [0x3F800001, 0, 1, 0x007FFFFF]
        scale = value(0x3F7FFFFE)
        runtime = self.runtime()
        actual = runtime.vector("sgd", "sgd", x, g, scale)
        expected = [reference(FSUB, a, reference(FMUL, b, bits(scale))) for a, b in zip(x, g)]
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], 0)  # Fusing the multiply/subtract would leave a residual.
        self.assertEqual(len(runtime.reports), 2)
        dy = runtime.vector("dy", "mse_gradient", x, g, 1 / 260)
        self.assertEqual(
            dy, [reference(FMUL, reference(FSUB, a, b), bits(1 / 260)) for a, b in zip(x, g)]
        )

    def test_loss_and_bias_gradient_keep_reduction_order_across_chunks(self):
        rng = random.Random(17)
        x = [bits(rng.uniform(-3, 3)) for _ in range(36 * 20)]
        target = [bits(rng.uniform(-1, 1)) for _ in range(257)]
        runtime = self.runtime()
        loss = runtime.loss(x[:257], target)
        total = 0
        for a, b in zip(x, target):
            delta = reference(FSUB, a, b)
            total = reference(FADD, total, reference(FMUL, delta, delta))
        self.assertEqual(loss, [reference(FMUL, total, bits(1 / 514))])
        reduced = runtime.column_sum("bias", x, 36, 20)
        expected = [0] * 20
        for row in range(36):
            for col in range(20):
                expected[col] = reference(FADD, expected[col], x[row * 20 + col])
        self.assertEqual(reduced, expected)
        self.assertTrue(
            all(r["allocated_words"] <= 1024 and r["instructions"] <= 1024 for r in runtime.reports)
        )

    def test_gelu_backward_signs_zero_and_saturated_tails(self):
        values = [i / 4 for i in range(-40, 41)] + [-3e38, 3e38, -1e-20, 1e-20, 0, 10, -10]
        x = list(map(bits, values))
        g = [bits((-2, 0, 0.25, 1)[i % 4]) for i in range(len(x))]
        actual = self.runtime().vector("gelu_backward", "gelu_backward", x, g)
        epsilon = 1e-5
        expected = [
            value(b)
            * (
                (gelu_tanh(value(a) + epsilon) - gelu_tanh(value(a) - epsilon)) / (2 * epsilon)
                if abs(value(a)) < 10
                else (1 if value(a) > 0 else 0)
            )
            for a, b in zip(x, g)
        ]
        verify_approx(actual, expected, "gelu_backward")

    def test_all_parameter_and_input_gradients_against_finite_differences(self):
        shape, parameters, x, target = self.fixture()
        result = train_step(self.runtime(), parameters, x, target, shape, 0.2)
        p = {k: list(map(value, v)) for k, v in parameters.items()}
        inputs, targets = list(map(value, x)), list(map(value, target))
        epsilon = 1e-5

        def loss():
            return forward(p, inputs, targets, 4, 8, 16, 4)["loss"][0]

        for name, tensor in [*p.items(), ("input", inputs)]:
            differences = []
            for i, old in enumerate(tensor):
                tensor[i] = old + epsilon
                plus = loss()
                tensor[i] = old - epsilon
                minus = loss()
                tensor[i] = old
                differences.append((plus - minus) / (2 * epsilon))
            actual = (
                result["tensors"]["dx"] if name == "input" else result["tensors"]["gradients"][name]
            )
            verify_approx(actual, differences, "training")
        self.assertEqual(sum(map(len, p.values())), 212)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional independent autograd oracle")
    def test_rtl_gradients_against_pytorch_autograd(self):
        import torch
        import torch.nn.functional as F

        torch.set_num_threads(1)
        shape, parameters, x, target = self.fixture()
        result = train_step(self.runtime(), parameters, x, target, shape, 0.2)
        p = {
            name: torch.tensor(list(map(value, words)), dtype=torch.float64, requires_grad=True)
            for name, words in parameters.items()
        }
        inputs = torch.tensor(list(map(value, x)), dtype=torch.float64, requires_grad=True)
        targets = torch.tensor(list(map(value, target)), dtype=torch.float64).reshape(4, 4)
        z = inputs.reshape(4, 8) @ p["w1"].reshape(8, 16) + p["b1"]
        y = F.gelu(z, approximate="tanh") @ p["w2"].reshape(16, 4) + p["b2"]
        loss = ((y - targets) ** 2).mean() / 2
        loss.backward()
        for name, tensor in p.items():
            verify_approx(result["tensors"]["gradients"][name], tensor.grad.tolist(), "training")
        verify_approx(result["tensors"]["dx"], inputs.grad.tolist(), "training")
        verify_approx(result["tensors"]["loss"], [loss.item()], "training")

    def test_complete_step_matches_icarus_and_verilator(self):
        if self.simulators["auto"].backend != "verilator":
            self.skipTest("optional Verilator simulator not available")
        shape, parameters, x, target = self.fixture()
        results = [
            train_step(self.runtime(name, simulator), parameters, x, target, shape, 0.1)
            for name, simulator in (
                ("icarus", self.simulators["icarus"]),
                ("verilator", self.session),
            )
        ]
        self.assertEqual(results[0]["tensors"], results[1]["tensors"])
        self.assertEqual(results[0]["cycles"], results[1]["cycles"])
        self.assertEqual(results[0]["launches"], results[1]["launches"])

    def test_backward_matrix_splits_beyond_shared_memory(self):
        shape, rows = Shape(inputs=36, hidden=64, outputs=20), 36
        rng = random.Random(193)
        parameters = initialize(shape, 53)
        x = [bits(rng.uniform(-1, 1)) for _ in range(rows * shape.inputs)]
        target = [bits(rng.uniform(-1, 1)) for _ in range(rows * shape.outputs)]
        result = train_step(self.runtime(), parameters, x, target, shape, 0.1)
        self.assertTrue(result["verified"])
        self.assertGreater(result["launches"], 100)
        self.assertTrue(
            all(
                r["allocated_words"] <= 1024 and r["instructions"] <= 1024 for r in result["stages"]
            )
        )

    def test_whole_graph_check_rejects_wrong_backward_wiring_and_removes_stale_success(self):
        shape, parameters, x, target = self.fixture()
        original = copy.deepcopy(parameters)
        runtime = self.runtime()
        train_step(runtime, parameters, x, target, shape, 0.2)
        matmul = runtime.matmul

        def wrong_gradient(name, *args, **kwargs):
            words = matmul(name, *args, **kwargs)
            return [0] * len(words) if name == "grad_w1" else words

        with patch.object(runtime, "matmul", side_effect=wrong_gradient):
            with self.assertRaisesRegex(ValueError, "gradients.*w1"):
                train_step(runtime, parameters, x, target, shape, 0.2)
        self.assertEqual(parameters, original)
        self.assertFalse((runtime.output / "step.json").exists())

    def test_invalid_values_overflow_and_cancellation_never_mutate_weights(self):
        shape, parameters, x, target = self.fixture()
        original = copy.deepcopy(parameters)
        runtime = self.runtime()
        for bad in (float("nan"), float("inf"), 0, -1, 1e100, 1e-100, True):
            with self.subTest(rate=bad), self.assertRaises(ValueError):
                learning_rate_word(bad)
        for bad in (0x7F800000, 0xFF800000, 0x7FC00000, -1, True):
            with self.subTest(word=bad), self.assertRaises(ValueError):
                train_step(runtime, parameters, [bad, *x[1:]], target, shape, 0.2)
        self.assertEqual(runtime.reports, [])
        with patch.object(runtime, "vector", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                train_step(runtime, parameters, x, target, shape, 0.2)
        self.assertFalse((runtime.output / "step.json").exists())
        with self.assertRaises((ValueError, OverflowError)):
            train_step(runtime, parameters, [bits(3e38)] * len(x), target, shape, 0.2)
        self.assertEqual(parameters, original)
        self.assertFalse((runtime.output / "step.json").exists())

    def test_repeated_training_reduces_training_and_heldout_rtl_loss(self):
        shape, seed = Shape(), 1729
        parameters = initialize(shape, seed)
        train, heldout = dataset(seed)
        initial = [
            evaluate(self.runtime(f"before_{i}"), parameters, *data, shape)["loss"]
            for i, data in enumerate((train, heldout))
        ]
        for step in range(60):
            result = train_step(
                self.runtime(f"step_{step}"), parameters, *batch(train, seed, step), shape, 0.2
            )
            parameters = result["tensors"]["updated"]
        final = [
            evaluate(self.runtime(f"after_{i}"), parameters, *data, shape)["loss"]
            for i, data in enumerate((train, heldout))
        ]
        for before, after in zip(initial, final):
            self.assertLess(after, before * 0.5)

    def test_saved_checkpoint_resume_matches_uninterrupted_updates(self):
        shape, seed = Shape(), 73
        train, _ = dataset(seed)
        parameters = initialize(shape, seed)
        for step in range(3):
            result = train_step(
                self.runtime(f"before_{step}"), parameters, *batch(train, seed, step), shape, 0.2
            )
            parameters = result["tensors"]["updated"]
        path = self.path / "checkpoint.json"
        write_json(path, checkpoint(parameters, shape, 3, seed, 0.2))
        resumed = load_checkpoint(path)
        expected = train_step(
            self.runtime("uninterrupted"), parameters, *batch(train, seed, 3), shape, 0.2
        )
        actual = train_step(
            self.runtime("resumed"),
            resumed["parameter_words"],
            *batch(train, resumed["seed"], resumed["completed_steps"]),
            shape,
            resumed["learning_rate"],
        )
        self.assertEqual(actual["tensors"], expected["tensors"])
        self.assertEqual(actual["cycles"], expected["cycles"])

    def test_terminal_cancellation_keeps_last_completed_checkpoint(self):
        from snackwell.cli.training import main

        output = self.path / "interrupted"
        calls = 0

        def cancel_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return train_step(*args, **kwargs)

        with patch.object(sys, "argv", ["train", "--steps", "3", "--out", str(output)]):
            with patch("snackwell.cli.training.train_step", side_effect=cancel_second):
                with redirect_stdout(StringIO()), self.assertRaises(SystemExit) as stopped:
                    main()
        self.assertEqual(stopped.exception.code, 130)
        saved = load_checkpoint(output / "checkpoint.json")
        self.assertEqual(saved["completed_steps"], 1)
        self.assertFalse((output / "result.json").exists())
        self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "cancelled")
        full_step = json.loads((output / "step_00001/step.json").read_text())
        self.assertEqual(saved["parameter_words"], full_step["tensors"]["updated"])

    def test_terminal_command_rejects_existing_output_and_bad_checkpoints(self):
        output = self.path / "used"
        output.mkdir()
        (output / "result.json").write_text("existing evidence\n")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_snackwell.py"),
                "--steps",
                "1",
                "--out",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new directory", result.stderr)
        self.assertEqual((output / "result.json").read_text(), "existing evidence\n")
        for document in (
            {"format": "character-transformer"},
            {**checkpoint(initialize(Shape(), 1), Shape(), 0, 1, 0.2), "completed_steps": -1},
        ):
            path = self.path / "bad.json"
            write_json(path, document)
            with self.assertRaises(ValueError):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
