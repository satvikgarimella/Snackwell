import json
import random
import subprocess
import sys
import unittest

from snackwell.assembler import assemble
from snackwell.inference_runner import execute_case
from snackwell.kernels.inference import build_case
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.simulator import compile_runner


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/transformer_python"
        cls.executable = compile_runner(cls.output)

    def run_case(self, label, **kwargs):
        case = build_case("transformer", **kwargs)
        report = execute_case(case, self.output / label, self.executable)
        self.assertEqual(report["launches"], 21)
        self.assertTrue(report["verified"])
        self.assertTrue(all(s["verified"] for s in report["stages"]))
        self.assertIn("end_to_end_accuracy", report)
        self.assertEqual(report["cycles"], sum(s["cycles"] for s in report["stages"]))
        for stage in case.stages:
            self.assertLessEqual(len(assemble(stage.source)), 1024)
        return case, report

    def test_nonzero_parameters_shapes_and_masks(self):
        rng = random.Random(0xB10C)
        for rows, width, causal in ((4, 4, True), (4, 4, False), (8, 4, True), (4, 8, False)):
            template = build_case("transformer", rows=rows, cols=width)
            inputs = {"causal": causal}
            for name, (_, count) in template.layout.regions.items():
                if name == "x" or name.startswith(("w", "b", "gamma")):
                    offset = 1 if name.startswith("gamma") else 0
                    inputs[name] = [offset + rng.uniform(-0.3, 0.3) for _ in range(count)]
            # Scratch buffers are excluded; every trainable matrix/bias and
            # both distinct affine norms are nontrivial in these tests.
            self.run_case(f"random_{rows}_{width}_{causal}", rows=rows, cols=width, inputs=inputs)

    def test_zero_branches_preserve_residual_exactly(self):
        x = [(-1) ** i * (i + 1) / 8 for i in range(16)]
        _, report = self.run_case(
            "residual_identity", inputs={"x": x, "wo": [0] * 16, "w2": [0] * 32}
        )
        self.assertEqual(report["output_bits"], [f"{bits(a):08x}" for a in x])

    def test_causal_first_row_independent_of_future_inputs(self):
        case, first = self.run_case("causal_original")
        start, count = case.layout.regions["x"]
        x = [value(w) for w in case.layout.memory[start : start + count]]
        x[4:] = [a + 3 for a in x[4:]]
        _, changed = self.run_case("causal_changed", inputs={"x": x})
        self.assertEqual(first["output_bits"][:4], changed["output_bits"][:4])
        self.assertNotEqual(first["output_bits"][4:], changed["output_bits"][4:])

    def test_end_to_end_check_catches_missing_residual(self):
        case = build_case("transformer")
        final = case.stages[-1]
        # Both the program and its local stage contract are changed together.
        # Only the independent complete-block reference can catch this error.
        # Replace the last residual sum by GELU of the projected branch,
        # which writes the correct output region.
        from snackwell.kernels.inference import elementwise

        final.source = elementwise(case.layout, "gelu", 4, 4, "projected", "out")
        final.operation, final.inputs = "gelu", ("projected",)
        with self.assertRaisesRegex(ValueError, "transformer output"):
            execute_case(case, self.output / "missing_residual", self.executable)
        self.assertFalse((self.output / "missing_residual/result.json").exists())

    def test_invalid_inputs_and_capacity(self):
        for kwargs in (
            {"inputs": {"wq": [1] * 15}},
            {"inputs": {"beta2": [float("nan")] * 4}},
            {"inputs": {"epsilon": 0}},
            {"inputs": {"causal": 1}},
            {"inputs": {"unknown": []}},
            {"inputs": []},
            {"rows": 16, "cols": 16},
            {"rows": 16, "cols": 4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, OverflowError)):
                build_case("transformer", **kwargs)

    def test_cli_and_stale_result_removal(self):
        output = self.output / "cli with spaces"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_kernel.py"),
            "transformer",
            "--out",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((output / "result.json").read_text())["verified"])
        result = subprocess.run(
            command + ["--cols", "16"], capture_output=True, text=True, timeout=30
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
