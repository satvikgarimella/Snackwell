import json
import math
import random
import subprocess
import sys
import unittest

from snackwell.assembler import OPCODES, assemble
from snackwell.inference_runner import execute_case
from snackwell.kernels.inference import NAMES, build_case
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.inference import verify_approx, verify_stage
from snackwell.simulator import compile_runner, read_hex, run_program


class InferenceKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/inference_python"
        cls.executable = compile_runner(cls.output)

    def run_case(self, label, name, **kwargs):
        case = build_case(name, **kwargs)
        report = execute_case(case, self.output / label, self.executable)
        self.assertTrue(report["verified"])
        self.assertEqual(report["cycles"], sum(s["cycles"] for s in report["stages"]))
        return case, report

    def test_every_default_kernel(self):
        for name in NAMES:
            with self.subTest(kernel=name):
                self.run_case("default_" + name, name)

    def test_tiled_matmul_shapes_and_partial_tile_accumulation(self):
        rng = random.Random(0x71E)
        for size in (4, 8, 16):
            identity = [int(r == c) for r in range(size) for c in range(size)]
            a = [rng.randrange(-32, 33) / 8 for _ in range(size * size)]
            _, report = self.run_case(
                f"identity_{size}", "tiled_matmul", size=size, inputs={"a": a, "b": identity}
            )
            self.assertEqual(report["output_bits"], [f"{bits(x):08x}" for x in a])
            self.assertEqual(
                report["stages"][0]["profile"]["opcode_counts"]["TMUL"], (size // 4) ** 3
            )
            profile = report["stages"][0]["profile"]
            self.assertEqual(profile["store_words"], size * size)
            self.assertEqual(profile["load_words"], (size // 4) ** 3 * 32)
            _, report = self.run_case(
                f"ones_{size}",
                "tiled_matmul",
                size=size,
                inputs={"a": [1] * (size * size), "b": [1] * (size * size)},
            )
            self.assertEqual(report["output_bits"], [f"{bits(size):08x}"] * (size * size))
        # Nonuniform terms in every K tile catch dropped or overwritten partials.
        for case in range(3):
            self.run_case(
                f"tile_random_{case}",
                "tiled_matmul",
                size=8,
                inputs={
                    key: [rng.randrange(-100, 101) / 16 for _ in range(64)] for key in ("a", "b")
                },
            )

    def test_basic_kernels_larger_shapes(self):
        rng = random.Random(0xB451C)
        for name in ("bias_add", "residual_add", "relu", "row_sum", "row_max"):
            for rows, cols in ((4, 8), (8, 4), (16, 16)):
                data = {"x": [rng.uniform(-10, 10) for _ in range(rows * cols)]}
                if name in ("bias_add", "residual_add"):
                    data["b"] = [
                        rng.uniform(-1, 1)
                        for _ in range(cols if name == "bias_add" else rows * cols)
                    ]
                self.run_case(f"{name}_{rows}_{cols}", name, rows=rows, cols=cols, inputs=data)

    def test_gelu_dense_grid_and_saturation(self):
        data = [-12 + i * 24 / 255 for i in range(256)]
        self.run_case("gelu_grid", "gelu", rows=16, cols=16, inputs={"x": data})
        data = [-1e30, -100, -10, -8, -3, -1, -1e-20, -0.0, 0.0, 1e-20, 1, 3, 8, 10, 100, 1e30]
        _, report = self.run_case("gelu_extremes", "gelu", inputs={"x": data})
        self.assertEqual(value(int(report["output_bits"][-1], 16)), value(bits(1e30)))

    def test_softmax_offsets_tails_and_masking(self):
        rng = random.Random(0x50F7)
        cases = [
            [0.0] * 16,
            [1e20] * 16,
            [1000, 1001, 1002, 1003] * 4,
            [-1000, -999, -998, -997] * 4,
            [0, -79.9, -80, -80.1] * 4,
            [1e30, -1e30, 0, -100] * 4,
            [0, -math.inf, -math.inf, -math.inf] * 4,
        ]
        for i, data in enumerate(cases):
            self.run_case(f"softmax_edge_{i}", "softmax", inputs={"x": data})
        for i in range(5):
            self.run_case(
                f"softmax_random_{i}",
                "softmax",
                rows=4,
                cols=16,
                inputs={"x": [rng.uniform(-40, 40) for _ in range(64)]},
            )
        self.run_case("softmax_16", "softmax", rows=16, cols=16)

    def test_layernorm_affine_constant_and_scale(self):
        rng = random.Random(0x1A9E)
        for size in (4, 8, 16):
            self.run_case(
                f"layernorm_{size}",
                "layernorm",
                rows=4,
                cols=size,
                inputs={
                    "x": [rng.uniform(-10, 10) for _ in range(4 * size)],
                    "gamma": [rng.uniform(-2, 2) for _ in range(size)],
                    "beta": [rng.uniform(-1, 1) for _ in range(size)],
                },
            )
        for case, scale in enumerate((0, 1e-15, 1e-3, 1, 1e10)):
            self.run_case(
                f"layernorm_scale_{case}",
                "layernorm",
                inputs={"x": [scale * (i - 8) for i in range(16)]},
            )
        self.run_case(
            "layernorm_small_epsilon",
            "layernorm",
            inputs={"x": [1e-10 * (i - 8) for i in range(16)], "epsilon": 1e-30},
        )
        self.run_case("layernorm_16", "layernorm", rows=16, cols=16)

    def test_linear_pipeline_and_attention_shapes(self):
        for size in (4, 16):
            self.run_case(f"linear_{size}", "linear", size=size)
        rng = random.Random(0xA77)
        for rows, cols in ((4, 4), (8, 8), (4, 16), (16, 8)):
            for causal in (False, True):
                inputs = {
                    key: [rng.uniform(-1, 1) for _ in range(rows * cols)] for key in ("q", "k", "v")
                }
                inputs["causal"] = causal
                case, report = self.run_case(
                    f"attention_{rows}_{cols}_{causal}",
                    "attention",
                    rows=rows,
                    cols=cols,
                    inputs=inputs,
                )
                self.assertEqual(report["launches"], 4)
                self.assertIn("end_to_end_accuracy", report)
                if causal:
                    # The first query can attend only to its own V row.
                    actual = [int(x, 16) for x in report["output_bits"][:cols]]
                    verify_approx(actual, [value(bits(v)) for v in inputs["v"][:cols]], "attention")
                    memory = read_hex(
                        self.output / f"attention_{rows}_{cols}_{causal}" / "softmax/output.hex",
                        1024,
                    )
                    start, _ = case.layout.regions["scores"]
                    for r in range(rows):
                        for c in range(r + 1, rows):
                            self.assertEqual(memory[start + r * rows + c], 0)

    def test_relu_special_encodings(self):
        data = [-math.inf, -1, -0.0, 0.0, 1, math.inf, math.nan, -1e-40] * 2
        self.run_case("relu_special", "relu", inputs={"x": data})

    def test_compare_shift_instructions_full_sm(self):
        a = [
            0,
            0x80000000,
            0x3F800000,
            0xBF800000,
            1,
            0x80000001,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x7F800001,
            0xFF800001,
            0xFFFFFFFF,
            0x3F800001,
            0xBF800001,
            0x7FFFFF,
            0x800000,
        ]
        b = list(reversed(a))
        for op in ("SHR", "SLTU", "FLT"):
            for masks in (0xFFFF, 0x8421):
                before = [0xDEADBEEF] * 1024
                before[:16], before[16:32] = a, b
                words = assemble(f"""TID R1
                    ADD R1, R1, R1
                    ADD R1, R1, R1
                    LD_SHARED R2, [R1]
                    LD_SHARED R3, [R1 + 64]
                    {op} R4, R2, R3
                    ADDI R5, R4, 7
                    ST_SHARED [R1 + 128], R5
                    EXIT""")
                after, _, _ = run_program(
                    self.executable, words, before, self.output / f"isa_{op}_{masks:x}", masks
                )
                for i in range(1024):
                    expected = before[i]
                    if 32 <= i < 48 and (masks >> (i - 32)) & 1:
                        x, y = a[i - 32], b[i - 32]
                        result = (
                            x >> (y & 31)
                            if op == "SHR"
                            else int(x < y)
                            if op == "SLTU"
                            else int(value(x) < value(y))
                        )
                        expected = (result + 7) & 0xFFFFFFFF
                    self.assertEqual(after[i], expected, (op, masks, i))

    def test_invalid_inputs_and_memory_capacity(self):
        cases = [
            ("softmax", {"inputs": {"x": [-math.inf] * 16}}),
            ("gelu", {"inputs": {"x": [math.inf] * 16}}),
            ("relu", {"inputs": {"x": [1] * 15}}),
            ("layernorm", {"inputs": {"epsilon": 0}}),
            ("attention", {"inputs": {"causal": 1}}),
            ("attention", {"rows": 16, "cols": 16}),
            ("tiled_matmul", {"size": 12}),
        ]
        cases += [("softmax", {"inputs": {"a": [1] * 16}})]
        for name, kwargs in cases:
            with self.subTest(name=name, kwargs=kwargs), self.assertRaises(ValueError):
                build_case(name, **kwargs)
        for name in NAMES:
            if name not in ("attention", "transformer"):
                case = build_case(name, rows=16, cols=16, size=16)
                for stage in case.stages:
                    self.assertLessEqual(len(assemble(stage.source)), 1024)

    def test_verifier_rejects_broken_assembly_and_memory_guards(self):
        for name in ("tiled_matmul", "gelu", "layernorm", "softmax"):
            case = build_case(name)
            stage = case.stages[0]
            words = assemble(stage.source)
            # Remove every output store, leaving sentinels or incomplete state.
            words = [OPCODES["NOP"] << 26 if w >> 26 == OPCODES["ST_SHARED"] else w for w in words]
            after, _, _ = run_program(
                self.executable,
                words,
                case.layout.memory,
                self.output / f"broken_{name}",
                warp_mask=1,
                max_cycles=200000,
            )
            with self.assertRaises(ValueError):
                verify_stage(stage, case.layout.memory, after, case.layout)
        case, _ = self.run_case("guard_valid", "softmax")
        after = read_hex(self.output / "guard_valid/output.hex", 1024)
        after[1023] ^= 1
        with self.assertRaisesRegex(ValueError, "memory guard"):
            verify_stage(case.stages[0], case.layout.memory, after, case.layout)

    def test_cli_custom_input_and_failed_rerun(self):
        output = self.output / "cli with spaces"
        output.mkdir(parents=True, exist_ok=True)
        inputs = output / "inputs.json"
        inputs.write_text(json.dumps({"x": list(range(32))}))
        command = [
            sys.executable,
            str(ROOT / "scripts/run_kernel.py"),
            "softmax",
            "--cols",
            "8",
            "--inputs",
            str(inputs),
            "--out",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((output / "result.json").read_text())["verified"])
        inputs.write_text(json.dumps({"x": [1]}))
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "result.json").exists())
