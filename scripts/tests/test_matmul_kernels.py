import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from assemble import assemble, OPCODES
from fp32_reference import pack
from run_kernel import (compile_runner, run_program, kernel_memory, default_inputs,
                        default_matrix_inputs, float_bits, verify_matmul, verify_vector_mul)


class MatmulKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/matmul_python"
        cls.executable = compile_runner(cls.output)
        cls.programs = {name: assemble((ROOT / "kernels" / f"{name}.asm").read_text())
                        for name in ("matmul_4x4", "matmul_4x4_simd", "vector_mul")}

    def execute(self, label, kernel, a, b, masks=0xFFFF, warp_mask=None, words=None):
        if warp_mask is None:
            warp_mask = 1 if kernel == "matmul_4x4" else 15
        before = kernel_memory(a, b)
        directory = self.output / label / kernel
        after, registers, cycles = run_program(
            self.executable, self.programs[kernel] if words is None else words,
            before, directory, active_masks=masks, warp_mask=warp_mask)
        profile = json.loads((directory / "profile.json").read_text())
        return before, after, registers, cycles, profile

    def check_pair(self, label, a, b):
        for kernel in ("matmul_4x4", "matmul_4x4_simd"):
            with self.subTest(case=label, kernel=kernel):
                before, after, _, _, _ = self.execute(label, kernel, a, b)
                verify_matmul(before, after)

    def test_known_product_and_profile_work_counts(self):
        expected = [80, 70, 60, 50, 240, 214, 188, 162,
                    400, 358, 316, 274, 560, 502, 444, 386]
        for kernel in ("matmul_4x4", "matmul_4x4_simd"):
            before, after, _, cycles, profile = self.execute("known", kernel, *default_matrix_inputs())
            verify_matmul(before, after)
            self.assertEqual(after[32:48], [float_bits(x) for x in expected])
            self.assertEqual(sum(profile["issue_cycles"].values()), cycles)
            self.assertEqual(sum(profile["tensor_phases"].values()), cycles)
            self.assertEqual(profile["store_words"], 16)
            self.assertEqual(profile["opcode_counts"]["ST_SHARED"], 4)
            if kernel == "matmul_4x4":
                self.assertEqual(profile["opcode_counts"]["TMUL"], 1)
                self.assertEqual(profile["load_words"], 32)
                self.assertEqual(profile["conflicting_memory_requests"], 0)
                self.assertEqual(profile["tensor_mac_pairs"], 64)
                self.assertEqual(profile["tensor_row_commits"], 4)
                self.assertEqual(profile["simd_lane_multiplies"], 0)
                self.assertEqual([step["step"] for step in profile["tensor_steps"]], list(range(10)))
                self.assertEqual([step["useful_mac_pairs"] for step in profile["tensor_steps"]],
                                 [1, 3, 6, 10, 12, 12, 10, 6, 3, 1])
                self.assertEqual(profile["tensor_useful_step_slot_fraction"], 0.4)
                self.assertGreater(profile["tensor_pe_arithmetic_fraction"], 0)
                self.assertLess(profile["tensor_pe_arithmetic_fraction"], 1)
            else:
                self.assertEqual(profile["load_words"], 128)
                self.assertEqual(profile["conflicting_memory_requests"], 16)
                self.assertEqual(profile["simd_lane_multiplies"], 64)
                self.assertEqual(profile["simd_lane_adds"], 64)
                self.assertEqual(profile["tensor_mac_pairs"], 0)
                self.assertEqual(profile["tensor_steps"], [])

    def test_zero_identity_and_every_basis_position(self):
        a, b = default_matrix_inputs()
        zero = [0]*16
        identity = [float_bits(int(r == c)) for r in range(4) for c in range(4)]
        for i, (left, right) in enumerate([(zero, b), (a, zero), (identity, b), (a, identity)]):
            self.check_pair(f"identity_zero_{i}", left, right)
        for index in range(16):
            basis = zero.copy()
            basis[index] = float_bits(1)
            self.check_pair(f"basis_left_{index}", basis, b)
            self.check_pair(f"basis_right_{index}", a, basis)

    def test_fp32_rounding_order_and_special_values(self):
        one = float_bits(1)
        cases = [
            # round32((1+2^-23)*(1-2^-23)) == 1; a fused second MAC would differ.
            ([0xBF800000, 0x3F800001, 0, 0]*4, [one]*4 + [0x3F7FFFFE]*4 + [0]*8),
            # (((2^24 + 1) - 2^24) + 2) == 2 in sequential binary32.
            ([0x4B800000, one, 0xCB800000, 0x40000000]*4, [one]*16),
            ([0x80000000]*16, [one]*16),
            ([1, 0x80000001, 0x007FFFFF, 0x00800000]*4,
             [0x3F000000, one, 0xBF800000, 0x40000000]*4),
            ([0x7F7FFFFF]*16, [float_bits(2)]*16),
            ([0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001]*4, [one]*16),
            ([0]*16, [0x7F800000]*16),
        ]
        for case, (a, b) in enumerate(cases):
            for kernel in ("matmul_4x4", "matmul_4x4_simd"):
                with self.subTest(case=case, kernel=kernel):
                    before, after, _, _, _ = self.execute(f"fp_edge_{case}", kernel, a, b)
                    verify_matmul(before, after)
                    if case in (0, 2):
                        self.assertEqual(after[32:48], [0]*16)
                    elif case == 1:
                        self.assertEqual(after[32:48], [float_bits(2)]*16)

    def test_seeded_random_matrices(self):
        rng = random.Random(0x4D4154)
        for case in range(24):
            if case < 12:
                a, b = ([pack(rng.randrange(-1000, 1001), -5) for _ in range(16)] for _ in range(2))
            elif case < 20:
                a, b = ([pack(rng.randrange(-1024, 1025), rng.randrange(-145, 110))
                         for _ in range(16)] for _ in range(2))
            else:
                a, b = ([rng.getrandbits(32) for _ in range(16)] for _ in range(2))
            self.check_pair(f"random_{case}", a, b)

    def test_tensor_single_warp_launch_and_tags(self):
        for warp in range(4):
            before, after, registers, _, profile = self.execute(
                f"tensor_warp_{warp}", "matmul_4x4", *default_matrix_inputs(), warp_mask=1 << warp)
            verify_matmul(before, after)
            self.assertEqual(profile["opcode_counts"]["TMUL"], 1)
            self.assertEqual(profile["opcode_counts"]["EXIT"], 1)
            for other in range(4):
                if other != warp:
                    self.assertEqual(registers[other*64:(other+1)*64], [0]*64)
            for lane in range(4):
                for row in range(4):
                    self.assertEqual(registers[(warp*4+lane)*16+8+row], after[32+row*4+lane])

    def test_simd_masks_and_selected_warps(self):
        cases = [(mask*0x1111, 15) for mask in range(16)] + [(0xFA50, 15), (0x8421, 15),
                                                            (0xFFFF, 1), (0xFFFF, 10)]
        for masks, warps in cases:
            before, after, _, _, _ = self.execute(f"simd_masks_{masks:04x}_{warps:x}", "matmul_4x4_simd",
                                                  *default_matrix_inputs(), masks=masks, warp_mask=warps)
            effective = masks & sum(15 << (4*warp) for warp in range(4) if (warps >> warp) & 1)
            verify_matmul(before, after, effective)

    def test_vector_multiply_inputs_and_masks(self):
        rng = random.Random(0x4D554C)
        edges = [0, 0x80000000, 1, 0x80000001, 0x7FFFFF, 0x800000, 0x3F800000, 0xBF800000,
                 0x3F800001, 0x33800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001]
        cases = [default_inputs(), (edges, edges), (edges, list(reversed(edges)))]
        cases += [([rng.getrandbits(32) for _ in range(16)], [rng.getrandbits(32) for _ in range(16)])
                  for _ in range(12)]
        for case, (a, b) in enumerate(cases):
            before, after, _, _, _ = self.execute(f"vector_mul_{case}", "vector_mul", a, b)
            verify_vector_mul(before, after)
        for masks in [m*0x1111 for m in range(16)] + [0x8421]:
            before, after, _, _, _ = self.execute(f"vector_mul_mask_{masks:04x}", "vector_mul",
                                                  *default_inputs(), masks=masks)
            verify_vector_mul(before, after, masks)

    def test_broken_programs_fail_verification(self):
        for kernel in ("matmul_4x4", "matmul_4x4_simd"):
            for fault in ("missing_store", "wrong_arithmetic"):
                words = self.programs[kernel].copy()
                opcode = OPCODES["ST_SHARED"] if fault == "missing_store" else (
                    OPCODES["TMUL"] if kernel == "matmul_4x4" else OPCODES["FMUL"])
                index = next(i for i, word in enumerate(words) if word >> 26 == opcode)
                words[index] = (OPCODES["NOP"] << 26) if fault == "missing_store" or kernel == "matmul_4x4" else (
                    (words[index] & ((1 << 26)-1)) | (OPCODES["FADD"] << 26))
                before, after, _, _, _ = self.execute(fault, kernel, *default_matrix_inputs(), words=words)
                with self.assertRaisesRegex(ValueError, "memory byte"):
                    verify_matmul(before, after)
        before, after, _, _, _ = self.execute("guard_check", "matmul_4x4", *default_matrix_inputs())
        after[0] ^= 1
        with self.assertRaisesRegex(ValueError, "memory byte 0:"):
            verify_matmul(before, after)

    def test_comparison_cli_and_failed_rerun(self):
        directory = self.output / "cli with spaces"
        directory.mkdir(parents=True, exist_ok=True)
        inputs = directory / "inputs.json"
        inputs.write_text(json.dumps({"a": list(range(-8, 8)), "b": [i/2 for i in range(16)]}))
        command = [sys.executable, str(ROOT / "scripts/run_kernel.py"), "matmul_compare",
                   "--inputs", str(inputs), "--out", str(directory)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / "result.json").read_text())
        self.assertTrue(report["verified"])
        self.assertEqual(report["tensor"]["c_bits"], report["simd"]["c_bits"])
        self.assertIn("Memory service cycles", (directory / "comparison.md").read_text())
        inputs.write_text(json.dumps({"a": [1]*15, "b": [1]*16}))
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly 16", result.stderr)
        for path in ("result.json", "comparison.md", "matmul_4x4/result.json", "matmul_4x4_simd/result.json"):
            self.assertFalse((directory / path).exists(), path)
