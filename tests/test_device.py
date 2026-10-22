"""Numerical and protocol checks for the physical-memory loading path."""

import os
import shutil
import unittest

from snackwell.assembler import assemble
from snackwell.device import compile_device, lint_device, run_device_program
from snackwell.kernels.basic import (
    default_inputs,
    default_matrix_inputs,
    kernel_memory,
    verify_matmul,
    verify_vector_add,
    verify_vector_mul,
)
from snackwell.paths import ROOT
from snackwell.simulator import checked_command, compile_runner, run_program, tool_path


class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/device_python"
        cls.device = compile_device(cls.output / "device")
        cls.original = compile_runner(cls.output / "original")

    def test_control_and_relaunch_with_simulation_initializers_disabled(self):
        checked_command(
            [tool_path("vvp"), str(self.device)],
            ROOT,
            self.output / "control.log",
        )

    def test_bus_loaded_kernels_match_original_path_and_numerical_reference(self):
        cases = (
            ("vector_add", 15, 0xFFFF, 0, default_inputs, verify_vector_add),
            ("vector_mul", 15, 0x5A31, 0, default_inputs, verify_vector_mul),
            ("vector_add", 10, 0xFFFF, 64, default_inputs, verify_vector_add),
            ("matmul_4x4", 1, 0xFFFF, 0, default_matrix_inputs, verify_matmul),
            ("matmul_4x4", 8, 0xFFFF, 0, default_matrix_inputs, verify_matmul),
            ("matmul_4x4_simd", 15, 0xFFFF, 0, default_matrix_inputs, verify_matmul),
        )
        for name, warps, active, entry, inputs, verify in cases:
            with self.subTest(kernel=name, warps=warps, active=active, entry=entry):
                memory = kernel_memory(*inputs())
                words = assemble("EXIT") * (entry // 4)
                words += assemble((ROOT / "kernels" / f"{name}.asm").read_text())
                output = self.output / f"{name}_{warps}"
                launch = dict(warp_mask=warps, active_masks=active, entry=entry)
                actual, cycles = run_device_program(
                    self.device, words, memory, output / "device", **launch
                )
                original, _, original_cycles = run_program(
                    self.original, words, memory, output / "original", **launch
                )
                self.assertEqual(actual, original)  # Includes all memory guards.
                self.assertEqual(cycles, original_cycles)
                mask = active & sum(15 << (4 * warp) for warp in range(4) if warps & (1 << warp))
                verify(memory, actual, 0xFFFF if name == "matmul_4x4" else mask)

    def test_failed_kernel_cannot_leave_a_success_report(self):
        output = self.output / "failed"
        memory = [0] * 1024
        run_device_program(self.device, assemble("EXIT"), memory, output)
        self.assertTrue((output / "result.json").exists())
        with self.assertRaisesRegex(RuntimeError, "Incorrect completion status"):
            run_device_program(
                self.device, assemble("again: BEQ R0, R0, again"), memory, output, max_cycles=10
            )
        for name in ("result.json", "output.hex", "cycles.txt"):
            self.assertFalse((output / name).exists())

    @unittest.skipUnless(
        shutil.which(os.environ.get("VERILATOR", "verilator")), "requires Verilator"
    )
    def test_device_lint_without_simulation_constructs(self):
        lint_device(self.output)
