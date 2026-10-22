import json
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from snackwell.assembler import assemble
from snackwell.kernels.basic import (
    default_inputs,
    default_matrix_inputs,
    kernel_memory,
    verify_matmul,
    verify_vector_add,
    verify_vector_mul,
)
from snackwell.model_runner import forward
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, compile_model_runner, read_hex, run_program
from tests.test_model import fixture


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/simulator_python"
        cls.simulator = compile_model_runner(cls.output, backend="icarus")

    def test_trace_toggle_preserves_results_and_removes_stale_trace(self):
        output = self.output / "trace_toggle"
        words = assemble((ROOT / "kernels/vector_add.asm").read_text())
        memory = kernel_memory(*default_inputs())
        # A bare Icarus path keeps the existing kernel/profiler API's tracing behavior.
        traced = run_program(self.simulator.executable, words, memory, output)
        self.assertTrue((output / "trace.csv").is_file())
        profile = json.loads((output / "profile.json").read_text())
        self.assertEqual(sum(profile["issue_cycles"].values()), traced[2])
        untraced = run_program(self.simulator, words, memory, output)
        self.assertEqual(traced, untraced)
        verify_vector_add(memory, untraced[0])
        self.assertFalse((output / "trace.csv").exists())
        self.assertEqual(
            json.loads((output / "profile.json").read_text()),
            dict(cycles=untraced[2], simulator="icarus", trace_enabled=False),
        )

    def test_memory_image_parser_rejects_unknown_or_malformed_words(self):
        path = self.output / "parser.hex"
        path.write_text("// comment\n00000000 DEADBEEF\nffffffff // another comment\n")
        self.assertEqual(read_hex(path, 3), [0, 0xDEADBEEF, 0xFFFFFFFF])
        for text in (
            "0000000x",
            "zzzzzzzz",
            "+0000001",
            "0000_001",
            "0000001",
            "000000001",
            "00000000 00000000",
            "",
        ):
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_hex(path, 1)

    def test_incomplete_execution_without_trace_cannot_reuse_success(self):
        output = self.output / "timeout"
        run_program(self.simulator, assemble("EXIT"), [0] * 1024, output)
        with self.assertRaisesRegex(RuntimeError, "Kernel timeout"):
            run_program(
                self.simulator,
                assemble("loop: BEQ R15, R15, loop"),
                [0] * 1024,
                output,
                max_cycles=10,
            )
        for name in ("output.hex", "registers.hex", "trace.csv", "profile.json"):
            self.assertFalse((output / name).exists(), name)


@unittest.skipUnless(shutil.which(os.environ.get("VERILATOR", "verilator")), "requires Verilator")
class VerilatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/verilator_python"
        cls.icarus = compile_model_runner(cls.output, backend="icarus")
        cls.verilator = compile_model_runner(cls.output, backend="verilator")

    def compare_kernel(self, name, memory, **launch):
        words = assemble((ROOT / "kernels" / f"{name}.asm").read_text())
        first = run_program(self.icarus, words, memory, self.output / "icarus" / name, **launch)
        second = run_program(
            self.verilator, words, memory, self.output / "verilator" / name, **launch
        )
        self.assertEqual(first, second, "memory, registers, and cycles must match exactly")
        with SimulatorSession(self.verilator) as session:
            direct = run_program(session, words, memory, self.output / "direct" / name, **launch)
        self.assertEqual(first, direct, "the direct clock driver must preserve every bit and cycle")
        return second[0]

    def test_fp32_edges_masks_and_tensor_rounding_match_icarus(self):
        edges = [
            0,
            0x80000000,
            1,
            0x80000001,
            0x7FFFFF,
            0x800000,
            0x3F800000,
            0xBF800000,
            0x3F800001,
            0x33800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
        ]
        for kernel, check in (("vector_add", verify_vector_add), ("vector_mul", verify_vector_mul)):
            for masks in (0xFFFF, 0x8421, 0):
                with self.subTest(kernel=kernel, masks=masks):
                    memory = kernel_memory(edges, list(reversed(edges)))
                    result = self.compare_kernel(kernel, memory, active_masks=masks)
                    check(memory, result, masks)
        # Separate multiply/add rounding must remain unchanged in the compiled simulator.
        a = [0xBF800000, 0x3F800001, 0, 0] * 4
        b = [0x3F800000] * 4 + [0x3F7FFFFE] * 4 + [0] * 8
        memory = kernel_memory(a, b)
        for warp in (0, 3):
            with self.subTest(warp=warp):
                result = self.compare_kernel("matmul_4x4", memory, warp_mask=1 << warp)
                verify_matmul(memory, result)
                self.assertEqual(result[32:48], [0] * 16)

    def test_tensor_trace_and_profile_match_icarus(self):
        words = assemble((ROOT / "kernels/matmul_4x4.asm").read_text())
        memory = kernel_memory(*default_matrix_inputs())
        profiles, traces, outputs = [], [], []
        for simulator in (self.icarus, self.verilator):
            output = self.output / "trace" / simulator.backend
            outputs.append(
                run_program(replace(simulator, trace=True), words, memory, output, warp_mask=1)
            )
            profile = json.loads((output / "profile.json").read_text())
            self.assertEqual(profile.pop("simulator"), simulator.backend)
            self.assertTrue(profile["trace_enabled"])
            profiles.append(profile)
            traces.append((output / "trace.csv").read_text())
            # Turning tracing off in the same directory must remove the previous trace.
            self.assertEqual(
                run_program(simulator, words, memory, output, warp_mask=1), outputs[-1]
            )
            self.assertFalse((output / "trace.csv").exists())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(profiles[0], profiles[1])
        self.assertEqual(traces[0], traces[1])

    def test_complete_model_matches_icarus(self):
        model = fixture()
        reports = []
        output = Path(tempfile.mkdtemp(prefix="complete-model-", dir=self.output))
        for simulator in (self.icarus, self.verilator):
            report = forward(model, [1, 2, 3, 4], output / simulator.backend, simulator)
            self.assertTrue(report["verified"])
            self.assertEqual(report.pop("simulation"), dict(backend=simulator.backend, trace=False))
            self.assertGreater(report.pop("wall_seconds"), 0)
            self.assertGreater(report.pop("reference_seconds"), 0)
            reports.append(report)
        self.assertEqual(reports[0], reports[1])
        launches = sorted((output / "icarus").glob("[0-9]*_*/profile.json"))
        self.assertEqual(len(launches), reports[0]["launches"])
        for profile in launches:
            for name, count in (("output.hex", 1024), ("registers.hex", 256)):
                self.assertEqual(
                    read_hex(profile.parent / name, count),
                    read_hex(output / "verilator" / profile.parent.name / name, count),
                )

    def test_session_resets_state_and_preserves_paths_masks_and_trace(self):
        words = assemble((ROOT / "kernels/vector_add.asm").read_text())
        memory = kernel_memory(*default_inputs())
        with SimulatorSession(replace(self.verilator, trace=True)) as session:
            for index, (program, data, masks, warps) in enumerate(
                [
                    (words, memory, 0x8421, 15),
                    (assemble("EXIT"), [0] * 1024, 0xFFFF, 2),
                    (words, list(reversed(memory)), 0xFFFF, 1),
                ]
            ):
                output = self.output / "session with spaces" / str(index)
                actual = run_program(session, program, data, output, masks, warp_mask=warps)
                expected = run_program(
                    replace(self.icarus, trace=True),
                    program,
                    data,
                    output / "one-shot",
                    masks,
                    warp_mask=warps,
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    (output / "trace.csv").read_text(),
                    (output / "one-shot/trace.csv").read_text(),
                )
                if index == 1:
                    self.assertEqual(actual[1], [0] * 256)
                process = session.process
            session.trace = False
            run_program(session, assemble("EXIT"), [0] * 1024, output)
            self.assertFalse((output / "trace.csv").exists())
            self.assertIs(session.process, process)
        self.assertIsNotNone(process.poll(), "leaked simulator process")

    def test_session_failure_clears_success_and_can_restart(self):
        output = self.output / "session_failure"
        with SimulatorSession(self.verilator) as session:
            run_program(session, assemble("EXIT"), [0] * 1024, output)
            process = session.process
            with self.assertRaisesRegex(RuntimeError, "Kernel timeout"):
                run_program(
                    session,
                    assemble("loop: BEQ R15, R15, loop"),
                    [0] * 1024,
                    output,
                    max_cycles=10,
                )
            self.assertIsNone(session.process)
            self.assertIsNotNone(process.poll())
            for name in ("output.hex", "registers.hex", "trace.csv", "profile.json"):
                self.assertFalse((output / name).exists(), name)
            after, registers, _ = run_program(session, assemble("EXIT"), [1] * 1024, output)
            self.assertEqual(after, [1] * 1024)
            self.assertEqual(registers, [0] * 256)

    def test_untraced_session_resets_memory_registers_and_entry(self):
        self.assertIsNotNone(self.verilator.session_executable)
        with SimulatorSession(self.verilator) as session:
            self.assertEqual(session.executable, self.verilator.session_executable)
            for index, (source, memory, masks, warps, entry) in enumerate(
                [
                    ("ADDI R2, R0, 27\nEXIT", [0xFFFFFFFF] * 1024, 0x8421, 15, 0),
                    ("ADDI R2, R0, 27\nEXIT", [0] * 1024, 0xFFFF, 2, 4),
                    ("EXIT", [1] * 1024, 0, 1, 0),
                    ("EXIT", [0] * 1024, 0xFFFF, 8, 0xFFFFFFFF),
                ]
            ):
                directory = self.output / "direct session with spaces" / str(index)
                words = assemble(source)
                actual = run_program(session, words, memory, directory, masks, entry, warps)
                expected = run_program(
                    self.icarus, words, memory, directory / "icarus", masks, entry, warps
                )
                self.assertEqual(actual, expected)
                if index:
                    self.assertEqual(actual[1], [0] * 256)
            direct_process = session.process
            session.trace = True
            output = self.output / "direct trace fallback"
            traced = run_program(session, assemble("EXIT"), [0] * 1024, output)
            expected = run_program(self.icarus, assemble("EXIT"), [0] * 1024, output / "icarus")
            self.assertEqual(traced, expected)
            self.assertTrue((output / "trace.csv").is_file())
            self.assertIsNot(session.process, direct_process)
            self.assertIsNotNone(direct_process.poll())

    def test_direct_host_rejects_malformed_requests_and_images(self):
        directory = self.output / "bad_direct_input"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "program.hex").write_text("00000000\n" * 1024)
        good_image = "00000000\n" * 1024
        for options, image in [
            ("ffff 1 0 10 0", good_image + "00000000\n"),
            ("ffff 1 0 10 0", "0000000x\n" + good_image[9:]),
            ("ffff 1 0 10 0", "-0000001\n" + good_image[9:]),
            ("ffff 1 0 10 0", good_image[:-9]),
            ("ffff 1 0 10 1", good_image),
            ("ffff 0 0 10 0", good_image),
            ("ffff 1 0 -1 0", good_image),
            ("ffff 1 0 1000001 0", good_image),
            ("ffff 1 4294967296 10 0", good_image),
            ("ffff 1 0 10 0 extra", good_image),
        ]:
            with self.subTest(options=options, image=image[:16]):
                (directory / "input.hex").write_text(image)
                completed = subprocess.run(
                    [str(self.verilator.session_executable), "+SESSION"],
                    input=f"{directory}\n{options}\n",
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ERROR", completed.stderr)
                self.assertNotIn("PASS", completed.stdout)
                self.assertFalse((directory / "output.hex").exists())
