import random
import unittest

from snackwell.assembler import assemble
from snackwell.kernels.basic import default_inputs, vector_add_memory, verify_vector_add
from snackwell.numeric import bits as float_bits
from snackwell.paths import ROOT
from snackwell.simulator import checked_command, compile_runner, run_program, tool_path, write_hex


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/kernel_python"
        cls.executable = compile_runner(cls.output)
        cls.vector_words = assemble((ROOT / "kernels/vector_add.asm").read_text())

    def execute(self, name, words, memory=None, masks=0xFFFF, entry=0):
        return run_program(
            self.executable,
            words,
            [0] * 1024 if memory is None else memory,
            self.output / name,
            masks,
            entry,
        )

    def test_vector_add_default(self):
        before = vector_add_memory(*default_inputs())
        after, registers, _ = self.execute("vector default", self.vector_words, before)
        verify_vector_add(before, after)
        for i in range(16):
            self.assertEqual(registers[i * 16 + 1], i * 4)
            self.assertEqual(registers[i * 16 + 2], 64 + i * 4)
            self.assertEqual(registers[i * 16 + 3], 128 + i * 4)
            self.assertEqual(after[32 + i], float_bits(11 * (i + 1)))

    def test_vector_add_fp32_edges_and_random(self):
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
        rng = random.Random(0xADD)
        cases = [(edges, edges), (edges, [x ^ 0x80000000 for x in edges])]
        cases += [
            ([rng.getrandbits(32) for _ in range(16)], [rng.getrandbits(32) for _ in range(16)])
            for _ in range(24)
        ]
        for case, (a, b) in enumerate(cases):
            with self.subTest(case=case):
                before = vector_add_memory(a, b)
                after, _, _ = self.execute(f"vector_fp_{case}", self.vector_words, before)
                verify_vector_add(before, after)

    def test_vector_add_masks(self):
        for masks in [m * 0x1111 for m in range(16)] + [0xFA50, 0x8421]:
            with self.subTest(masks=masks):
                before = vector_add_memory(*default_inputs())
                after, _, _ = self.execute(
                    f"vector_mask_{masks:04x}", self.vector_words, before, masks
                )
                verify_vector_add(before, after, masks)

    def test_addi_tid_and_labels_beyond_old_memory_limit(self):
        program = (
            "BEQ R15, R15, start\n"
            + "NOP\n" * 32
            + """
        start: TID R1
            ADDI R2, R1, -1
            ADDI R3, R15, 8191
            ADDI R4, R3, 1
            ADDI R5, R4, -8192
            ADDI R6, R15, -8192
            ADDI R7, R15, -1
            ADDI R8, R7, 1
            ADDI R9, R15, 3
        loop: ADDI R10, R10, 2
            ADDI R9, R9, -1
            BEQ R9, R15, done
            BEQ R15, R15, loop
        done: TID R0
            EXIT
        """
        )
        words = assemble(program)
        for masks in [0xFFFF, 0x8421, 0]:
            with self.subTest(masks=masks):
                _, regs, _ = self.execute(f"instructions_{masks:04x}", words, masks=masks)
                for i in range(16):
                    expected = (
                        [
                            i,
                            i,
                            (i - 1) & 0xFFFFFFFF,
                            8191,
                            8192,
                            0,
                            0xFFFFE000,
                            0xFFFFFFFF,
                            0,
                            0,
                            6,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ]
                        if (masks >> i) & 1
                        else [0] * 16
                    )
                    self.assertEqual(regs[i * 16 : (i + 1) * 16], expected)

    def test_addi_tid_load_fp_dependencies(self):
        words = assemble("""
            LD_SHARED R0, [R15]
            TID R1
            TID R0
            ADDI R2, R0, 1
            LD_SHARED R3, [R15]
            FADD R4, R3, R3
            ADDI R5, R4, -1
            FADD R6, R3, R3
            TID R6
            ADDI R7, R6, 1
            EXIT
        """)
        memory = [0] * 1024
        memory[0] = float_bits(1)
        for masks in [0xFFFF, 0x8421]:
            _, regs, _ = self.execute(f"instruction_hazards_{masks:04x}", words, memory, masks)
            for i in range(16):
                expected = [i, i, i + 1, float_bits(1), float_bits(2), float_bits(2) - 1, i, i + 1]
                self.assertEqual(
                    regs[i * 16 : i * 16 + 8], expected if (masks >> i) & 1 else [0] * 8
                )

    def test_branch_reconvergence_from_assembly(self):
        words = assemble("""
            TID R1
            ADDI R2, R15, 1
            AND R3, R1, R2
            SSY join
            BEQ R3, R15, even
            ADDI R4, R1, 100
            BEQ R15, R15, join
        even: ADDI R4, R1, 200
        join: ADDI R5, R4, 1
            EXIT
        """)
        _, regs, _ = self.execute("assembly_divergence", words)
        for i in range(16):
            self.assertEqual(regs[i * 16 + 4], i + (100 if i % 2 else 200))
            self.assertEqual(regs[i * 16 + 5], i + (101 if i % 2 else 201))

    def test_invalid_fetch_and_exit_padding(self):
        before = vector_add_memory(*default_inputs())
        for entry in [2, 4096, 0xFFFFFFFC, 4092]:
            after, regs, _ = self.execute(f"fetch_{entry}", self.vector_words, before, entry=entry)
            self.assertEqual(after, before)
            self.assertEqual(regs, [0] * 256)

    def test_missing_program_is_fatal(self):
        directory = self.output / "missing_program"
        directory.mkdir(parents=True, exist_ok=True)
        write_hex(directory / "input.hex", [0] * 1024)
        with self.assertRaisesRegex(RuntimeError, "Cannot open program"):
            checked_command(
                [
                    tool_path("vvp"),
                    str(self.executable),
                    f"+PROGRAM={directory / 'does-not-exist.hex'}",
                    f"+DATA={directory / 'input.hex'}",
                    f"+RESULT={directory / 'output.hex'}",
                ],
                directory,
                directory / "simulation.log",
            )
