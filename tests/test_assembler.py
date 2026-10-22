import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from snackwell.assembler import EXIT_WORD, OPCODES, AssemblyError, assemble, write_image
from snackwell.paths import ROOT


class AssemblerTests(unittest.TestCase):
    def test_known_machine_words(self):
        self.assertEqual(
            assemble("TMUL R8, R0, R4\nFMUL R3, R1, R2\nEXIT"), [0x3E010000, 0x30C48000, 0xFC000000]
        )
        self.assertEqual(
            assemble("TID R1\nADDI R2, R1, -1\nLD_SHARED R4, [R2 + 16]\nST_SHARED [R2 - 4], R4"),
            [0x1C400000, 0x18843FFF, 0x21080010, 0x24093FFC],
        )

    def test_forward_backward_labels_and_comments(self):
        words = assemble("""start: tid r1 ; first instruction
            BEQ R1, R0, done // forward
            ADDI R1, R1, -1
            BEQ R0, R0, start
            done: SSY exit
            NOP
            exit: EXIT # final
        """)
        self.assertEqual(words[1] & 0x3FFF, 2)
        self.assertEqual(words[3] & 0x3FFF, 0x3FFC)
        self.assertEqual(words[4] & 0x3FFF, 1)
        self.assertEqual(words[-1], EXIT_WORD)

    def test_immediate_endpoints(self):
        self.assertEqual(assemble("ADDI R1, R0, -8192\nADDI R1, R0, 8191")[0] & 0x3FFF, 0x2000)
        self.assertEqual(assemble("ADDI R1, R0, -8192\nADDI R1, R0, 8191")[1] & 0x3FFF, 0x1FFF)
        self.assertEqual(assemble("BEQ R1, R2, -2")[0] & 0x3FFF, 0x3FFE)
        self.assertEqual(assemble("ADDI R1, R0, 0x10"), assemble("ADDI R1, R0, 16"))

    def test_every_opcode_matches_rtl(self):
        rtl = dict(
            (name, int(value, 16))
            for name, value in re.findall(
                r"OP_(\w+)\s*=\s*6'h([0-9A-Fa-f]+)", (ROOT / "rtl/snackwell_pkg.sv").read_text()
            )
        )
        self.assertEqual(OPCODES, rtl)
        for name in (
            "ADD",
            "SUB",
            "AND",
            "OR",
            "XOR",
            "FMUL",
            "FADD",
            "FSUB",
            "TMUL",
            "SHR",
            "SLTU",
            "FLT",
        ):
            word = assemble(f"{name} R8, R0, R4")[0]
            self.assertEqual(word >> 26, rtl[name])
            self.assertEqual((word >> 22) & 15, 8)
            self.assertEqual((word >> 14) & 15, 4)

    def test_bad_programs_are_rejected(self):
        bad = [
            "",
            "hello",
            "ADD R16, R0, R1",
            "ADD R1 R2 R3",
            "TID R1, R2",
            "EXIT R1",
            "ADDI R1, R0, 8192",
            "ADDI R1, R0, -8193",
            "FMUL R1, R2, 3",
            "BEQ R1, R2, nowhere",
            "a: NOP\na: EXIT",
            "TMUL R13, R0, R4",
            "TMUL R0, R13, R4",
            "TMUL R0, R4, R13",
            "LD_SHARED R1, R2",
            "ST_SHARED [R1 + -4], R2",
            "ST_SHARED [R1 - 8193], R2",
            "NOP,",
            "ADDI R1, R2, 1.5",
        ]
        for source in bad:
            with self.subTest(source=source), self.assertRaises(AssemblyError):
                assemble(source)
        with self.assertRaisesRegex(AssemblyError, "line 2"):
            assemble("NOP\nADDI R1, R0, 8192")

    def test_capacity_and_padding(self):
        words = assemble("NOP\n" * 1023 + "EXIT")
        self.assertEqual(len(words), 1024)
        with self.assertRaises(AssemblyError):
            assemble("NOP\n" * 1025)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "program.hex"
            write_image(output, assemble("TID R3\nEXIT"))
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 1024)
            self.assertTrue(all(int(word, 16) == EXIT_WORD for word in lines[1:]))

    def test_cli_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "bad.asm", Path(directory) / "program.hex"
            source.write_text("NOP\nFADD R20, R1, R2\n")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/assemble.py"), str(source), "-o", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("line 2", result.stderr)
            self.assertFalse(output.exists())
