"""Async-copy integration: ownership, memory guards, arithmetic and overlap."""

import csv
import random
import shutil
import unittest

from snackwell.assembler import AssemblyError, assemble
from snackwell.global_memory import GLOBAL_WORDS, compile_global_runner, run_global_program
from snackwell.kernels.async_tensor import images, program
from snackwell.numeric import bits
from snackwell.paths import ROOT, rtl_sources
from snackwell.reference.tensor import matmul
from snackwell.simulator import checked_command, tool_path


class AsyncCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/async_copy_python"
        cls.executable = compile_global_runner(cls.output / "runner")

    def execute(self, name, source, shared, memory, **options):
        words = assemble(source) if isinstance(source, str) else source
        return run_global_program(
            self.executable, words, shared, memory, self.output / name, **options
        )

    def trace(self, name):
        with (self.output / name / "trace.csv").open() as stream:
            return list(csv.DictReader(stream))

    def test_assembler_collective_copy_instructions(self):
        self.assertEqual(
            assemble("CPASYNC R8, S1, R6, 5\nCPWAIT S1, R8\nCPRELEASE S1\nCPSTATUS R4"),
            [
                (0x20 << 26) | (8 << 22) | (6 << 18) | 11,
                (0x21 << 26) | (8 << 18) | 1,
                (0x22 << 26) | 1,
                (0x23 << 26) | (4 << 22),
            ],
        )
        for text in (
            "CPASYNC R8, S2, R6, 1",
            "CPASYNC R8, S0, R6, 0",
            "CPASYNC R8, S0, R6, 16",
            "CPWAIT T0, R8",
            "CPRELEASE S0, R1",
        ):
            with self.subTest(text=text), self.assertRaises(AssemblyError):
                assemble(text)

    def test_rectangular_strides_masks_and_captured_descriptor(self):
        rng = random.Random(0xC0F1)
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            XOR R6, R6, R6
            CPWAIT S0, R8
            CPRELEASE S0
            CPSTATUS R7
            ST_SHARED [R12 + 512], R7
            EXIT
        """
        for case, (rows, cols) in enumerate(((1, 1), (3, 5), (16, 16), (7, 2))):
            shared, memory = [0xDEADBEEF] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
            stride, dest_stride = (cols + 2) * 4, (cols + 1) * 4
            # Largest rectangle goes at byte 1024, away from descriptors/status.
            dest = 1024 if rows == 16 else 128
            shared[192:196] = [0x4000, stride, (cols << 8) | rows, (dest_stride << 12) | dest]
            wanted = shared.copy()
            for row in range(rows):
                values = [rng.getrandbits(32) for _ in range(cols)]
                offset = 4096 + row * stride // 4
                memory[offset : offset + cols] = values
                offset = (dest + row * dest_stride) // 4
                wanted[offset : offset + cols] = values
            wanted[128:132] = [0] * 4
            after, external, metrics = self.execute(
                f"rectangle_{case}", source, shared, memory, latency=case * 3
            )
            self.assertEqual(after, wanted)
            self.assertEqual(external, memory)
            self.assertEqual(
                (metrics["copy_reads"], metrics["copy_writes"], metrics["copy_status"]),
                (rows * cols, rows * cols, 0),
            )

    def test_double_buffered_random_matmul_matches_serial_at_multiple_latencies(self):
        rng = random.Random(0xD0B1E)
        for stages, latency, stride in ((3, 0, 16), (4, 7, 24), (5, 19, 32)):
            a = [bits(rng.uniform(-4, 4)) for _ in range(16 * stages)]
            b = [bits(rng.uniform(-4, 4)) for _ in range(16 * stages)]
            shared, memory, wanted, expected = images(
                stages, a=a, b=b, source_stride=stride, destination_stride=stride
            )
            results = []
            for overlap in (False, True):
                name = f"matrix_{stages}_{overlap}"
                after, external, metrics = self.execute(
                    name, program(stages, overlap=overlap), shared, memory, latency=latency
                )
                self.assertEqual(after, wanted)
                self.assertEqual(external, expected)
                self.assertEqual(
                    (metrics["reads"], metrics["writes"], metrics["copy_status"]),
                    (32 * stages, 16, 0),
                )
                trace = self.trace(name)
                self.assertEqual(sum(int(row["matrix_job"]) for row in trace), stages)
                transfers_during_arithmetic = sum(
                    int(row["copy_read"]) + int(row["copy_write"])
                    for row in trace
                    if int(row["arithmetic_active"])
                )
                self.assertEqual(transfers_during_arithmetic > 0, overlap)
                self.assertEqual(metrics["copy_arithmetic_overlap_cycles"] > 0, overlap)
                results.append(metrics)
            self.assertLess(results[1]["cycles"], results[0]["cycles"])

    def test_shared_reads_stores_and_tensor_gathers_wait_for_region_release(self):
        prefix = """TID R12
            ADDI R11, R0, 2
            SHR R10, R12, R11
            ADD R12, R12, R12
            ADD R12, R12, R12
            BEQ R10, R0, producer
        poll:
            LD_SHARED R9, [R0 + 704]
            BEQ R9, R0, poll
        """
        producer = """producer:
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            ADDI R9, R0, 1
            ST_SHARED [R0 + 704], R9
            CPWAIT S0, R8
            ADDI R9, R0, 20
        delay:
            ADDI R9, R9, -1
            BEQ R9, R0, release
            BEQ R0, R0, delay
        release:
            CPRELEASE S0
            EXIT
        """
        cases = {
            "load": ("LD_SHARED R4, [R0]\nST_SHARED [R0 + 512], R4\nEXIT\n", 0x08),
            "store": ("ADDI R4, R0, 99\nST_SHARED [R0], R4\nEXIT\n", 0x09),
            "tensor": (
                """TALLOC T0
                LD_SHARED R1, [R0 + 720]
                LD_SHARED R2, [R0 + 724]
                TMMA T0, R1, R2
                TCOMMIT R3, T0, 2
                TWAIT T0, R3
                TLD R4, T0, 0
                TRELEASE T0
                TFREE T0
                ST_SHARED [R12 + 512], R4
                EXIT
            """,
                0x17,
            ),
        }
        for name, (consumer, op) in cases.items():
            shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
            shared[192:196] = [0x4000, 16, 0x0404, 16 << 12]
            shared[180:182] = [16 << 12, (16 << 12) | 64]
            shared[16:32] = [bits(int(r == c)) for r in range(4) for c in range(4)]
            values = [bits(i + 1) for i in range(16)]
            memory[4096:4112] = values
            after, external, metrics = self.execute(
                name, prefix + consumer + producer, shared, memory, warp_mask=3, latency=9
            )
            self.assertEqual(external, memory)
            self.assertEqual(metrics["copy_status"], 0)
            if name == "load":
                self.assertEqual(after[128], values[0])
            elif name == "store":
                self.assertEqual(after[0], 99)
            else:
                self.assertEqual(after[132:136], values[:4])
            trace = self.trace(name)
            guard_pc = next(
                i * 4
                for i, word in enumerate(assemble(prefix + consumer + producer))
                if i >= 8 and word >> 26 == op
            )
            released = next(
                int(row["cycle"])
                for row in trace
                if row["accepted"] == "1" and int(row["opcode"]) == 0x22
            )
            consumed = [
                int(row["cycle"])
                for row in trace
                if row["accepted"] == "1"
                and row["warp"] == "1"
                and int(row["opcode"]) == op
                and int(row["pc"]) == guard_pc
            ]
            self.assertTrue(consumed)
            self.assertGreater(min(consumed), released)
            self.assertTrue(
                any(
                    row["resource_stall"] == "1"
                    and row["warp"] == "1"
                    and int(row["opcode"]) == op
                    and int(row["cycle"]) < released
                    for row in trace
                )
            )

    def test_two_consumers_release_before_generation_reuse(self):
        shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
        shared[192:196] = [0x4000, 16, 0x0401, 16 << 12]
        shared[196:200] = [0x4010, 16, 0x0401, 16 << 12]
        memory[4096:4104] = list(range(11, 19))
        source = """TID R12
            ADDI R11, R0, 2
            SHR R10, R12, R11
            ADD R12, R12, R12
            ADD R12, R12, R12
            BEQ R10, R0, producer
            ADDI R8, R0, 1
            CPWAIT S0, R8
            LD_SHARED R4, [R0]
            ST_SHARED [R12 + 512], R4
            ADD R9, R10, R10
        delay:
            ADDI R9, R9, -1
            BEQ R9, R0, release
            BEQ R0, R0, delay
        release:
            CPRELEASE S0
            ADDI R8, R0, 2
            CPWAIT S0, R8
            LD_SHARED R4, [R0]
            ST_SHARED [R12 + 576], R4
            CPRELEASE S0
            EXIT
        producer:
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 6
            LD_SHARED R6, [R12 + 784]
            CPASYNC R8, S0, R6, 6
            EXIT
        """
        after, external, metrics = self.execute("consumers", source, shared, memory, warp_mask=7)
        wanted = shared.copy()
        wanted[:4] = list(range(15, 19))
        wanted[132:140], wanted[148:156] = [11] * 8, [15] * 8
        self.assertEqual(after, wanted)
        self.assertEqual(external, memory)
        self.assertEqual(metrics["copy_status"], 0)
        trace = self.trace("consumers")
        starts = [
            int(row["cycle"])
            for row in trace
            if row["accepted"] == "1" and int(row["opcode"]) == 0x20
        ]
        releases = [
            int(row["cycle"])
            for row in trace
            if row["accepted"] == "1" and int(row["opcode"]) == 0x22
        ]
        self.assertEqual(len(starts), 2)
        self.assertLess(releases[1], starts[1])

    def test_reuse_preserves_matrix_operand_snapshot(self):
        shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
        shared[192:196] = [0x4000, 16, 0x0408, 16 << 12]
        shared[196:200] = [0x4080, 16, 0x0408, 16 << 12]
        shared[224:226] = [16 << 12, (16 << 12) | 64]
        a = [bits(i + 1) for i in range(16)]
        b = [bits((i % 5) - 2) for i in range(16)]
        memory[4096:4128] = a + b
        replacement = [bits(99)] * 32
        memory[4128:4160] = replacement
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            TALLOC T0
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            CPWAIT S0, R8
            LD_SHARED R1, [R0 + 896]
            LD_SHARED R2, [R0 + 900]
            TMMA T0, R1, R2
            TCOMMIT R3, T0, 1
            CPRELEASE S0
            LD_SHARED R6, [R12 + 784]
            CPASYNC R8, S0, R6, 1
            CPWAIT S0, R8
            CPRELEASE S0
            TWAIT T0, R3
            TLD R4, T0, 0
            TLD R5, T0, 1
            TLD R6, T0, 2
            TLD R7, T0, 3
            TRELEASE T0
            TFREE T0
            ST_SHARED [R12 + 512], R4
            ST_SHARED [R12 + 528], R5
            ST_SHARED [R12 + 544], R6
            ST_SHARED [R12 + 560], R7
            EXIT
        """
        after, external, metrics = self.execute("snapshot", source, shared, memory, latency=0)
        expected = shared.copy()
        expected[:32], expected[128:144] = replacement, matmul(a, b)
        self.assertEqual(after, expected)
        self.assertEqual(external, memory)
        self.assertGreater(metrics["copy_arithmetic_overlap_cycles"], 0)
        trace = self.trace("snapshot")
        matrix_start = next(int(row["cycle"]) for row in trace if row["matrix_job"] == "1")
        copies = [
            int(row["cycle"])
            for row in trace
            if row["accepted"] == "1" and int(row["opcode"]) == 0x20
        ]
        self.assertGreater(copies[1], matrix_start)

    def test_invalid_tensor_command_retires_instead_of_waiting_on_copy_guard(self):
        shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
        shared[192:196] = [0x4000, 16, 0x0404, 16 << 12]
        shared[224:228] = [16 << 12, 16 << 12, 16 << 12, (16 << 12) | 4]
        shared[228] = (16 << 12) | 64
        memory[4096:4112] = [bits(1)] * 16
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            TALLOC T0
            LD_SHARED R1, [R12 + 896]
            LD_SHARED R2, [R0 + 912]
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            TMMA T0, R1, R2
            TSTATUS R4
            ST_SHARED [R12 + 512], R4
            TFREE T0
            CPWAIT S0, R8
            CPRELEASE S0
            EXIT
        """
        after, _, metrics = self.execute("invalid_tensor", source, shared, memory)
        self.assertEqual(after[128:132], [1] * 4)
        self.assertEqual(metrics["copy_status"], 0)
        self.assertFalse(any(row["matrix_job"] == "1" for row in self.trace("invalid_tensor")))

    def test_failed_copy_releases_without_exposing_partial_result(self):
        shared, memory = [0xDEADBEEF] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
        shared[192:196] = [65520, 32, 0x0801, 32 << 12]
        memory[-4:] = [1, 2, 3, 4]
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            CPWAIT S0, R8
            CPSTATUS R4
            ST_SHARED [R12 + 512], R4
            CPRELEASE S0
            EXIT
        """
        after, external, metrics = self.execute(
            "failed_copy", source, shared, memory, expect_fault=True
        )
        wanted = shared.copy()
        wanted[:4], wanted[128:132] = [1, 2, 3, 4], [16] * 4
        self.assertEqual(after, wanted)
        self.assertEqual(external, memory)
        self.assertEqual(
            (metrics["reads"], metrics["copy_writes"], metrics["status"], metrics["copy_status"]),
            (5, 4, 0, 16),
        )

    def test_bad_descriptors_and_encodings_have_no_memory_side_effects(self):
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            EXIT
        """
        cases = (
            [0xFFFFFFFF, 16, 0x0401, 16 << 12],
            [0x4000, 12, 0x0404, 16 << 12],
            [0x4000, 16, 0x0400, 16 << 12],
            [0x4000, 16, 0x0402, (16 << 12) | 4080],
            [0x4000, 16, 0x10404, 16 << 12],
        )
        for n, descriptor in enumerate(cases):
            shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
            shared[192:196] = descriptor
            after, external, metrics = self.execute(
                f"bad_descriptor_{n}", source, shared, memory, expect_fault=True
            )
            self.assertEqual(after, shared)
            self.assertEqual(external, memory)
            self.assertEqual((metrics["reads"], metrics["copy_status"]), (0, 4))
        shared[192:196] = [0x4000, 16, 0x0404, 16 << 12]
        for n, bit in enumerate((1 << 14, 1 << 13)):
            words = assemble(source)
            words[4] |= bit
            _, external, metrics = self.execute(
                f"bad_encoding_{n}", words, shared, memory, expect_fault=True
            )
            self.assertEqual(external, memory)
            self.assertEqual((metrics["reads"], metrics["copy_status"]), (0, 1))
        _, _, metrics = self.execute(
            "partial_mask", source, shared, memory, active_masks=7, expect_fault=True
        )
        self.assertEqual((metrics["reads"], metrics["copy_status"]), (0, 1))

    def test_normal_global_traffic_shares_port_without_losing_responses(self):
        shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
        shared[192:196] = [0x4000, 16, 0x0408, 16 << 12]
        memory[4096:4128] = list(range(100, 132))
        memory[1024:1028] = [11, 22, 33, 44]
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R6, [R12 + 768]
            CPASYNC R8, S0, R6, 1
            LD_GLOBAL R4, [R12 + 4096]
            ADDI R4, R4, 7
            ST_GLOBAL [R12 + 4160], R4
            CPWAIT S0, R8
            CPRELEASE S0
            EXIT
        """
        wanted = shared.copy()
        wanted[:32] = list(range(100, 132))
        for broadcast in (False, True):
            name = f"port_contention_{broadcast}"
            program = source.replace("[R12 + 4096]", "[R0 + 4096]") if broadcast else source
            after, external, metrics = self.execute(name, program, shared, memory, latency=17)
            expected = memory.copy()
            expected[1040:1044] = [18] * 4 if broadcast else [18, 29, 40, 51]
            self.assertEqual(after, wanted)
            self.assertEqual(external, expected)
            self.assertEqual(
                (metrics["reads"], metrics["writes"], metrics["copy_status"]),
                (33 if broadcast else 36, 4, 0),
            )
            self.assertTrue(
                any(
                    row["global_busy"] == "1" and row["copy_read"] == "1"
                    for row in self.trace(name)
                )
            )

    @unittest.skipUnless(shutil.which("verilator"), "requires Verilator")
    def test_verilator_copy_pipeline_and_control_parity(self):
        output = self.output / "verilator"
        output.mkdir(parents=True, exist_ok=True)
        executable = (output / "copy_sim").resolve()
        # Relink a fresh executable while retaining the generated C++ cache.
        executable.unlink(missing_ok=True)
        checked_command(
            [
                tool_path("verilator"),
                "--binary",
                "--timing",
                "--assert",
                "-DSYNTHESIS",
                "-Wno-fatal",
                "--top-module",
                "tb_snackwell_global_memory",
                "--Mdir",
                str(output.resolve()),
                "-o",
                str(executable),
                "-j",
                "4",
                *map(str, rtl_sources()),
                str(ROOT / "tb/tb_snackwell_global_memory.sv"),
            ],
            ROOT,
            output / "compile.log",
            timeout=300,
        )
        checked_command([str(executable)], ROOT, output / "control.log")
        shared, memory, _, _ = images()
        expected = self.execute("icarus_parity", program(), shared, memory)
        actual = run_global_program(
            executable, assemble(program()), shared, memory, output / "kernel", backend="verilator"
        )
        self.assertEqual(actual, expected)
