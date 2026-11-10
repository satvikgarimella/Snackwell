"""Global-memory programs: independent references and full-memory guard checks."""

import csv
import random
import shutil
import unittest

from snackwell.assembler import AssemblyError, assemble
from snackwell.global_memory import GLOBAL_WORDS, compile_global_runner, run_global_program
from snackwell.numeric import bits
from snackwell.paths import ROOT, rtl_sources
from snackwell.reference.tensor import matmul
from snackwell.simulator import checked_command, tool_path


class GlobalMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/global_memory_python"
        cls.executable = compile_global_runner(cls.output / "runner")

    def run_kernel(self, label, source, shared, memory, **options):
        return run_global_program(
            self.executable, assemble(source), shared, memory, self.output / label, **options
        )

    def test_assembler_global_addressing(self):
        self.assertEqual(
            assemble("LD_GLOBAL R3, [R2 - 4]\nST_GLOBAL [R7 + 8191], R5\nGSTATUS R8"),
            [
                (0x1D << 26) | (3 << 22) | (2 << 18) | 0x3FFC,
                (0x1E << 26) | (7 << 18) | (5 << 14) | 8191,
                (0x1F << 26) | (8 << 22),
            ],
        )
        for source in ("LD_GLOBAL R2, [R0 + 8192]", "ST_GLOBAL R1, [R2]", "GSTATUS R16"):
            with self.subTest(source=source), self.assertRaises(AssemblyError):
                assemble(source)

    def test_masked_gather_scatter_all_warps_and_full_addresses(self):
        rng = random.Random(0x610BA1)
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R1, [R12]
            LD_SHARED R2, [R12 + 64]
            LD_GLOBAL R3, [R1 - 4]
            ADDI R4, R3, 7
            ST_GLOBAL [R2 + 4], R4
            GSTATUS R5
            ST_SHARED [R12 + 128], R5
            EXIT
        """
        for case, masks in enumerate((0xFFFF, 0x5A31, 0x0F0F, 0)):
            with self.subTest(masks=masks):
                memory = [0xDEADBEEF] * GLOBAL_WORDS
                shared = [0xC0FFEE11] * 1024
                inputs = rng.sample(range(1024, 4096), 16)
                outputs = rng.sample(range(8192, GLOBAL_WORDS - 1), 16)
                for tid in range(16):
                    memory[inputs[tid]] = rng.getrandbits(32)
                    # The lower 12 address bits must never alias shared memory.
                    shared[tid] = inputs[tid] * 4 + 4 if masks & (1 << tid) else 1
                    shared[16 + tid] = outputs[tid] * 4 - 4 if masks & (1 << tid) else 0xFFFFFFFF
                wanted_shared, wanted_global = shared.copy(), memory.copy()
                for tid in range(16):
                    if masks & (1 << tid):
                        wanted_global[outputs[tid]] = (memory[inputs[tid]] + 7) & 0xFFFFFFFF
                        wanted_shared[32 + tid] = 0
                after, external, metrics = self.run_kernel(
                    f"gather_{case}",
                    source,
                    shared,
                    memory,
                    warp_mask=15,
                    active_masks=masks,
                    latency=case * 5,
                )
                self.assertEqual(after, wanted_shared)
                self.assertEqual(external, wanted_global)
                self.assertEqual(metrics["status"], 0)
                self.assertEqual((metrics["reads"], metrics["writes"]), (masks.bit_count(),) * 2)

    def test_tensor_pipeline_random_values_and_latency(self):
        source = (ROOT / "kernels/global_tensor.asm").read_text()
        rng = random.Random(0x74696C65)
        a = [bits(rng.uniform(-5, 5)) for _ in range(32)]
        b = [bits(rng.uniform(-5, 5)) for _ in range(32)]
        memory = [0xDEADBEEF] * GLOBAL_WORDS
        memory[4096:4128], memory[5120:5152] = a, b
        shared = [0xDEADBEEF] * 1024
        shared[192:196] = [32 << 12, (16 << 12) | 128, (32 << 12) | 16, (16 << 12) | 192]
        expected = [0] * 16
        for k in (0, 4):
            tile = [a[r * 8 + k + c] for r in range(4) for c in range(4)]
            expected = matmul(tile, b[k * 4 : (k + 4) * 4], expected)
        wanted_shared, wanted_global = shared.copy(), memory.copy()
        wanted_shared[:64], wanted_shared[144:152] = a + b, [0] * 8
        wanted_global[6144:6160] = expected
        times = []
        for latency in (0, 3, 19):
            after, external, metrics = self.run_kernel(
                f"tensor_{latency}",
                source,
                shared,
                memory,
                latency=latency,
            )
            self.assertEqual(after, wanted_shared)
            self.assertEqual(external, wanted_global)
            self.assertEqual((metrics["reads"], metrics["writes"], metrics["status"]), (64, 16, 0))
            times.append(metrics["cycles"])
            with (self.output / f"tensor_{latency}/trace.csv").open() as stream:
                trace = list(csv.DictReader(stream))
            self.assertEqual(sum(int(row["matrix_job"]) for row in trace), 2)
        self.assertEqual(times, sorted(set(times)))

    def test_invalid_reads_and_stores_report_errors_without_wrapping(self):
        shared = [0xBADC0FFE] * 1024
        # First load: good, misaligned, overflow, unmapped external address.
        shared[:4] = [4092, 1, 0xFFFFFFFC, 65532]
        # Store: good, misaligned, underflow, unmapped external address.
        shared[4:8] = [4100, 6, 0, 65540]
        memory = [0xDEADBEEF] * GLOBAL_WORDS
        memory[1024] = 0x12345678
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R1, [R12]
            LD_GLOBAL R3, [R1 + 4]
            ST_SHARED [R12 + 64], R3
            LD_SHARED R2, [R12 + 16]
            ST_GLOBAL [R2 - 4], R3
            GSTATUS R4
            ST_SHARED [R12 + 80], R4
            EXIT
        """
        after, external, metrics = self.run_kernel(
            "errors", source, shared, memory, expect_fault=True
        )
        wanted = shared.copy()
        wanted[16:20], wanted[20:24] = [0x12345678, 0, 0, 0], [7] * 4
        self.assertEqual(after, wanted)
        self.assertEqual(external, memory)
        self.assertEqual(metrics["status"], 7)
        self.assertEqual((metrics["reads"], metrics["writes"]), (2, 2))

    def test_duplicate_store_order_and_load_after_store(self):
        shared = [0] * 1024
        memory = [0xDEADBEEF] * GLOBAL_WORDS
        source = """TID R1
            LD_GLOBAL R5, [R0 + 4096]
            ST_GLOBAL [R0 + 4096], R1
            ADDI R1, R0, 99
            LD_GLOBAL R2, [R0 + 4096]
            TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            ST_SHARED [R12], R2
            ST_SHARED [R12 + 16], R1
            ST_SHARED [R12 + 32], R5
            EXIT
        """
        after, external, metrics = self.run_kernel(
            "store_order", source, shared, memory, latency=11
        )
        wanted = memory.copy()
        wanted[1024] = 3  # Stores snapshot R1; last active lane wins.
        self.assertEqual(external, wanted)
        self.assertEqual(after, [3] * 4 + [99] * 4 + [0xDEADBEEF] * 4 + [0] * 1012)
        self.assertEqual((metrics["reads"], metrics["writes"]), (2, 4))

    def test_load_broadcast_groups_masks_warps_and_latency(self):
        rng = random.Random(0xB04DCA57)
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R1, [R12]
            LD_GLOBAL R2, [R1 - 4]
            ST_SHARED [R12 + 64], R2
            EXIT
        """
        for case in range(16):
            masks = rng.randrange(65536) if case > 3 else (0xFFFF, 0x5555, 0xAAAA, 0)[case]
            # Include all-equal, interleaved and all-distinct address groups.
            addresses = [4096 + rng.randrange(1 + case % 4) * 4 for _ in range(16)]
            shared, memory = [0xC0FFEE11] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
            for i in range(4):
                memory[1024 + i] = rng.getrandbits(32)
            expected = shared.copy()
            for tid, address in enumerate(addresses):
                shared[tid] = expected[tid] = address + 4 if masks & (1 << tid) else 1
                if masks & (1 << tid):
                    expected[16 + tid] = memory[address // 4]
            reads = sum(
                len({addresses[i] for i in range(w * 4, w * 4 + 4) if masks & (1 << i)})
                for w in range(4)
            )
            with self.subTest(case=case, masks=hex(masks)):
                after, external, metrics = self.run_kernel(
                    f"broadcast_{case}",
                    source,
                    shared,
                    memory,
                    warp_mask=15,
                    active_masks=masks,
                    latency=(0, 3, 19)[case % 3],
                )
                self.assertEqual(after, expected)
                self.assertEqual(external, memory)
                self.assertEqual(metrics["status"], 0)
                self.assertEqual(
                    (metrics["reads"], metrics["writes"], metrics["responses"]), (reads, 0, reads)
                )

    def test_independent_warp_progress_and_dependent_stall(self):
        memory = [0xDEADBEEF] * GLOBAL_WORDS
        memory[1024] = 42
        source = """TID R12
            ADDI R11, R0, 2
            SHR R10, R12, R11
            BEQ R10, R0, reader
            ADDI R6, R0, 30
        loop:
            ADDI R7, R7, 1
            BEQ R7, R6, finished
            BEQ R0, R0, loop
        finished:
            ST_SHARED [R0 + 64], R7
            EXIT
        reader:
            LD_GLOBAL R3, [R0 + 4096]
            ADDI R4, R3, 1
            ST_SHARED [R0], R4
            EXIT
        """
        after, external, metrics = self.run_kernel(
            "overlap",
            source,
            [0] * 1024,
            memory,
            warp_mask=3,
            latency=80,
        )
        self.assertEqual(after[0], 43)
        self.assertEqual(after[16], 30)
        self.assertEqual(external, memory)
        self.assertGreater(metrics["independent_integer_issues"], 5)
        self.assertGreater(metrics["dependency_stalls"], 0)
        with (self.output / "overlap/trace.csv").open() as stream:
            trace = list(csv.DictReader(stream))
        first_response = next(i for i, row in enumerate(trace) if int(row["response"]))
        self.assertTrue(
            any(row["warp"] == "1" and row["accepted"] == "1" for row in trace[:first_response])
        )
        # EXIT cannot retire warp 0 while any of its global lanes are pending.
        self.assertFalse(
            any(
                row["warp"] == "0"
                and row["opcode"] == "63"
                and row["accepted"] == "1"
                and row["global_busy"] == "1"
                for row in trace
            )
        )

    def test_fp_shared_and_global_writeback_contention(self):
        shared = [0] * 1024
        shared[:4] = [bits(1.5)] * 4
        memory = [0xDEADBEEF] * GLOBAL_WORDS
        memory[1024:1028] = [bits(2.25)] * 4
        source = """TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R1, [R12]
            LD_GLOBAL R2, [R12 + 4096]
            FADD R3, R1, R1
            LD_SHARED R5, [R12]
            FADD R4, R2, R3
            FADD R6, R4, R5
            ST_GLOBAL [R12 + 4160], R6
            EXIT
        """
        wanted = memory.copy()
        wanted[1040:1044] = [bits(6.75)] * 4
        for latency in (0, 2, 9):
            after, external, _ = self.run_kernel(
                f"writeback_{latency}", source, shared, memory, latency=latency
            )
            self.assertEqual(after, shared)
            self.assertEqual(external, wanted)

    def test_failed_kernel_removes_previous_result(self):
        output = self.output / "failed"
        shared, memory = [0] * 1024, [0] * GLOBAL_WORDS
        self.run_kernel("failed", "EXIT", shared, memory)
        self.assertTrue((output / "result.json").exists())
        with self.assertRaisesRegex(RuntimeError, "Incorrect device completion"):
            self.run_kernel("failed", "loop: BEQ R0, R0, loop", shared, memory, max_cycles=10)
        self.assertFalse((output / "result.json").exists())

    @unittest.skipUnless(shutil.which("verilator"), "requires Verilator")
    def test_verilator_public_bus_and_abort_parity(self):
        output = self.output / "verilator"
        output.mkdir(parents=True, exist_ok=True)
        executable = (output / "global_sim").resolve()
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
        source = "LD_GLOBAL R1, [R0 + 4096]\nST_GLOBAL [R0 + 4100], R1\nEXIT"
        shared, memory = [0] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
        memory[1024] = 0x12345678
        expected = self.run_kernel("icarus_parity", source, shared, memory, latency=7)
        actual = run_global_program(
            executable,
            assemble(source),
            shared,
            memory,
            output / "kernel",
            latency=7,
            backend="verilator",
        )
        self.assertEqual(actual, expected)
