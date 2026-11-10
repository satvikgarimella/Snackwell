"""Full-SM tensor memory programs checked against exact integer-based FP32 math."""

import csv
import random
import shutil
import unittest

from snackwell.assembler import AssemblyError, assemble
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.reference.tensor import matmul
from snackwell.simulator import SimulatorSession, compile_model_runner, compile_runner, run_program


def descriptor(base, stride=16):
    return (stride << 12) | base


class TensorMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/tensor_memory_python"
        cls.executable = compile_runner(cls.output)

    def execute(self, label, source, memory, **kwargs):
        return run_program(
            self.executable,
            assemble(source),
            memory,
            self.output / label,
            warp_mask=kwargs.pop("warp_mask", 1),
            **kwargs,
        )

    def test_assembler_rejects_invalid_tiles_rows_and_masks(self):
        for source in (
            "TALLOC T2",
            "TLD R1, T0, 4",
            "TST T0, -1, R2",
            "TCOMMIT R1, T0, 0",
            "TCOMMIT R1, T0, 16",
            "TMMA T0, R1, R16",
            "TWAIT T0, 1",
            "TFREE R0",
        ):
            with self.subTest(source=source), self.assertRaises(AssemblyError):
                assemble(source)
        self.assertEqual(
            assemble("TST T1, 3, R2\nTLD R4, T1, 2\nTCOMMIT R3, T1, 9"),
            [0x54080007, 0x59000005, 0x64C00013],
        )

    def test_tiled_program_random_values_strides_and_guards(self):
        source = (ROOT / "kernels/tensor_memory.asm").read_text()
        rng = random.Random(0x544D454D)
        for case in range(6):
            memory = [0xDEADBEEF] * 1024
            # Distinct values in both K chunks catch incorrect descriptor strides.
            a = [bits(rng.uniform(-4, 4)) for _ in range(32)]
            b = [bits(rng.uniform(-4, 4)) for _ in range(32)]
            memory[:32], memory[32:64] = a, b
            memory[192:196] = [
                descriptor(0, 32),
                descriptor(128),
                descriptor(16, 32),
                descriptor(192),
            ]
            expected = [0] * 16
            for k in (0, 4):
                aa = [a[r * 8 + k + c] for r in range(4) for c in range(4)]
                expected = matmul(aa, b[k * 4 : (k + 4) * 4], expected)
            after, _, _ = self.execute(f"random_{case}", source, memory)
            wanted = memory.copy()
            wanted[128:144], wanted[144:148] = expected, [0] * 4
            self.assertEqual(after, wanted)

    def test_nonzero_accumulator_and_rounding_contract(self):
        memory = [0xDEADBEEF] * 1024
        a = [bits(-1e20), bits(3), 0, 0] * 4
        b = [bits(1)] * 16
        seed = [bits(1e20)] * 16
        memory[:16], memory[16:32], memory[32:48] = a, b, seed
        memory[192:194] = [descriptor(0), descriptor(64)]
        source = """XOR R0, R0, R0
            TID R12
            ADD R12, R12, R12
            ADD R12, R12, R12
            TALLOC T1
        """
        for row in range(4):
            source += f"LD_SHARED R4, [R12 + {128 + row * 16}]\nTST T1, {row}, R4\n"
        source += """LD_SHARED R1, [R0 + 768]
            LD_SHARED R2, [R0 + 772]
            TMMA_ACC T1, R1, R2
            TCOMMIT R3, T1, 1
            TWAIT T1, R3
        """
        for row in range(4):
            source += f"TLD R4, T1, {row}\nST_SHARED [R12 + {512 + row * 16}], R4\n"
        source += "TRELEASE T1\nTFREE T1\nTSTATUS R8\nEXIT"
        after, regs, _ = self.execute("seed_rounding", source, memory)
        expected = matmul(a, b, seed)
        self.assertEqual(expected, [bits(3)] * 16)
        self.assertEqual(after[128:144], expected)
        self.assertEqual([regs[lane * 16 + 8] for lane in range(4)], [0] * 4)

    def test_two_warps_completion_and_release_before_reuse(self):
        memory = [0] * 1024
        a = [bits(i + 1) for i in range(16)]
        b = [bits(int(r == c)) for r in range(4) for c in range(4)]
        memory[:16], memory[16:32] = a, b
        memory[192:194] = [descriptor(0), descriptor(64)]
        source = """XOR R0, R0, R0
            TID R12
            ADDI R11, R0, 2
            SHR R10, R12, R11
            BEQ R10, R0, producer
            ; Consumer can wait before the producer even allocates the slot.
            ADDI R3, R0, 1
            TWAIT T0, R3
            ADDI R12, R12, -4
            ADD R12, R12, R12
            ADD R12, R12, R12
            TLD R4, T0, 0
            TLD R5, T0, 1
            TLD R6, T0, 2
            TLD R7, T0, 3
            TRELEASE T0
            ST_SHARED [R12 + 512], R4
            ST_SHARED [R12 + 528], R5
            ST_SHARED [R12 + 544], R6
            ST_SHARED [R12 + 560], R7
            TSTATUS R8
            EXIT
        producer:
            TALLOC T0
            LD_SHARED R1, [R0 + 768]
            LD_SHARED R2, [R0 + 772]
            TMMA T0, R1, R2
            TCOMMIT R3, T0, 2
            ; Free must wait for the other warp's TLD snapshots and release.
            TFREE T0
            TALLOC T0
            TMMA T0, R1, R2
            TCOMMIT R3, T0, 1
            TWAIT T0, R3
            TLD R4, T0, 0
            TRELEASE T0
            TFREE T0
            TSTATUS R8
            EXIT
        """
        after, regs, _ = self.execute("cooperation", source, memory, warp_mask=3)
        self.assertEqual(after[128:144], a)
        self.assertEqual(regs[3], 2)  # Ticket survives free/reallocation.
        for warp in (0, 1):
            for lane in range(4):
                self.assertEqual(regs[warp * 64 + lane * 16 + 8], 0)
        with (self.output / "cooperation/trace.csv").open() as stream:
            trace = list(csv.DictReader(stream))
        self.assertTrue(any(int(r["resource_stall"]) and int(r["opcode"]) == 0x14 for r in trace))

    def test_invalid_descriptor_early_read_and_stale_ticket_report_errors(self):
        memory = [0] * 1024
        memory[192:194] = [descriptor(4092), descriptor(64)]
        source = """XOR R0, R0, R0
            TALLOC T0
            LD_SHARED R1, [R0 + 768]
            LD_SHARED R2, [R0 + 772]
            TMMA T0, R1, R2
            TLD R5, T0, 0
            TWAIT T0, R0
            TFREE T0
            TSTATUS R8
            EXIT
        """
        after, regs, _ = self.execute("bad_descriptor", source, memory)
        self.assertEqual(after, memory)
        self.assertEqual(regs[5], 0)
        self.assertEqual(regs[8], 0x08 | 0x04 | 0x10)

    def demo_inputs(self):
        memory = [0] * 1024
        memory[:32] = [bits(i + 1) for i in range(32)]
        memory[32:64] = [bits((i % 7) - 3) for i in range(32)]
        memory[192:196] = [descriptor(0, 32), descriptor(128), descriptor(16, 32), descriptor(192)]
        return memory

    def test_old_ticket_cannot_publish_new_generation(self):
        source = (ROOT / "kernels/tensor_memory.asm").read_text()
        source = source.replace(
            "TWAIT T0, R3\nTRELEASE", "TWAIT T0, R3\nADDI R10, R3, 0\nTRELEASE", 1
        )
        source = source.replace(
            "TMMA_ACC T0, R1, R2\nTCOMMIT R3, T0, 1",
            "TMMA_ACC T0, R1, R2\nTWAIT T0, R10\nTLD R9, T0, 0\nTCOMMIT R3, T0, 1",
        )
        after, regs, _ = self.execute("old_ticket", source, self.demo_inputs())
        self.assertEqual(after[144:148], [0x10 | 0x04] * 4)
        self.assertEqual(regs[3], 2)
        self.assertEqual(regs[9], 0)
        self.assertEqual(after[128:132], [bits(v) for v in (-10, -16, -8, 14)])

    def test_legacy_array_contention_independent_fp_and_shared_snapshot(self):
        source = (ROOT / "kernels/tensor_memory.asm").read_text()
        source = source.replace(
            "TMMA T0, R1, R2",
            """TMMA T0, R1, R2
            FADD R13, R4, R4
            ADDI R14, R0, 9
            ST_SHARED [R0], R0
            TMUL R8, R4, R4""",
            1,
        )
        after, regs, _ = self.execute("contention", source, self.demo_inputs())
        self.assertEqual(after[0], 0)  # Store ran after the operand snapshot.
        self.assertEqual(after[128:132], [bits(v) for v in (-10, -16, -8, 14)])
        self.assertEqual(after[144:148], [0] * 4)
        self.assertEqual(regs[14], 9)
        with (self.output / "contention/trace.csv").open() as stream:
            trace = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        self.assertTrue(
            any(
                r["instruction_accepted"] and r["opcode"] == 0x0D and r["tmem_state"] for r in trace
            )
        )
        self.assertTrue(any(r["resource_stall"] and r["opcode"] == 0x0F for r in trace))
        self.assertTrue(any(r["resource_stall"] and r["opcode"] == 0x09 for r in trace))
        self.assertEqual(sum(r["tmem_memory_response"] for r in trace), 16)

    def test_partial_warp_and_malformed_encoding_retire_without_side_effects(self):
        words = assemble("TALLOC T0\nTSTATUS R8\nEXIT")
        # A partial collective is rejected. Its register result still retires.
        _, regs, _ = run_program(
            self.executable, words, [0] * 1024, self.output / "partial", active_masks=5, warp_mask=1
        )
        self.assertEqual(regs[8], 0)
        # Reserved bit must not turn into a different tile or wrap an address.
        words[0] |= 1 << 13
        after, regs, _ = run_program(
            self.executable, words, [0] * 1024, self.output / "reserved", warp_mask=1
        )
        self.assertEqual(regs[8], 1)
        self.assertEqual(after, [0] * 1024)

    def test_public_device_bus_matches_full_sm(self):
        from snackwell.device import compile_device, run_device_program

        source = (ROOT / "kernels/tensor_memory.asm").read_text()
        expected, _, cycles = self.execute("bus_reference", source, self.demo_inputs())
        output = self.output / "public_bus"
        device = compile_device(output)
        actual, device_cycles = run_device_program(
            device, assemble(source), self.demo_inputs(), output, warp_mask=1
        )
        self.assertEqual(actual, expected)
        self.assertEqual(device_cycles, cycles)

    @unittest.skipUnless(shutil.which("verilator"), "requires Verilator")
    def test_verilator_and_persistent_session_match_icarus(self):
        source = (ROOT / "kernels/tensor_memory.asm").read_text()
        expected = self.execute("backend_reference", source, self.demo_inputs())
        simulator = compile_model_runner(self.output / "verilator", backend="verilator")
        actual = run_program(
            simulator, assemble(source), self.demo_inputs(), self.output / "verilator", warp_mask=1
        )
        self.assertEqual(actual, expected)
        with SimulatorSession(simulator) as session:
            for launch in range(2):
                actual = run_program(
                    session,
                    assemble(source),
                    self.demo_inputs(),
                    self.output / f"session_{launch}",
                    warp_mask=1,
                )
                self.assertEqual(actual, expected)
