"""Packed NVFP4 format, arithmetic, shared ownership, and full-SM execution."""

import csv
import math
import random
import shutil
import unittest

from snackwell.assembler import AssemblyError, assemble
from snackwell.global_memory import GLOBAL_WORDS, compile_global_runner, run_global_program
from snackwell.kernels import async_tensor
from snackwell.kernels.nvfp4 import images, program
from snackwell.numeric import SENTINEL, bits, value
from snackwell.paths import ROOT, rtl_sources
from snackwell.reference.nvfp4 import (
    QuantizedVectors,
    e2m1,
    encode_e2m1,
    encode_ue4m3,
    matmul,
    quantize,
    scaled,
    ue4m3,
)
from snackwell.simulator import checked_command, tool_path


def local_case(a, b, initial=None):
    shared = [SENTINEL] * 1024
    shared[:12], shared[32:44] = a.group_words(0, 0), b.group_words(0, 0)
    shared[192:194] = [12 << 12, (12 << 12) | 128]
    lines = ["XOR R0, R0, R0", "TID R12", "ADD R12, R12, R12", "ADD R12, R12, R12", "TALLOC T1"]
    if initial is not None:
        shared[64:80] = initial
        for row in range(4):
            lines.extend((f"LD_SHARED R4, [R12 + {256 + row * 16}]", f"TST T1, {row}, R4"))
    lines.extend(
        (
            "LD_SHARED R1, [R0 + 768]",
            "LD_SHARED R2, [R0 + 772]",
            f"{'TMMA_NVFP4_ACC' if initial is not None else 'TMMA_NVFP4'} T1, R1, R2",
            "TCOMMIT R3, T1, 1",
            "TWAIT T1, R3",
        )
    )
    for row in range(4):
        lines.extend((f"TLD R4, T1, {row}", f"ST_SHARED [R12 + {512 + row * 16}], R4"))
    lines.extend(("TRELEASE T1", "TFREE T1", "TSTATUS R14", "ST_SHARED [R12 + 592], R14", "EXIT"))
    return shared, "\n".join(lines)


class NVFP4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/nvfp4_python"
        cls.executable = compile_global_runner(cls.output / "runner")

    def execute(self, name, source, shared, memory=None, **options):
        words = assemble(source) if isinstance(source, str) else source
        return run_global_program(
            self.executable,
            words,
            shared,
            [SENTINEL] * GLOBAL_WORDS if memory is None else memory,
            self.output / name,
            max_cycles=100000,
            **options,
        )

    def trace(self, name):
        with (self.output / name / "trace.csv").open() as stream:
            return list(csv.DictReader(stream))

    def test_exact_encodings_and_nearest_even_boundaries(self):
        positive = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
        self.assertEqual(tuple(e2m1(c) for c in range(8)), positive)
        self.assertEqual([bits(e2m1(c + 8)) for c in range(8)], [bits(-float(x)) for x in positive])
        self.assertEqual(
            [ue4m3(c) for c in (0, 1, 7, 8, 0x38, 0x7E)], [0, 2**-9, 7 * 2**-9, 2**-6, 1, 448]
        )
        for code in range(16):
            self.assertEqual(encode_e2m1(e2m1(code)), code)
        for code in range(127):
            self.assertEqual(encode_ue4m3(ue4m3(code)), code)
        for code in range(7):
            midpoint = (positive[code] + positive[code + 1]) / 2
            tie = code if code % 2 == 0 else code + 1
            for sign in (1, -1):
                sign_code = 0 if sign == 1 else 8
                self.assertEqual(encode_e2m1(sign * midpoint), tie | sign_code)
                self.assertEqual(encode_e2m1(sign * math.nextafter(midpoint, 0)), code | sign_code)
                self.assertEqual(
                    encode_e2m1(sign * math.nextafter(midpoint, math.inf)), (code + 1) | sign_code
                )
        for code in range(126):
            midpoint = (ue4m3(code) + ue4m3(code + 1)) / 2
            self.assertEqual(encode_ue4m3(midpoint), code if code % 2 == 0 else code + 1)
            self.assertEqual(encode_ue4m3(math.nextafter(midpoint, 0)), code)
            self.assertEqual(encode_ue4m3(math.nextafter(midpoint, math.inf)), code + 1)
        self.assertEqual(encode_e2m1(1e30), 7)
        self.assertEqual(encode_e2m1(-1e30), 15)
        self.assertEqual(encode_ue4m3(1e30), 0x7E)
        for invalid in (127, 128, 255, -1):
            with self.assertRaises(ValueError):
                ue4m3(invalid)

    def test_quantizer_two_levels_zeros_subnormals_and_bad_inputs(self):
        zero = quantize([[0.0, -0.0] * 8])
        self.assertEqual(zero.tensor_scale, bits(1))
        self.assertEqual(zero.codes, [[0, 8] * 8])
        self.assertEqual(zero.scales, [[0]])
        tiny = quantize([[value(1)] * 16])
        self.assertEqual(tiny.tensor_scale, 1)
        self.assertTrue(tiny.scales[0][0] > 0)
        sample = quantize([[1.0] * 16 + [100.0] * 16] * 4)
        self.assertLess(sample.scales[0][0], sample.scales[0][1])
        self.assertEqual(len(sample.group_words(0, 1)), 12)
        for vector, options in (
            ([math.nan] * 16, {}),
            ([math.inf] * 16, {}),
            ([0] * 15, {}),
            ([1] * 16, {"tensor_scale": 0}),
            ([1] * 16, {"tensor_scale": -1}),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                quantize([vector], **options)

    def test_assembler_and_invalid_collective_encodings(self):
        self.assertEqual(
            assemble("TMMA_NVFP4 T0, R1, R2\nTMMA_NVFP4_ACC T1, R3, R4"),
            [(0x24 << 26) | (1 << 18) | (2 << 14), (0x25 << 26) | (3 << 18) | (4 << 14) | 1],
        )
        for bad in ("TMMA_NVFP4 T2, R1, R2", "TMMA_NVFP4 R0, R1, R2", "TMMA_NVFP4_ACC T0, R1, R16"):
            with self.assertRaises(AssemblyError):
                assemble(bad)
        q = quantize([[1] * 16] * 4, tensor_scale=1)
        shared, source = local_case(q, q)
        # No accepted job means no ticket to commit or wait for.
        source = (
            source[: source.index("TCOMMIT")]
            + "TFREE T1\nTSTATUS R14\nST_SHARED [R12 + 592], R14\nEXIT"
        )
        for name, instruction in (
            ("reserved_rd", (0x24 << 26) | (1 << 22) | (1 << 18) | (2 << 14) | 1),
            ("reserved_imm", (0x24 << 26) | (1 << 18) | (2 << 14) | 3),
        ):
            words = assemble(source)
            words[next(i for i, word in enumerate(words) if word >> 26 == 0x24)] = instruction
            after, _, _ = self.execute(name, words, shared)
            wanted = shared.copy()
            wanted[148:152] = [1] * 4
            self.assertEqual(after, wanted)
            self.assertFalse(any(int(row["matrix_job"]) for row in self.trace(name)))

        divergent = source.replace(
            "TMMA_NVFP4 T1", "SSY joined\nBEQ R12, R0, joined\nTMMA_NVFP4 T1"
        )
        divergent = divergent.replace("TFREE T1", "joined: TFREE T1")
        after, _, _ = self.execute("partial_warp", divergent, shared)
        wanted = shared.copy()
        wanted[148:152] = [1] * 4
        self.assertEqual(after, wanted)
        after, _, _ = self.execute(
            "nonuniform", source.replace("[R0 + 768]", "[R12 + 768]"), shared
        )
        wanted[148:152] = [1] * 4
        self.assertEqual(after, wanted)

    def test_random_full_pipeline_strides_scale_groups_and_output_tiles(self):
        rng = random.Random(0xF0416)
        for case, (m, n, k, latency, source_stride, dest_stride) in enumerate(
            (
                (4, 4, 16, 0, 12, 12),
                (4, 8, 32, 7, 20, 16),
                (8, 4, 48, 19, 28, 32),
                (8, 8, 64, 2, 16, 20),
            )
        ):
            a = quantize(
                [[rng.uniform(-2, 2) * 2 ** (p // 16) for p in range(k)] for _ in range(m)]
            )
            b = quantize(
                [[rng.uniform(-3, 3) * 2 ** -(p // 16) for p in range(k)] for _ in range(n)]
            )
            shared, memory, wanted, external = images(
                a, b, source_stride=source_stride, destination_stride=dest_stride
            )
            for overlap in (False, True):
                name = f"random_{case}_{overlap}"
                actual, output, metrics = self.execute(
                    name, program(m, n, k, overlap=overlap), shared, memory, latency=latency
                )
                self.assertEqual(actual, wanted)
                self.assertEqual(output, external)
                stages = m * n * k // 256
                self.assertEqual(
                    (metrics["copy_reads"], metrics["copy_writes"], metrics["writes"]),
                    (stages * 24, stages * 24, m * n),
                )
                self.assertEqual((metrics["status"], metrics["copy_status"]), (0, 0))
                trace = self.trace(name)
                self.assertEqual(sum(int(row["matrix_job"]) for row in trace), stages * 4)
                self.assertEqual(sum(int(row["nvfp4_mac"]).bit_count() for row in trace), m * n * k)
                self.assertFalse(any(int(row["fp32_tensor_multiply"]) for row in trace))

    def test_matches_fp32_array_for_the_same_decoded_values(self):
        rng = random.Random(5424)
        a = quantize([[rng.uniform(-20, 20) for _ in range(32)] for _ in range(4)], tensor_scale=1)
        b = quantize([[rng.uniform(-20, 20) for _ in range(32)] for _ in range(4)], tensor_scale=1)
        shared, memory, wanted, expected = images(a, b)
        after, external, _ = self.execute("narrow_parity", program(4, 4, 32), shared, memory)
        self.assertEqual((after, external), (wanted, expected))
        aa = [
            bits(scaled(a.codes[r][k], a.scales[r][k // 16])) for r in range(4) for k in range(32)
        ]
        bb = [
            bits(scaled(b.codes[c][k], b.scales[c][k // 16])) for k in range(32) for c in range(4)
        ]
        shared, memory, wanted, expected = async_tensor.images(8, a=aa, b=bb)
        after, fp32_output, _ = self.execute("wide_parity", async_tensor.program(8), shared, memory)
        self.assertEqual((after, fp32_output), (wanted, expected))
        self.assertEqual(fp32_output[6144:6160], external[6144:6160])

    def test_initial_accumulators_rounding_special_values_and_capture(self):
        q = QuantizedVectors([[2] * 16] * 4, [[0x38]] * 4, bits(1))
        initial = [
            bits(2**24),
            bits(-(2**24)),
            bits(0.5),
            0x80000000,
            1,
            0x80000001,
            0x7F800000,
            0xFF800000,
            0x7FC00123,
            0x7F800001,
            bits(-16),
            bits(-15.5),
            bits(2**25),
            bits(-(2**25)),
            bits(1e30),
            bits(-1e30),
        ]
        shared, source = local_case(q, q, initial)
        source = source.replace("TCOMMIT", "ADDI R1, R0, 1\nADDI R2, R0, 3\nTCOMMIT")
        after, _, _ = self.execute("accumulator", source, shared)
        wanted = shared.copy()
        wanted[128:144] = matmul(q, q, initial, apply_tensor_scale=False)
        wanted[148:152] = [0] * 4
        self.assertEqual(wanted[128], bits(2**24))  # Group-first summation would add 16.
        self.assertEqual(after, wanted)

    def test_invalid_scale_completes_failed_without_changing_accumulators(self):
        q = quantize([[1] * 16] * 4, tensor_scale=1)
        initial = [bits(i + 10) for i in range(16)]
        for index, invalid in ((2, 0x7F), (11, 0x80), (34, 0x100), (43, 0x80000000)):
            shared, source = local_case(q, q, initial)
            shared[index] = invalid
            # Failed TLD returns zero. Reinitialize row 0 after release, then
            # verify untouched storage in all four rows, including rows 1..3.
            recovery = "TRELEASE T1\nLD_SHARED R4, [R12 + 256]\nTST T1, 0, R4\n"
            for row in range(4):
                recovery += f"TLD R4, T1, {row}\nST_SHARED [R12 + {640 + row * 16}], R4\n"
            source = source.replace("TRELEASE T1\n", recovery)
            name = f"bad_scale_{index}"
            after, _, _ = self.execute(name, source, shared)
            wanted = shared.copy()
            wanted[128:144], wanted[148:152], wanted[160:176] = [0] * 16, [0x44] * 4, initial
            self.assertEqual(after, wanted)
            self.assertFalse(any(int(row["matrix_job"]) for row in self.trace(name)))

    def test_descriptor_bounds_and_twelve_byte_vectors(self):
        q = quantize([[1] * 16] * 4, tensor_scale=1)
        shared, source = local_case(q, q)
        shared[1012:1024], shared[:12] = shared[:12], [SENTINEL] * 12
        shared[192] = (12 << 12) | 4048
        after, _, _ = self.execute("last_byte", source, shared)
        wanted = shared.copy()
        wanted[128:144], wanted[148:152] = matmul(q, q), [0] * 4
        self.assertEqual(after, wanted)
        invalid_source = (
            source[: source.index("TCOMMIT")]
            + "TFREE T1\nTSTATUS R14\nST_SHARED [R12 + 592], R14\nEXIT"
        )
        for case, descriptor in enumerate(
            ((12 << 12) | 4052, (8 << 12), (13 << 12), 1 << 24, (12 << 12) | 1)
        ):
            shared[192] = descriptor
            after, _, _ = self.execute(f"bad_descriptor_{case}", invalid_source, shared)
            wanted = shared.copy()
            wanted[148:152] = [8] * 4
            self.assertEqual(after, wanted)

    def test_reused_copy_slot_preserves_all_sixteen_operand_values(self):
        q = quantize([[float(i % 7 - 3) for i in range(16)]] * 4, tensor_scale=1)
        other = quantize([[6] * 16] * 4, tensor_scale=1)
        shared, memory, wanted, expected = images(q, q)
        packed = other.group_words(0, 0) * 2
        memory[5000:5024] = packed
        shared[200:204] = [20000, 12, 0x0308, 12 << 12]
        expected[5000:5024] = packed
        wanted[:24], wanted[200:204] = packed, shared[200:204]
        wanted[156:160] = [2] * 4
        source = program(4, 4, 16).replace(
            "CPRELEASE S0\nTWAIT",
            "CPRELEASE S0\nLD_SHARED R6, [R12 + 800]\nCPASYNC R8, S0, R6, 1\nCPWAIT S0, R8\nCPRELEASE S0\nTWAIT",
        )
        after, external, _ = self.execute("snapshot", source, shared, memory)
        self.assertEqual((after, external), (wanted, expected))
        self.assertTrue(
            any(
                int(row["copy_write"]) and int(row["arithmetic_active"])
                for row in self.trace("snapshot")
            )
        )

    def test_copy_guard_includes_scale_word_until_other_consumer_releases(self):
        q = quantize([[1] * 16] * 4, tensor_scale=1)
        shared, local = local_case(q, q)
        memory = [SENTINEL] * GLOBAL_WORDS
        memory[4096], shared[11] = shared[11], 0x7F
        shared[196:200] = [0x4000, 4, 0x0101, (4 << 12) | 44]
        shared[200] = 0
        source = """XOR R0, R0, R0
            TID R12
            ADDI R11, R0, 2
            SHR R10, R12, R11
            BEQ R10, R0, consumer
            ADDI R12, R12, -4
            ADD R12, R12, R12
            ADD R12, R12, R12
            LD_SHARED R6, [R12 + 784]
            CPASYNC R8, S0, R6, 2
            ADDI R1, R0, 1
            ST_SHARED [R0 + 800], R1
            CPWAIT S0, R8
            CPRELEASE S0
            CPSTATUS R14
            ST_SHARED [R12 + 608], R14
            EXIT
        consumer:
            ADD R12, R12, R12
            ADD R12, R12, R12
        wait_producer:
            LD_SHARED R10, [R0 + 800]
            BEQ R10, R0, wait_producer
        """ + local[local.index("TALLOC T1") :]
        after, external, _ = self.execute(
            "scale_guard", source, shared, memory, warp_mask=3, latency=200
        )
        wanted = shared.copy()
        wanted[11], wanted[200] = memory[4096], 1
        wanted[128:144], wanted[148:156] = matmul(q, q), [0] * 8
        self.assertEqual((after, external), (wanted, memory))
        trace = self.trace("scale_guard")
        matrix_cycle = next(
            int(r["cycle"]) for r in trace if r["accepted"] == "1" and r["opcode"] == str(0x24)
        )
        release_cycle = next(
            int(r["cycle"]) for r in trace if r["accepted"] == "1" and r["opcode"] == str(0x22)
        )
        self.assertLess(release_cycle, matrix_cycle)
        self.assertTrue(any(r["opcode"] == str(0x24) and r["resource_stall"] == "1" for r in trace))

    def test_failed_job_propagates_to_other_warp_and_releases(self):
        q = quantize([[1] * 16] * 4, tensor_scale=1)
        shared, _ = local_case(q, q)
        shared[43] = 0x7F
        source = """XOR R0, R0, R0
            TID R12
            ADDI R11, R0, 2
            SHR R10, R12, R11
            BEQ R10, R0, producer
            ADDI R12, R12, -4
            ADD R12, R12, R12
            ADD R12, R12, R12
            ADDI R3, R0, 1
            TWAIT T1, R3
            TLD R4, T1, 0
            ST_SHARED [R12 + 512], R4
            TRELEASE T1
            TSTATUS R14
            ST_SHARED [R12 + 608], R14
            EXIT
        producer:
            ADD R12, R12, R12
            ADD R12, R12, R12
            TALLOC T1
            LD_SHARED R1, [R0 + 768]
            LD_SHARED R2, [R0 + 772]
            TMMA_NVFP4 T1, R1, R2
            TCOMMIT R3, T1, 2
            TFREE T1
            TSTATUS R14
            ST_SHARED [R12 + 592], R14
            EXIT
        """
        after, _, _ = self.execute("failed_consumer", source, shared, warp_mask=3)
        wanted = shared.copy()
        wanted[128:132], wanted[148:152], wanted[152:156] = [0] * 4, [0x40] * 4, [0x44] * 4
        self.assertEqual(after, wanted)

    @unittest.skipUnless(shutil.which("verilator"), "requires Verilator")
    def test_verilator_full_pipeline_and_control_parity(self):
        output = self.output / "verilator"
        output.mkdir(parents=True, exist_ok=True)
        executable = (output / "nvfp4_sim").resolve()
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
        a = quantize([[float((r * 32 + k) % 19 - 9) for k in range(32)] for r in range(4)])
        b = quantize([[float((c * 32 + k) % 13 - 6) for k in range(32)] for c in range(8)])
        shared, memory, wanted, expected = images(a, b)
        result = self.execute("icarus_parity", program(), shared, memory)
        actual = run_global_program(
            executable, assemble(program()), shared, memory, output / "kernel", backend="verilator"
        )
        self.assertEqual(actual, result)
        self.assertEqual(actual[:2], (wanted, expected))
