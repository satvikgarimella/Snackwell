"""Packed NVFP4 GEMM through external memory, async staging, and tensor memory."""

from snackwell.global_memory import GLOBAL_WORDS
from snackwell.numeric import SENTINEL
from snackwell.reference.nvfp4 import matmul


def jobs(rows, columns, k_size):
    if any(type(x) is not int for x in (rows, columns, k_size)):
        raise ValueError("matrix dimensions must be integers")
    if rows not in (4, 8) or columns not in (4, 8) or k_size not in (16, 32, 48, 64):
        raise ValueError("demo supports M/N 4 or 8 and K 16, 32, 48, or 64")
    return [
        (r, c, g)
        for r in range(0, rows, 4)
        for c in range(0, columns, 4)
        for g in range(k_size // 16)
    ]


def program(rows=4, columns=8, k_size=32, *, overlap=True):
    schedule = jobs(rows, columns, k_size)
    lines = [
        "; NVFP4: A rows and B columns contain sixteen elements per scale.",
        "; Each vector is two packed words plus one zero-padded scale word.",
        "XOR R0, R0, R0",
        "TID R12",
        "ADD R12, R12, R12",
        "ADD R12, R12, R12",
        "ADDI R11, R0, 4096",
        "ADD R11, R11, R11",
        "ADD R11, R11, R11",
        "ADDI R11, R11, 8191",
        "ADDI R11, R11, 1",
        "ADD R11, R11, R12",
        "LD_SHARED R1, [R0 + 1168]",
        "LD_SHARED R2, [R0 + 1172]",
        "FMUL R10, R1, R2",
        "TALLOC T0",
    ]

    def submit(stage):
        slot = stage % 2
        lines.extend(
            (f"LD_SHARED R6, [R12 + {768 + stage * 16}]", f"CPASYNC R{8 + slot}, S{slot}, R6, 1")
        )

    submit(0)
    for stage, (row_base, column_base, group) in enumerate(schedule):
        slot = stage % 2
        lines.append(f"CPWAIT S{slot}, R{8 + slot}")
        if overlap and stage + 1 < len(schedule):
            submit(stage + 1)
        lines.extend(
            (
                f"LD_SHARED R1, [R0 + {1152 + slot * 8}]",
                f"LD_SHARED R2, [R0 + {1156 + slot * 8}]",
                f"{'TMMA_NVFP4' if group == 0 else 'TMMA_NVFP4_ACC'} T0, R1, R2",
                "TCOMMIT R3, T0, 1",
                f"CPRELEASE S{slot}",
                "TWAIT T0, R3",
            )
        )
        if group == k_size // 16 - 1:
            for row in range(4):
                offset = ((row_base + row) * columns + column_base) * 4
                lines.extend(
                    (f"TLD R4, T0, {row}", "FMUL R4, R4, R10", f"ST_GLOBAL [R11 + {offset}], R4")
                )
        lines.append("TRELEASE T0")
        if not overlap and stage + 1 < len(schedule):
            submit(stage + 1)
    lines.append("TFREE T0")
    for opcode, offset in (("CPSTATUS", 576), ("TSTATUS", 592), ("GSTATUS", 608)):
        lines.extend((f"{opcode} R14", f"ST_SHARED [R12 + {offset}], R14"))
    lines.extend(("ST_SHARED [R12 + 624], R8", "ST_SHARED [R12 + 640], R9", "EXIT"))
    return "\n".join(lines) + "\n"


def images(a, b, *, source_stride=12, destination_stride=12):
    rows, columns, k_size = len(a.codes), len(b.codes), len(a.codes[0])
    schedule = jobs(rows, columns, k_size)
    if any(len(vector) != k_size for vector in a.codes + b.codes):
        raise ValueError("matrix inner dimensions must agree")
    if source_stride not in (12, 16, 20, 28, 32) or destination_stride not in (12, 16, 20, 28, 32):
        raise ValueError("demo strides must be aligned and between 12 and 32 bytes")
    shared, memory = [SENTINEL] * 1024, [SENTINEL] * GLOBAL_WORDS
    wanted_shared = shared.copy()
    for stage, (row, column, group) in enumerate(schedule):
        packed = a.group_words(row, group) + b.group_words(column, group)
        source = 0x4000 + stage * 8 * source_stride
        destination = (stage % 2) * 256
        for vector in range(8):
            words = packed[vector * 3 : (vector + 1) * 3]
            index = (source + vector * source_stride) // 4
            memory[index : index + 3] = words
            index = (destination + vector * destination_stride) // 4
            wanted_shared[index : index + 3] = words
        shared[192 + stage * 4 : 196 + stage * 4] = [
            source,
            source_stride,
            0x0308,
            (destination_stride << 12) | destination,
        ]
    for slot in (0, 1):
        shared[288 + slot * 2 : 290 + slot * 2] = [
            (destination_stride << 12) | (slot * 256),
            (destination_stride << 12) | (slot * 256 + 4 * destination_stride),
        ]
    shared[292:294] = [a.tensor_scale, b.tensor_scale]
    wanted_shared[192:256] = shared[192:256]
    wanted_shared[288:294] = shared[288:294]
    wanted_shared[144:156] = [0] * 12
    wanted_shared[156:160] = [(len(schedule) + 1) // 2] * 4
    wanted_shared[160:164] = [len(schedule) // 2] * 4
    wanted_global = memory.copy()
    wanted_global[6144 : 6144 + rows * columns] = matmul(a, b)
    return shared, memory, wanted_shared, wanted_global
