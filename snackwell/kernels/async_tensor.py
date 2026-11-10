"""Serial and overlapped copies for a 4-by-K times K-by-4 FP32 product."""

from snackwell.global_memory import GLOBAL_WORDS
from snackwell.numeric import bits
from snackwell.reference.tensor import matmul


def program(stages=3, *, overlap=True):
    if type(stages) is not int or not 1 <= stages <= 8:
        raise ValueError("expected 1 through 8 four-wide K tiles")
    lines = [
        "; Global -> two shared staging slots -> persistent tensor accumulation.",
        "; Launch warp 0 with all four lanes. Copy descriptors: shared byte 768.",
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
        "TALLOC T0",
    ]

    def submit(stage):
        slot = stage % 2
        lines.extend(
            (f"LD_SHARED R6, [R12 + {768 + stage * 16}]", f"CPASYNC R{8 + slot}, S{slot}, R6, 1")
        )

    submit(0)
    for stage in range(stages):
        slot = stage % 2
        lines.append(f"CPWAIT S{slot}, R{8 + slot}")
        if overlap and stage + 1 < stages:
            submit(stage + 1)
        lines.extend(
            (
                f"LD_SHARED R1, [R0 + {896 + slot * 8}]",
                f"LD_SHARED R2, [R0 + {900 + slot * 8}]",
                f"{'TMMA' if stage == 0 else 'TMMA_ACC'} T0, R1, R2",
                "TCOMMIT R3, T0, 1",
                f"CPRELEASE S{slot}",
                "TWAIT T0, R3",
            )
        )
        if stage + 1 < stages:
            lines.append("TRELEASE T0")
        if not overlap and stage + 1 < stages:
            submit(stage + 1)
    for row in range(4):
        lines.append(f"TLD R{4 + row}, T0, {row}")
    lines.extend(("TRELEASE T0", "TFREE T0"))
    for row in range(4):
        lines.append(f"ST_GLOBAL [R11 + {row * 16}], R{4 + row}")
    for op, offset in (("CPSTATUS", 576), ("TSTATUS", 592), ("GSTATUS", 608)):
        lines.extend((f"{op} R14", f"ST_SHARED [R12 + {offset}], R14"))
    lines.extend(("ST_SHARED [R12 + 624], R8", "ST_SHARED [R12 + 640], R9", "EXIT"))
    return "\n".join(lines) + "\n"


def images(stages=3, *, a=None, b=None, source_stride=16, destination_stride=16):
    program(stages)  # Validate the supported shape.
    if source_stride not in (16, 20, 24, 28, 32) or destination_stride not in (16, 20, 24, 28, 32):
        raise ValueError("demo strides must be aligned, from 16 through 32 bytes")
    k_size = 4 * stages
    a = [bits(i + 1) for i in range(4 * k_size)] if a is None else list(a)
    b = [bits((i % 7) - 3) for i in range(k_size * 4)] if b is None else list(b)
    if len(a) != 4 * k_size or len(b) != k_size * 4:
        raise ValueError("incorrect A/B shape")
    shared, memory = [0xDEADBEEF] * 1024, [0xDEADBEEF] * GLOBAL_WORDS
    wanted_shared = shared.copy()
    expected = [0] * 16
    for stage in range(stages):
        slot, k = stage % 2, stage * 4
        aa = [a[row * k_size + k + col] for row in range(4) for col in range(4)]
        bb = b[k * 4 : (k + 4) * 4]
        expected = matmul(aa, bb, expected)
        source_base = 0x4000 + stage * 8 * source_stride
        for row in range(8):
            start = (source_base + row * source_stride) // 4
            values = (aa + bb)[row * 4 : row * 4 + 4]
            memory[start : start + 4] = values
            dest = (slot * 256 + row * destination_stride) // 4
            wanted_shared[dest : dest + 4] = values
        shared[192 + stage * 4 : 196 + stage * 4] = [
            source_base,
            source_stride,
            0x0408,
            (destination_stride << 12) | (slot * 256),
        ]
    for slot in (0, 1):
        shared[224 + slot * 2 : 226 + slot * 2] = [
            (destination_stride << 12) | (slot * 256),
            (destination_stride << 12) | (slot * 256 + 4 * destination_stride),
        ]
    wanted_shared[192:228] = shared[192:228]
    wanted_shared[144:156] = [0] * 12
    wanted_shared[156:160], wanted_shared[160:164] = [(stages + 1) // 2] * 4, [stages // 2] * 4
    wanted_global = memory.copy()
    wanted_global[6144:6160] = expected
    return shared, memory, wanted_shared, wanted_global
