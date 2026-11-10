"""Native NVFP4 linear tiles in the model runner's shared-memory transport.

One launch owns four rows, up to sixteen columns, and up to 128 K elements.
K chunks carry the unscaled accumulator through TST; only the last chunk
applies the per-row tensor factors and bias using SIMD FP32 instructions.
"""

from snackwell.kernels.inference import Assembly


def program(layout, columns, inner, *, accumulate=False, final=True, bias=False):
    if columns not in (4, 8, 12, 16) or inner not in range(16, 129, 16):
        raise ValueError("NVFP4 launch needs 4..16 columns and 16..128 inner elements")
    groups = inner // 16
    a = Assembly(layout, "native NVFP4 linear projection; FP32 accumulation and epilogue")
    a.emit("XOR R15, R15, R15\nTID R12\nADD R12, R12, R12\nADD R12, R12, R12\nTALLOC T0")
    for column in range(0, columns, 4):
        if accumulate:
            for row in range(4):
                address = layout.address("initial") + (row * columns + column) * 4
                a.emit(f"LD_SHARED R4, [R12 + {address}]\nTST T0, {row}, R4")
        for group in range(groups):
            left = layout.address("a") + group * 48
            right = layout.address("b") + (column // 4 * groups + group) * 48
            a.const(1, word=(12 << 12) | left)
            a.const(2, word=(12 << 12) | right)
            opcode = "TMMA_NVFP4_ACC" if accumulate or group else "TMMA_NVFP4"
            a.emit(f"{opcode} T0, R1, R2\nTCOMMIT R3, T0, 1\nTWAIT T0, R3")
            if group + 1 < groups:
                a.emit("TRELEASE T0")
        if final:
            a.emit(f"LD_SHARED R2, [R15 + {layout.address('scale_b')}]")
            if bias:
                a.emit(f"LD_SHARED R5, [R12 + {layout.address('bias') + column * 4}]")
        for row in range(4):
            a.emit(f"TLD R4, T0, {row}")
            if final:
                a.emit(f"LD_SHARED R1, [R15 + {layout.address('scale_a') + row * 4}]")
                a.emit("FMUL R10, R1, R2\nFMUL R4, R4, R10")
                if bias:
                    a.emit("FADD R4, R4, R5")
            address = layout.address("out") + (row * columns + column) * 4
            a.emit(f"ST_SHARED [R12 + {address}], R4")
        a.emit("TRELEASE T0")
    a.emit("TFREE T0\nTSTATUS R14\nEXIT")
    return a.text()
