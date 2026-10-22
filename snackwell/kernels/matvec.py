"""One-row matrix products using the four FP32 SIMD lanes."""

from snackwell.kernels.inference import Assembly


def simd_matvec(layout, n, k, *, transpose_b=False, accumulate=False, warps=1, bias=None):
    """Match TMUL's four-term sums and their accumulation order exactly.

    Each lane owns one output column. All lanes read the same vector element;
    shared memory broadcasts that read. Independent products occupy different
    registers so subsequent loads can overlap multiplication.
    """
    if any(type(size) is not int or size < 4 or size % 4 for size in (n, k)):
        raise ValueError("matvec dimensions must be positive multiples of four")
    if type(warps) is not int or not 1 <= warps <= min(4, n // 4):
        raise ValueError("matvec needs 1..4 warps and four output columns per warp")
    a = Assembly(layout, f"SIMD FP32 matvec 1x{k} @ {k}x{n}; B transposed={transpose_b}")
    a.lines[1] = f"; Launch warps 0..{warps - 1}, each with four active lanes."
    a.emit("XOR R15, R15, R15")
    if warps > 1:
        a.emit("TID R0\nADDI R1, R15, 2\nSHR R0, R0, R1")
        for warp in range(warps):
            a.emit(f"ADDI R1, R15, {warp}\nBEQ R0, R1, warp_{warp}")
        a.emit("EXIT")
    for warp in range(warps):
        if warps > 1:
            a.emit(f"warp_{warp}:")
        for col in range(4 * warp, n, 4 * warps):
            a.emit("TID R13")
            if warp:
                a.emit(f"ADDI R13, R13, {-4 * warp}")
            a.emit("ADD R14, R13, R15")
            a.scale_index(14, 4)
            a.emit(f"ADDI R14, R14, {layout.address('out') + col * 4}")
            a.scale_index(13, k * 4 if transpose_b else 4)
            a.emit(f"ADDI R13, R13, {layout.address('b') + col * (k if transpose_b else 1) * 4}")
            a.emit(f"ADDI R12, R15, {layout.address('a')}")
            a.emit("LD_SHARED R8, [R14]" if accumulate else "XOR R8, R8, R8")
            a.emit(f"ADDI R9, R15, {k // 4}\ncolumn_{col}:\nXOR R0, R0, R0")
            for term, (left, right) in enumerate(((1, 2), (3, 4), (5, 6), (7, 10))):
                a.emit(f"""LD_SHARED R{left}, [R12 + {4 * term}]
                    LD_SHARED R{right}, [R13 + {4 * term * (1 if transpose_b else n)}]
                    FMUL R{left}, R{left}, R{right}""")
            for product in (1, 3, 5, 7):
                a.emit(f"FADD R0, R0, R{product}")
            a.emit(f"""FADD R8, R8, R0
                ADDI R12, R12, 16
                ADDI R13, R13, {16 * (1 if transpose_b else n)}
                ADDI R9, R9, -1
                BEQ R9, R15, column_{col}_done
                BEQ R15, R15, column_{col}
                column_{col}_done:""")
            if bias is not None:
                offset = layout.address(bias) - layout.address("out")
                a.emit(f"LD_SHARED R0, [R14 {offset:+d}]\nFADD R8, R8, R0")
            a.emit("ST_SHARED [R14], R8")
        a.emit("EXIT")
    return a.text()
