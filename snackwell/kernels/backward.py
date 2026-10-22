"""Counted-loop normalization/attention derivatives and stable cross-entropy."""

import math

from snackwell.kernels.inference import Assembly, exp_negative, maximum, reciprocal, rsqrt
from snackwell.numeric import bits


def row_setup(a, cols):
    a.emit("XOR R15, R15, R15\nTID R12")
    a.scale_index(12, cols * 4)


def loop_start(a, label, cols):
    a.emit(f"XOR R13, R13, R13\nADDI R14, R15, {cols}\n{label}:\nADD R6, R12, R13")


def loop_end(a, label):
    a.emit(f"""ADDI R13, R13, 4
        ADDI R14, R14, -1
        BEQ R14, R15, {label}_done
        BEQ R15, R15, {label}
        {label}_done:""")


def layernorm_backward(layout, rows, cols, epsilon):
    """One lane per row. Output contains dx followed by per-row dgamma."""
    a = Assembly(layout, f"LayerNorm backward: {rows} rows, {cols} features")
    row_setup(a, cols)
    a.emit("XOR R8, R8, R8")
    loop_start(a, "mean", cols)
    a.emit(f"LD_SHARED R0, [R6 + {layout.address('x')}]\nFADD R8, R8, R0")
    loop_end(a, "mean")
    a.const(7, 1 / cols)
    a.emit("FMUL R8, R8, R7\nXOR R0, R0, R0")
    loop_start(a, "variance", cols)
    a.emit(f"""LD_SHARED R1, [R6 + {layout.address("x")}]
        FSUB R1, R1, R8
        FMUL R1, R1, R1
        FADD R0, R0, R1""")
    loop_end(a, "variance")
    a.emit("FMUL R0, R0, R7")
    a.const(7, epsilon)
    a.emit("FADD R0, R0, R7")
    rsqrt(a, 0, 9, 10, 11, 7)
    a.emit("XOR R10, R10, R10\nXOR R11, R11, R11")
    loop_start(a, "reduce", cols)
    a.emit(f"""LD_SHARED R0, [R6 + {layout.address("x")}]
        FSUB R0, R0, R8
        FMUL R0, R0, R9
        LD_SHARED R1, [R6 + {layout.address("dy")}]
        LD_SHARED R2, [R13 + {layout.address("gamma")}]
        FMUL R3, R1, R2
        FADD R10, R10, R3
        FMUL R4, R3, R0
        FADD R11, R11, R4
        FMUL R4, R1, R0
        ST_SHARED [R6 + {layout.address("out") + rows * cols * 4}], R4""")
    loop_end(a, "reduce")
    a.const(7, 1 / cols)
    a.emit("FMUL R10, R10, R7\nFMUL R11, R11, R7")
    loop_start(a, "derivative", cols)
    a.emit(f"""LD_SHARED R0, [R6 + {layout.address("x")}]
        FSUB R0, R0, R8
        FMUL R0, R0, R9
        LD_SHARED R1, [R6 + {layout.address("dy")}]
        LD_SHARED R2, [R13 + {layout.address("gamma")}]
        FMUL R1, R1, R2
        FSUB R1, R1, R10
        FMUL R0, R0, R11
        FSUB R1, R1, R0
        FMUL R1, R1, R9
        ST_SHARED [R6 + {layout.address("out")}], R1""")
    loop_end(a, "derivative")
    a.emit("EXIT")
    return a.text()


def softmax_backward(layout, cols, scale):
    a = Assembly(layout, f"softmax derivative, scaled by {scale}")
    row_setup(a, cols)
    a.emit("XOR R8, R8, R8")
    loop_start(a, "dot", cols)
    a.emit(f"""LD_SHARED R0, [R6 + {layout.address("p")}]
        LD_SHARED R1, [R6 + {layout.address("dy")}]
        FMUL R0, R0, R1
        FADD R8, R8, R0""")
    loop_end(a, "dot")
    a.const(7, scale)
    loop_start(a, "derivative", cols)
    a.emit(f"""LD_SHARED R0, [R6 + {layout.address("p")}]
        LD_SHARED R1, [R6 + {layout.address("dy")}]
        FSUB R1, R1, R8
        FMUL R0, R0, R1
        FMUL R0, R0, R7
        ST_SHARED [R6 + {layout.address("out")}], R0""")
    loop_end(a, "derivative")
    a.emit("EXIT")
    return a.text()


def cross_entropy(layout, cols):
    """Loss=max(logits)-target+log(sum(exp(logits-max))); no log(probability)."""
    # The sum is in [1,cols], including the uniform-logit upper endpoint.
    # Keep the original allocation for small classifiers to preserve their layout.
    entries = max(8, cols.bit_length())
    log2 = layout.allocate("log2", entries, [bits(i * math.log(2)) for i in range(entries)])
    a = Assembly(layout, f"stable cross entropy over {cols} padded classes")
    row_setup(a, cols)
    a.emit(f"LD_SHARED R10, [R12 + {layout.address('x')}]")
    loop_start(a, "maximum", cols)
    a.emit(f"LD_SHARED R0, [R6 + {layout.address('x')}]")
    maximum(a, 10, 10, 0, 1, 2)
    loop_end(a, "maximum")
    a.emit("XOR R11, R11, R11\nADD R8, R12, R15")
    a.emit(f"ADDI R9, R15, {cols}\nexponentials:")
    a.emit(f"LD_SHARED R2, [R8 + {layout.address('x')}]\nFSUB R2, R2, R10")
    exp_negative(a, 2, 3, 4, 5, 6, 7)
    a.emit("""FADD R11, R11, R3
        ADDI R8, R8, 4
        ADDI R9, R9, -1
        BEQ R9, R15, exponentials_done
        BEQ R15, R15, exponentials
        exponentials_done:
        TID R13""")
    a.scale_index(13, 4)
    a.emit(f"LD_SHARED R0, [R13 + {layout.address('targets')}]")
    a.scale_index(0, 4)
    a.emit(f"ADD R0, R0, R12\nLD_SHARED R0, [R0 + {layout.address('x')}]\nFSUB R10, R10, R0")
    a.emit("ADDI R0, R15, 23\nSHR R1, R11, R0\nADDI R1, R1, -127")
    a.scale_index(1, 4)
    a.emit(f"LD_SHARED R9, [R1 + {log2}]")
    a.const(0, word=0x007FFFFF)
    a.const(2, 1)
    a.emit("AND R1, R11, R0\nOR R1, R1, R2\nFSUB R3, R1, R2\nFADD R4, R1, R2")
    reciprocal(a, 4, 5, 6, 7)
    a.emit("FMUL R3, R3, R5\nFMUL R4, R3, R3")
    # log(m)=2*(t+t^3/3+...+t^13/13), t=(m-1)/(m+1), 0<=t<1/3.
    a.const(5, 1 / 13)
    for denominator in (11, 9, 7, 5, 3, 1):
        a.const(7, 1 / denominator)
        a.emit("FMUL R5, R5, R4\nFADD R5, R5, R7")
    a.emit("FMUL R5, R5, R3\nFADD R5, R5, R5\nFADD R5, R5, R9\nFADD R5, R5, R10")
    a.emit(f"ST_SHARED [R13 + {layout.address('out')}], R5\nEXIT")
    return a.text()
