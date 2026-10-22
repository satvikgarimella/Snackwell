"""Loss, backward and SGD assembly; every numerical operation executes on the SM."""

import math

from snackwell.kernels.inference import Assembly, exp_negative, reciprocal, select


def gelu_gradient(a):
    """R0=x, R11=upstream gradient; return dL/dx in R0. Preserve R12..R15."""
    a.const(1, word=0x7FFFFFFF)
    a.emit("AND R1, R0, R1")
    a.const(2, 10)
    a.emit("FLT R3, R2, R1")
    select(a, 1, 1, 2, 3, 4)
    # Bound the derivative's intermediates just as the forward GELU bounds u.
    a.const(2, word=0x80000000)
    a.emit("AND R0, R0, R2\nOR R0, R0, R1\nFMUL R2, R1, R1")
    a.const(3, 0.044715)
    a.emit("FMUL R2, R2, R3")
    a.const(3, 1)
    a.emit("FADD R2, R2, R3\nFMUL R2, R2, R1")
    a.const(3, -2 * math.sqrt(2 / math.pi))
    a.emit("FMUL R2, R2, R3")
    exp_negative(a, 2, 3, 4, 5, 6, 7)
    a.const(4, 1)
    a.emit("FADD R4, R4, R3")
    reciprocal(a, 4, 5, 6, 7)
    # GELU(x)=x*s(v), v=2*sqrt(2/pi)*(x+0.044715*x^3).
    # Compute s*(1-s) as e/(1+e)^2 to avoid cancellation in the tails.
    a.emit("FMUL R6, R3, R5\nFLT R7, R0, R15")
    select(a, 6, 5, 6, 7, 8)
    a.emit("FMUL R3, R3, R5\nFMUL R3, R3, R5\nFMUL R2, R1, R1")
    a.const(4, 3 * 0.044715)
    a.emit("FMUL R2, R2, R4")
    a.const(4, 1)
    a.emit("FADD R2, R2, R4")
    a.const(4, 2 * math.sqrt(2 / math.pi))
    a.emit("""FMUL R2, R2, R4
        FMUL R2, R2, R3
        FMUL R2, R2, R0
        FADD R0, R6, R2
        FMUL R0, R0, R11""")


def vector(layout, kind, count, scale=1.0):
    """Four lanes own consecutive elements; counted loop keeps ROM use bounded."""
    if count < 4 or count % 4:
        raise ValueError("training vectors require a positive multiple of four elements")
    a = Assembly(layout, f"{kind}: {count} elements")
    a.emit("XOR R15, R15, R15\nTID R12")
    a.scale_index(12, 4)
    a.emit(f"""ADDI R14, R15, {count // 4}
        elements:
        LD_SHARED R0, [R12 + {layout.address("x")}]
        LD_SHARED R11, [R12 + {layout.address("b")}]""")
    if kind == "mse_gradient":
        a.const(1, scale)
        a.emit("FSUB R0, R0, R11\nFMUL R0, R0, R1")
    elif kind == "sgd":
        a.const(1, scale)
        a.emit("FMUL R11, R11, R1\nFSUB R0, R0, R11")
    elif kind == "gelu_backward":
        gelu_gradient(a)
    elif kind == "add":
        a.emit("FADD R0, R0, R11")
    elif kind == "scale":
        a.const(1, scale)
        a.emit("FMUL R0, R0, R1")
    else:
        raise ValueError(kind)
    a.emit(f"""ST_SHARED [R12 + {layout.address("out")}], R0
        ADDI R12, R12, 16
        ADDI R14, R14, -1
        BEQ R14, R15, done
        BEQ R15, R15, elements
        done: EXIT""")
    return a.text()


def squared_loss(layout, count, scale):
    """One active lane accumulates squared errors in element order."""
    a = Assembly(layout, f"squared loss: {count} elements, scale {scale}")
    a.emit(f"""XOR R15, R15, R15
        XOR R12, R12, R12
        ADDI R14, R15, {count}
        LD_SHARED R2, [R15 + {layout.address("initial")}]
        elements:
        LD_SHARED R0, [R12 + {layout.address("x")}]
        LD_SHARED R1, [R12 + {layout.address("b")}]
        FSUB R0, R0, R1
        FMUL R0, R0, R0
        FADD R2, R2, R0
        ADDI R12, R12, 4
        ADDI R14, R14, -1
        BEQ R14, R15, done
        BEQ R15, R15, elements
        done:""")
    a.const(1, scale)
    a.emit(f"FMUL R2, R2, R1\nST_SHARED [R15 + {layout.address('out')}], R2\nEXIT")
    return a.text()


def column_sum(layout, rows, cols):
    """Reduce rows for a bias gradient, continuing an optional previous chunk."""
    a = Assembly(layout, f"column sum: {rows} rows, {cols} columns")
    a.emit("XOR R15, R15, R15\nTID R12")
    a.scale_index(12, 4)
    a.emit(f"""ADDI R14, R15, {cols // 4}
        columns:
        LD_SHARED R0, [R12 + {layout.address("initial")}]""")
    for row in range(rows):
        a.emit(f"LD_SHARED R1, [R12 + {layout.address('x') + row * cols * 4}]\nFADD R0, R0, R1")
    a.emit(f"""ST_SHARED [R12 + {layout.address("out")}], R0
        ADDI R12, R12, 16
        ADDI R14, R14, -1
        BEQ R14, R15, done
        BEQ R15, R15, columns
        done: EXIT""")
    return a.text()
