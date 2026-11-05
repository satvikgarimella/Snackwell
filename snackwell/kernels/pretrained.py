"""Bounded assembly kernels for pretrained decoder operations.

Large reductions carry their FP32 state between launches. Vector kernels process
four independent elements at a time; the host only supplies words and constants.
"""

from snackwell.kernels.inference import Assembly, exp_negative, maximum, reciprocal, rsqrt, select


def reduce(layout, count, kind):
    a = Assembly(layout, f"streamed {kind} reduction, {count} words")
    a.emit(f"""XOR R15, R15, R15
        LD_SHARED R0, [R15 + {layout.address("state")}]
        ADDI R8, R15, {layout.address("x")}
        ADDI R9, R15, {count}
        loop:
        LD_SHARED R1, [R8]""")
    if kind == "max":
        maximum(a, 0, 0, 1, 2, 3)
    else:
        if kind == "squares":
            a.emit("FMUL R1, R1, R1")
        elif kind != "sum":
            raise ValueError(kind)
        a.emit("FADD R0, R0, R1")
    a.emit(f"""ADDI R8, R8, 4
        ADDI R9, R9, -1
        BEQ R9, R15, done
        BEQ R15, R15, loop
        done:
        ST_SHARED [R15 + {layout.address("state")}], R0
        EXIT""")
    return a.text()


def inverse(layout, kind, width=1, epsilon=0.0):
    a = Assembly(layout, f"{kind} of a completed reduction")
    a.emit(f"XOR R15, R15, R15\nLD_SHARED R0, [R15 + {layout.address('state')}]")
    if kind == "rms_inverse":
        a.const(1, 1 / width)
        a.emit("FMUL R0, R0, R1")
        a.const(1, epsilon)
        a.emit("FADD R0, R0, R1")
        rsqrt(a, 0, 2, 3, 4, 5)
    elif kind == "reciprocal":
        reciprocal(a, 0, 2, 3, 4)
    else:
        raise ValueError(kind)
    a.emit(f"ST_SHARED [R15 + {layout.address('state')}], R2\nEXIT")
    return a.text()


def vector(layout, count, kind):
    if count < 4 or count % 4:
        raise ValueError("vector launches require a positive multiple of four elements")
    a = Assembly(layout, f"{kind}, {count} elements")
    a.emit("XOR R15, R15, R15\nTID R12")
    a.scale_index(12, 4)
    a.emit(f"ADDI R14, R15, {count // 4}\nloop:\nLD_SHARED R0, [R12 + {layout.address('x')}]")
    if kind in ("add", "multiply"):
        a.emit(f"LD_SHARED R1, [R12 + {layout.address('other')}]")
        op = "FADD" if kind == "add" else "FMUL"
        a.emit(f"{op} R0, R0, R1")
    elif kind == "scale":
        a.emit(f"LD_SHARED R1, [R15 + {layout.address('scalar')}]\nFMUL R0, R0, R1")
    elif kind == "norm":
        a.emit(f"""LD_SHARED R1, [R15 + {layout.address("scalar")}]
            LD_SHARED R2, [R12 + {layout.address("other")}]
            FMUL R0, R0, R1
            FMUL R0, R0, R2""")
    elif kind == "rope":
        a.emit(f"""LD_SHARED R1, [R12 + {layout.address("other")}]
            LD_SHARED R2, [R12 + {layout.address("cos")}]
            LD_SHARED R3, [R12 + {layout.address("sin")}]
            FMUL R0, R0, R2
            FMUL R1, R1, R3
            FADD R0, R0, R1""")
    elif kind == "exp":
        a.emit(f"""LD_SHARED R1, [R15 + {layout.address("maximum")}]
            FSUB R0, R0, R1
            LD_SHARED R1, [R15 + {layout.address("temperature_inverse")}]
            FMUL R0, R0, R1""")
        exp_negative(a, 0, 1, 2, 3, 4, 5)
        a.emit("ADD R0, R1, R15")
    elif kind == "swiglu":
        a.emit(f"LD_SHARED R7, [R12 + {layout.address('other')}]")
        a.const(6, word=0x7FFFFFFF)
        a.emit("AND R1, R0, R6")
        a.const(6, word=0x80000000)
        a.emit("XOR R1, R1, R6")
        exp_negative(a, 1, 2, 3, 4, 5, 6)
        a.const(3, 1)
        a.emit("FADD R3, R2, R3")
        reciprocal(a, 3, 4, 5, 6)
        a.emit("FMUL R3, R4, R2\nFLT R5, R0, R15")
        select(a, 4, 4, 3, 5, 6)
        a.emit("FMUL R0, R0, R4\nFMUL R0, R0, R7")
    else:
        raise ValueError(kind)
    a.emit(f"""ST_SHARED [R12 + {layout.address("x")}], R0
        ADDI R12, R12, 16
        ADDI R14, R14, -1
        BEQ R14, R15, done
        BEQ R15, R15, loop
        done: EXIT""")
    return a.text()


def argmax(layout, count, offset):
    a = Assembly(layout, "streamed argmax; the first maximum wins ties")
    a.emit(f"""XOR R15, R15, R15
        LD_SHARED R0, [R15 + {layout.address("state")}]
        LD_SHARED R1, [R15 + {layout.address("state") + 4}]
        ADDI R8, R15, {layout.address("x")}
        ADDI R9, R15, {count}""")
    a.const(10, word=offset)
    a.emit("loop:\nLD_SHARED R2, [R8]\nFLT R3, R0, R2\nADD R5, R3, R15")
    select(a, 0, 0, 2, 3, 4)
    select(a, 1, 1, 10, 5, 4)
    a.emit(f"""ADDI R8, R8, 4
        ADDI R10, R10, 1
        ADDI R9, R9, -1
        BEQ R9, R15, done
        BEQ R15, R15, loop
        done:
        ST_SHARED [R15 + {layout.address("state")}], R0
        ST_SHARED [R15 + {layout.address("state") + 4}], R1
        EXIT""")
    return a.text()


def categorical(layout, count, offset):
    a = Assembly(layout, "streamed categorical scan with an FP32 threshold")
    a.emit(f"""XOR R15, R15, R15
        LD_SHARED R0, [R15 + {layout.address("state")}]
        LD_SHARED R2, [R15 + {layout.address("state") + 4}]
        LD_SHARED R3, [R15 + {layout.address("state") + 8}]
        LD_SHARED R6, [R15 + {layout.address("threshold")}]
        ADDI R8, R15, {layout.address("x")}
        ADDI R9, R15, {count}""")
    a.const(7, word=0xFFFFFFFF)
    a.const(10, word=offset)
    a.emit("""loop:
        LD_SHARED R1, [R8]
        FADD R0, R0, R1
        FLT R4, R15, R1""")
    select(a, 3, 3, 10, 4, 5)
    a.emit("""BEQ R2, R7, check_threshold
        BEQ R15, R15, advance
        check_threshold:
        FLT R4, R6, R0""")
    select(a, 2, 2, 10, 4, 5)
    a.emit(f"""advance:
        ADDI R8, R8, 4
        ADDI R10, R10, 1
        ADDI R9, R9, -1
        BEQ R9, R15, done
        BEQ R15, R15, loop
        done:
        ST_SHARED [R15 + {layout.address("state")}], R0
        ST_SHARED [R15 + {layout.address("state") + 4}], R2
        ST_SHARED [R15 + {layout.address("state") + 8}], R3
        EXIT""")
    return a.text()


def lookup(layout, columns):
    a = Assembly(layout, "embedding lookup within a supplied table page")
    a.emit(f"XOR R15, R15, R15\nLD_SHARED R0, [R15 + {layout.address('ids')}]\nTID R12")
    a.scale_index(0, columns * 4)
    a.scale_index(12, 4)
    a.emit(f"ADD R0, R0, R12\nADDI R0, R0, {layout.address('table')}")
    for col in range(0, columns, 4):
        a.emit(f"""LD_SHARED R1, [R0 + {col * 4}]
            ST_SHARED [R12 + {layout.address("out") + col * 4}], R1""")
    a.emit("EXIT")
    return a.text()
