"""Model input/output kernels; all lookup and arithmetic executes on the SM."""

from snackwell.kernels.inference import (
    Assembly,
    exp_negative,
    maximum,
    reciprocal,
    row_end,
    row_start,
    select,
)


def embedding(layout, rows, columns=4):
    a = Assembly(layout, "token embedding lookup plus positional embedding")
    if rows == 1:
        a.emit(f"XOR R15, R15, R15\nLD_SHARED R0, [R15 + {layout.address('ids')}]\nTID R12")
        a.scale_index(0, columns * 4)
        a.scale_index(12, 4)
        a.emit(f"ADD R0, R0, R12\nADDI R0, R0, {layout.address('table')}")
        for col in range(0, columns, 4):
            a.emit(f"""LD_SHARED R1, [R0 + {col * 4}]
                LD_SHARED R2, [R12 + {layout.address("position") + col * 4}]
                FADD R1, R1, R2
                ST_SHARED [R12 + {layout.address("out") + col * 4}], R1""")
        a.emit("EXIT")
        return a.text()
    a.emit("XOR R15, R15, R15\nTID R12\nTID R13\nTID R11")
    a.scale_index(12, 4)
    a.scale_index(13, columns * 4)
    a.scale_index(11, columns * 4)
    a.emit(f"""ADDI R12, R12, {layout.address("ids")}
        ADDI R13, R13, {layout.address("out")}
        ADDI R11, R11, {layout.address("position")}
        ADDI R14, R15, {rows // 4}
        row_group:
        LD_SHARED R0, [R12]""")
    a.scale_index(0, columns * 4)
    a.emit(f"ADDI R0, R0, {layout.address('table')}")
    for c in range(columns):
        a.emit(f"""LD_SHARED R1, [R0 + {c * 4}]
            LD_SHARED R2, [R11 + {c * 4}]
            FADD R1, R1, R2
            ST_SHARED [R13 + {c * 4}], R1""")
    a.emit(f"""ADDI R12, R12, 16
        ADDI R13, R13, {columns * 16}
        ADDI R11, R11, {columns * 16}
        ADDI R14, R14, -1
        BEQ R14, R15, done
        BEQ R15, R15, row_group
        done: EXIT""")
    return a.text()


def loop_softmax(layout, rows, cols, temperature=1.0):
    """Counted column loops keep a large vocabulary inside the program ROM."""
    a = Assembly(layout, f"stable in-place softmax over {cols} vocabulary entries")
    row_start(a, rows, cols, layout.address("x"), layout.address("x"))
    a.emit(f"""LD_SHARED R0, [R12]
        ADD R8, R12, R15
        ADDI R9, R15, {cols}
        max_loop:
        LD_SHARED R1, [R8]""")
    maximum(a, 0, 0, 1, 2, 3)
    a.emit(f"""ADDI R8, R8, 4
        ADDI R9, R9, -1
        BEQ R9, R15, max_done
        BEQ R15, R15, max_loop
        max_done:
        XOR R1, R1, R1
        ADD R8, R12, R15
        ADDI R9, R15, {cols}
        exp_loop:
        LD_SHARED R2, [R8]
        FSUB R2, R2, R0""")
    if temperature != 1.0:
        a.const(3, 1 / temperature)
        a.emit("FMUL R2, R2, R3")
    exp_negative(a, 2, 3, 4, 5, 6, 7)
    a.emit("""ST_SHARED [R8], R3
        FADD R1, R1, R3
        ADDI R8, R8, 4
        ADDI R9, R9, -1
        BEQ R9, R15, exp_done
        BEQ R15, R15, exp_loop
        exp_done:""")
    reciprocal(a, 1, 2, 3, 4)
    a.emit(f"""ADD R8, R12, R15
        ADDI R9, R15, {cols}
        scale_loop:
        LD_SHARED R3, [R8]
        FMUL R3, R3, R2
        ST_SHARED [R8], R3
        ADDI R8, R8, 4
        ADDI R9, R9, -1
        BEQ R9, R15, scale_done
        BEQ R15, R15, scale_loop
        scale_done:""")
    row_end(a, cols)
    return a.text()


def vocabulary_mask(layout, rows, cols, vocabulary):
    a = Assembly(layout, "mask padding outside the character vocabulary")
    row_start(a, rows, cols, layout.address("x"), layout.address("x"))
    if vocabulary < cols:
        a.const(0, word=0xFF800000)
        a.emit(f"""ADDI R8, R12, {vocabulary * 4}
            ADDI R9, R15, {cols - vocabulary}
            mask_loop:
            ST_SHARED [R8], R0
            ADDI R8, R8, 4
            ADDI R9, R9, -1
            BEQ R9, R15, mask_done
            BEQ R15, R15, mask_loop
            mask_done:""")
    row_end(a, cols)
    return a.text()


def argmax(layout, rows, cols):
    a = Assembly(layout, "greedy token selection; lowest index wins ties")
    row_start(a, rows, cols, layout.address("x"), layout.address("out"), reduced=True)
    a.emit(f"""LD_SHARED R0, [R12]
        XOR R1, R1, R1
        ADDI R8, R12, 4
        ADDI R9, R15, 1
        ADDI R10, R15, {cols}
        column_loop:
        LD_SHARED R2, [R8]
        FLT R3, R0, R2""")
    # select destroys its predicate; preserve it for the index update.
    a.emit("ADD R5, R3, R15")
    select(a, 0, 0, 2, 3, 4)
    select(a, 1, 1, 9, 5, 4)
    a.emit("""ADDI R8, R8, 4
        ADDI R9, R9, 1
        BEQ R9, R10, column_done
        BEQ R15, R15, column_loop
        column_done:
        ST_SHARED [R13], R1""")
    row_end(a, cols, reduced=True)
    return a.text()


def categorical(layout, rows, cols):
    """Select from cumulative probability using one supplied uniform draw per row."""
    a = Assembly(layout, "categorical sampling; host supplies uniform draws, SM selects IDs")
    row_start(a, rows, cols, layout.address("x"), layout.address("out"), reduced=True)
    a.emit(f"""XOR R0, R0, R0
        ADD R8, R12, R15
        ADDI R9, R15, {cols}
        sum_loop:
        LD_SHARED R2, [R8]
        FADD R0, R0, R2
        ADDI R8, R8, 4
        ADDI R9, R9, -1
        BEQ R9, R15, sum_done
        BEQ R15, R15, sum_loop
        sum_done:
        LD_SHARED R2, [R13 {layout.address("draws") - layout.address("out"):+d}]
        FMUL R0, R0, R2
        XOR R1, R1, R1
        XOR R5, R5, R5
        XOR R6, R6, R6
        ADD R8, R12, R15
        XOR R9, R9, R9
        ADDI R10, R15, {cols}
        sample_loop:
        LD_SHARED R2, [R8]
        FADD R1, R1, R2
        FLT R3, R15, R2
        ADDI R4, R15, 1
        XOR R4, R4, R5
        AND R3, R3, R4""")
    # Until crossing the threshold, retain the latest positive entry. This also
    # handles a threshold rounded to the total without selecting zero padding.
    select(a, 6, 6, 9, 3, 7)
    a.emit("""FLT R3, R0, R1
        OR R5, R5, R3
        ADDI R8, R8, 4
        ADDI R9, R9, 1
        BEQ R9, R10, sample_done
        BEQ R15, R15, sample_loop
        sample_done:
        ST_SHARED [R13], R6""")
    row_end(a, cols, reduced=True)
    return a.text()
