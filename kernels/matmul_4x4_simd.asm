; Same C = A * B tile, using the four-lane SIMD FP unit instead of TMUL.
; Launch four warps, all lanes active: warp=row, lane=column of C.
; Shared-memory byte bases: A=0, B=64, C=128. No register preload.
; Each lane accumulates +0 + round32(A[row][k] * B[k][column]), k=0..3.
; Multiplication and addition round separately, exactly as in the tensor unit.

    XOR       R15, R15, R15
    TID       R0
    ADDI      R1, R15, 12
    AND       R2, R0, R1     ; row * 4
    ADD       R2, R2, R2
    ADD       R2, R2, R2     ; A row byte address
    ADDI      R1, R15, 3
    AND       R3, R0, R1     ; column
    ADD       R3, R3, R3
    ADD       R3, R3, R3
    ADDI      R3, R3, 64     ; B[0][column] byte address
    ADD       R4, R0, R0
    ADD       R4, R4, R4
    ADDI      R4, R4, 128    ; C[row][column] byte address
    XOR       R6, R6, R6     ; accumulator = positive FP32 zero

    LD_SHARED R7, [R2]
    LD_SHARED R8, [R3]
    FMUL      R9, R7, R8
    FADD      R6, R6, R9

    LD_SHARED R7, [R2 + 4]
    LD_SHARED R8, [R3 + 16]
    FMUL      R9, R7, R8
    FADD      R6, R6, R9

    LD_SHARED R7, [R2 + 8]
    LD_SHARED R8, [R3 + 32]
    FMUL      R9, R7, R8
    FADD      R6, R6, R9

    LD_SHARED R7, [R2 + 12]
    LD_SHARED R8, [R3 + 48]
    FMUL      R9, R7, R8
    FADD      R6, R6, R9

    ST_SHARED [R4], R6
    EXIT
