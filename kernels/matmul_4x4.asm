; C = A * B, one row-major 4x4 FP32 tile through the tensor unit.
; Launch ONE warp with all four lanes active. Any warp ID is supported.
; Shared-memory byte bases: A=0, B=64, C=128. No register preload.
; Every input row must be loaded in full before TMUL reads the register tiles.

    XOR       R15, R15, R15
    ADDI      R14, R15, 3
    TID       R12
    AND       R12, R12, R14  ; column = local thread index & 3
    ADD       R12, R12, R12
    ADD       R12, R12, R12  ; byte offset = 4 * column

    LD_SHARED R0, [R12]
    LD_SHARED R1, [R12 + 16]
    LD_SHARED R2, [R12 + 32]
    LD_SHARED R3, [R12 + 48]
    LD_SHARED R4, [R12 + 64]
    LD_SHARED R5, [R12 + 80]
    LD_SHARED R6, [R12 + 96]
    LD_SHARED R7, [R12 + 112]

    TMUL      R8, R0, R4     ; A: R0..R3, B: R4..R7, C: R8..R11

    ST_SHARED [R12 + 128], R8
    ST_SHARED [R12 + 144], R9
    ST_SHARED [R12 + 160], R10
    ST_SHARED [R12 + 176], R11
    EXIT
