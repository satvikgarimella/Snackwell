; C[i] = A[i] * B[i], 16 FP32 elements, four warps of four lanes.
; Shared-memory byte bases: A=0, B=64, C=128. No register preload.

    TID       R1
    ADD       R1, R1, R1
    ADD       R1, R1, R1
    ADDI      R2, R1, 64
    ADDI      R3, R1, 128
    LD_SHARED R4, [R1]
    LD_SHARED R5, [R2]
    FMUL      R6, R4, R5
    ST_SHARED [R3], R6
    EXIT
