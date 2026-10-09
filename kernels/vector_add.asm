; Add two arrays of 16 FP32 values: C[i] = A[i] + B[i].
; Launch four warps, four active lanes each, starting at byte PC 0.
; Shared memory: A at 0, B at 64, C at 128 (byte addresses).
; All addresses are calculated by the kernel; no register preload required.

    TID       R1             ; i = warp_id * 4 + lane_id
    ADD       R1, R1, R1     ; 2 * i
    ADD       R1, R1, R1     ; byte offset = 4 * i
    ADDI      R2, R1, 64     ; address of B[i]
    ADDI      R3, R1, 128    ; address of C[i]
    LD_SHARED R4, [R1]       ; A[i]
    LD_SHARED R5, [R2]       ; B[i]
    FADD      R6, R4, R5
    ST_SHARED [R3], R6
    EXIT
