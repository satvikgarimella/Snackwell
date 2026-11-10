"""External-memory model launches; NVFP4 uses two protected shared buffers."""

from snackwell.kernels.inference import Assembly
from snackwell.numeric import SENTINEL

GLOBAL_WORDS = 16384
COPY_DESCRIPTORS = 6144
PACKED_OPERANDS = 16384


def prepare(layout, source, operation, params, *, overlap=True):
    """Return the executed program, external image and final shared guard image."""
    memory, shared = [SENTINEL] * GLOBAL_WORDS, [SENTINEL] * 1024
    if operation == "nvfp4_matmul":
        source = native_program(layout, params, memory, shared, overlap=overlap)
    memory[:1024] = layout.memory
    # All ordinary operands/constants/results live in the external kernel window.
    source = source.replace("LD_SHARED", "LD_GLOBAL").replace("ST_SHARED", "ST_GLOBAL")
    return source, memory, shared


def native_program(layout, params, memory, wanted_shared, *, overlap):
    columns, inner = params["n"], params["k"]
    groups = inner // 16
    schedule = [(c, g) for c in range(0, columns, 4) for g in range(groups)]
    a = Assembly(layout, "NVFP4 global operands -> async shared staging -> tensor -> global result")
    a.emit("XOR R15, R15, R15\nTID R12\nADD R12, R12, R12\nADD R12, R12, R12\nTALLOC T0")

    for stage, (column, group) in enumerate(schedule):
        left = layout.regions["a"][0] + group * 12
        right = layout.regions["b"][0] + (column // 4 * groups + group) * 12
        packed = layout.memory[left : left + 12] + layout.memory[right : right + 12]
        source = PACKED_OPERANDS + stage * 96
        destination = stage % 2 * 256
        memory[source // 4 : source // 4 + 24] = packed
        descriptor = COPY_DESCRIPTORS // 4 + stage * 4
        memory[descriptor : descriptor + 4] = [source, 12, 0x0308, (12 << 12) | destination]
        wanted_shared[destination // 4 : destination // 4 + 24] = packed

    def submit(stage):
        slot = stage % 2
        a.emit(
            f"LD_GLOBAL R6, [R12 + {COPY_DESCRIPTORS + stage * 16}]\nCPASYNC R{8 + slot}, S{slot}, R6, 1"
        )

    submit(0)
    for stage, (column, group) in enumerate(schedule):
        slot = stage % 2
        a.emit(f"CPWAIT S{slot}, R{8 + slot}")
        if overlap and stage + 1 < len(schedule):
            submit(stage + 1)
        if group == 0 and params["accumulate"]:
            for row in range(4):
                address = layout.address("initial") + (row * columns + column) * 4
                a.emit(f"LD_GLOBAL R4, [R12 + {address}]\nTST T0, {row}, R4")
        a.const(1, word=(12 << 12) | (slot * 256))
        a.const(2, word=(12 << 12) | (slot * 256 + 48))
        opcode = "TMMA_NVFP4_ACC" if params["accumulate"] or group else "TMMA_NVFP4"
        a.emit(f"{opcode} T0, R1, R2\nTCOMMIT R3, T0, 1\nCPRELEASE S{slot}\nTWAIT T0, R3")
        if group + 1 == groups:
            if params["final"]:
                a.emit(f"LD_GLOBAL R2, [R15 + {layout.address('scale_b')}]")
                if params["bias"]:
                    a.emit(f"LD_GLOBAL R5, [R12 + {layout.address('bias') + column * 4}]")
            for row in range(4):
                a.emit(f"TLD R4, T0, {row}")
                if params["final"]:
                    a.emit(
                        f"LD_GLOBAL R1, [R15 + {layout.address('scale_a') + row * 4}]\nFMUL R10, R1, R2\nFMUL R4, R4, R10"
                    )
                    if params["bias"]:
                        a.emit("FADD R4, R4, R5")
                address = layout.address("out") + (row * columns + column) * 4
                a.emit(f"ST_GLOBAL [R12 + {address}], R4")
        a.emit("TRELEASE T0")
        if not overlap and stage + 1 < len(schedule):
            submit(stage + 1)
    a.emit("TFREE T0\nTSTATUS R14\nEXIT")
    return a.text()
