"""Lower logical model loads/stores to physical external pointers held in shared RAM.

The address lookup is an executed LD_SHARED instruction, not simulator address
translation. Stores publish a new output pointer so in-place, multi-pass kernels
can read their own results without modifying any aliased input allocation.
"""

import re
from collections import Counter, deque
from functools import lru_cache

from snackwell.assembler import memory_operand, register
from snackwell.kernels.inference import Assembly

OUTPUT_DELTA = 4092


def instructions(source):
    labels, code = {}, []
    for raw in source.splitlines():
        line = re.split(r";|//|#", raw, maxsplit=1)[0].strip()
        while match := re.match(r"^([A-Za-z_][A-Za-z_0-9]*):", line):
            if match[1] in labels:
                raise ValueError("duplicate direct-kernel label")
            labels[match[1]] = len(code)
            line = line[match.end() :].strip()
        if line:
            op, *rest = line.split(None, 1)
            operands = [x.strip() for x in rest[0].split(",")] if rest else []
            code.append((op.upper(), operands))
    return labels, code


def registers(op, args):
    """Scalar-register reads/writes, including the four-register TMUL groups."""
    if op in ("NOP", "EXIT", "SSY", "TALLOC", "TFREE", "TRELEASE", "CPRELEASE"):
        return set(), set()
    if op == "BEQ":
        return {register(x) for x in args[:2]}, set()
    if op in ("LD_SHARED", "LD_GLOBAL"):
        return {memory_operand(args[1])[0]}, {register(args[0])}
    if op in ("ST_SHARED", "ST_GLOBAL"):
        return {memory_operand(args[0])[0], register(args[1])}, set()
    if op in ("TID", "TLD", "TSTATUS", "GSTATUS", "CPSTATUS", "TCOMMIT"):
        return set(), {register(args[0])}
    if op in ("TWAIT", "CPWAIT"):
        return {register(args[1])}, set()
    if op == "TST":
        return {register(args[2])}, set()
    if op.startswith("TMMA"):
        return {register(x) for x in args[1:]}, set()
    if op == "CPASYNC":
        return {register(args[2])}, {register(args[0])}
    if op == "TMUL":
        rd, a, b = map(register, args)
        return set(range(a, a + 4)) | set(range(b, b + 4)), set(range(rd, rd + 4))
    if op == "ADDI":
        return {register(args[1])}, {register(args[0])}
    if op in ("ADD", "SUB", "AND", "OR", "XOR", "FMUL", "FADD", "FSUB", "SHR", "SLTU", "FLT"):
        return {register(x) for x in args[1:]}, {register(args[0])}
    raise ValueError(f"unsupported direct-kernel instruction {op}")


@lru_cache(maxsize=128)
def analysis(source):
    labels, code = instructions(source)
    uses_defs = [registers(op, args) for op, args in code]
    successors = []
    for i, (op, args) in enumerate(code):
        edges = [] if op == "EXIT" else [i + 1]
        if op == "BEQ":
            if args[2] not in labels:
                raise ValueError("direct kernels require symbolic branch targets")
            edges.append(labels[args[2]])
        if op == "SSY" and args[0] not in labels:
            raise ValueError("direct kernels require symbolic reconvergence targets")
        successors.append(edges)
    live_in, live_out = [set() for _ in code], [set() for _ in code]
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            after = set().union(*(live_in[j] for j in successors[i] if j < len(code)))
            reads, writes = uses_defs[i]
            before = reads | (after - writes)
            if before != live_in[i] or after != live_out[i]:
                live_in[i], live_out[i] = before, after
                changed = True
    # Only values proven equal on every incoming path are known. Registers
    # start unknown, regardless of simulator reset values. Loops widen to
    # unknown when successive iterations disagree; unreachable edges are ignored.
    known = [None for _ in code]
    if code:
        known[0] = (None,) * 16
    pending = deque([0] if code else [])
    while pending:
        i = pending.popleft()
        op, args = code[i]
        before = known[i]
        after = list(before)
        for rd in uses_defs[i][1]:
            after[rd] = None
        if op in ("XOR", "SUB") and args[1] == args[2]:
            after[register(args[0])] = 0
        elif op == "ADDI" and before[register(args[1])] is not None:
            after[register(args[0])] = (before[register(args[1])] + int(args[2], 0)) & 0xFFFFFFFF
        for j in successors[i]:
            if j >= len(code):
                continue
            merged = (
                tuple(after)
                if known[j] is None
                else tuple(a if a == b else None for a, b in zip(known[j], after, strict=True))
            )
            if merged != known[j]:
                known[j] = merged
                pending.append(j)
    return labels, code, uses_defs, live_out, known


@lru_cache(maxsize=128)
def literal_loads(source, constants):
    """Count loads whose complete logical address is a compiler-owned literal."""
    _, code, _, _, known = analysis(source)
    uses = Counter()
    for i, (op, args) in enumerate(code):
        if op != "LD_GLOBAL" or known[i] is None:
            continue
        base, offset = memory_operand(args[1])
        value = known[i][base]
        if value is not None and value + offset in constants:
            uses[value + offset] += 1
    return uses


@lru_cache(maxsize=128)
def lower(source, table_base=0, *, literals=(), optimize=True):
    labels, code, uses_defs, live_out, known = analysis(source)
    literals = dict(literals) if optimize else {}
    # An otherwise unused register can retain the output displacement for the
    # whole launch, including loop backedges and divergent paths.
    used = set().union(*(reads | writes for reads, writes in uses_defs))
    unused = sorted(set(range(16)) - used)
    delta = unused[-1] if optimize and unused and any(op == "ST_GLOBAL" for op, _ in code) else None
    lines = ["; Direct external pointers; address lookups execute on the SM."]
    if delta is not None:
        lines.extend(
            (
                f"XOR R{delta}, R{delta}, R{delta}",
                f"LD_SHARED R{delta}, [R{delta} + {OUTPUT_DELTA}]",
            )
        )
    pointers = {}
    at = {}
    for label, i in labels.items():
        at.setdefault(i, []).append(label)
    for i, (op, args) in enumerate(code):
        if i in at:
            pointers.clear()
        lines.extend(label + ":" for label in at.get(i, []))
        if op == "LD_GLOBAL":
            base, offset = memory_operand(args[1])
            value = known[i][base] if known[i] is not None else None
            literal = literals.get(value + offset) if value is not None else None
            if literal is not None and -8192 <= literal - value <= 8191:
                lines.append(f"LD_SHARED {args[0]}, [R{base} {literal - value:+d}]")
            else:
                key = (base, offset)
                pointer = pointers.get(key)
                if pointer is None:
                    free = sorted(
                        set(range(16))
                        - live_out[i]
                        - uses_defs[i][0]
                        - uses_defs[i][1]
                        - set(pointers.values())
                        - {delta}
                    )
                    pointer = free[-1] if optimize and free else register(args[0])
                    lines.append(f"LD_SHARED R{pointer}, [R{base} {offset + table_base:+d}]")
                    if pointer != register(args[0]):
                        pointers[key] = pointer
                lines.append(f"LD_GLOBAL {args[0]}, [R{pointer}]")
        elif op == "ST_GLOBAL":
            pointers.clear()  # A published pointer may alias any cached address.
            base, offset = memory_operand(args[0])
            free = sorted(set(range(16)) - live_out[i] - uses_defs[i][0] - {delta})
            if delta is not None:
                if not free:
                    raise ValueError("direct store needs a dead scalar register")
                physical = free[0]
                if offset:
                    lines.append(f"ADDI R{physical}, R{base}, {offset}")
                lines.extend(
                    (
                        f"ADD R{physical}, R{physical if offset else base}, R{delta}",
                        f"ST_GLOBAL [R{physical}], {args[1]}",
                        f"ST_SHARED [R{base} {offset + table_base:+d}], R{physical}",
                    )
                )
            else:
                if len(free) < 2:
                    raise ValueError("direct store needs two dead scalar registers")
                logical, physical = free[:2]
                lines.extend(
                    (
                        f"ADDI R{logical}, R{base}, {offset}",
                        f"XOR R{physical}, R{physical}, R{physical}",
                        f"LD_SHARED R{physical}, [R{physical} + {OUTPUT_DELTA}]",
                        f"ADD R{physical}, R{logical}, R{physical}",
                        f"ST_GLOBAL [R{physical}], {args[1]}",
                        f"ST_SHARED [R{logical} + {table_base}], R{physical}",
                    )
                )
        else:
            lines.append(op + (" " + ", ".join(args) if args else ""))
        # Cached pointers never cross control-flow joins, base-register writes,
        # scratch reuse or operations that can modify their shared table.
        writes = uses_defs[i][1]
        pointers = {
            key: reg for key, reg in pointers.items() if key[0] not in writes and reg not in writes
        }
        if op in ("BEQ", "ST_SHARED", "CPASYNC"):
            pointers.clear()
    lines.extend(label + ":" for label in at.get(len(code), []))
    return "\n".join(lines) + "\n"


def native_program(layout, params, address):
    """Copy A and B from separate physical allocations, then run native NVFP4.

    The two copy slots protect A and B independently. Once the tensor unit has
    gathered both, release them and prefetch the next pair during arithmetic.
    No combined A/B buffer is gathered in external RAM.
    """
    columns, inner = params["n"], params["k"]
    groups = inner // 16
    schedule = [(c, g) for c in range(0, columns, 4) for g in range(groups)]
    left_descriptor = layout.constant(12 << 12)
    right_descriptor = layout.constant((12 << 12) | 48)
    table_base = 128
    descriptors = (layout.cursor + table_base // 4 + 3) & ~3
    if descriptors + len(schedule) * 8 > OUTPUT_DELTA // 4:
        raise ValueError("native direct descriptors exceed shared RAM")
    descriptor_words = []
    for column, group in schedule:
        for name, index, destination in (
            ("a", group * 12, 0),
            ("b", (column // 4 * groups + group) * 12, 48),
        ):
            logical = layout.regions[name][0] + index
            pointers = [address(logical + j) for j in range(12)]
            if pointers != list(range(pointers[0], pointers[0] + 12)):
                raise ValueError("each packed NVFP4 group must be physically contiguous")
            descriptor_words.extend((pointers[0] * 4, 12, 0x0304, (12 << 12) | destination))
    a = Assembly(layout, "resident NVFP4: separate A/B copies, persistent result")
    a.emit("XOR R15, R15, R15\nTID R12\nADD R12, R12, R12\nADD R12, R12, R12\nTALLOC T0")

    def submit(stage):
        for slot in range(2):
            descriptor = descriptors * 4 + stage * 32 + slot * 16
            a.emit(
                f"LD_SHARED R6, [R12 + {descriptor}]\nCPASYNC R{8 + slot}, S{slot}, R6, 1\nCPWAIT S{slot}, R{8 + slot}"
            )

    submit(0)
    for stage, (column, group) in enumerate(schedule):
        if group == 0 and params["accumulate"]:
            for row in range(4):
                offset = layout.address("initial") + (row * columns + column) * 4
                a.emit(f"LD_GLOBAL R4, [R12 + {offset}]\nTST T0, {row}, R4")
        a.emit(f"LD_GLOBAL R1, [R15 + {left_descriptor}]\nLD_GLOBAL R2, [R15 + {right_descriptor}]")
        opcode = "TMMA_NVFP4_ACC" if params["accumulate"] or group else "TMMA_NVFP4"
        a.emit(f"{opcode} T0, R1, R2\nTCOMMIT R3, T0, 1\nCPRELEASE S0\nCPRELEASE S1")
        if stage + 1 < len(schedule):
            submit(stage + 1)
        a.emit("TWAIT T0, R3")
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
                a.emit(
                    f"ST_GLOBAL [R12 + {layout.address('out') + (row * columns + column) * 4}], R4"
                )
        a.emit("TRELEASE T0")
    a.emit("TFREE T0\nTSTATUS R14\nEXIT")
    return a.text(), table_base, descriptors, descriptor_words
