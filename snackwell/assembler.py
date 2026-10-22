#!/usr/bin/env python3
"""Assemble Snackwell instructions into a padded $readmemh program image."""

import re
from pathlib import Path

IMEM_WORDS = 1024
OPCODES = {
    "NOP": 0x00,
    "ADD": 0x01,
    "SUB": 0x02,
    "AND": 0x03,
    "OR": 0x04,
    "XOR": 0x05,
    "ADDI": 0x06,
    "TID": 0x07,
    "LD_SHARED": 0x08,
    "ST_SHARED": 0x09,
    "BEQ": 0x0A,
    "SSY": 0x0B,
    "FMUL": 0x0C,
    "FADD": 0x0D,
    "FSUB": 0x0E,
    "TMUL": 0x0F,
    "SHR": 0x10,
    "SLTU": 0x11,
    "FLT": 0x12,
    "EXIT": 0x3F,
}
EXIT_WORD = OPCODES["EXIT"] << 26
OPERAND_COUNTS = {"NOP": 0, "EXIT": 0, "TID": 1, "SSY": 1, "LD_SHARED": 2, "ST_SHARED": 2}


class AssemblyError(ValueError):
    pass


def number(text):
    if not re.fullmatch(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)", text):
        raise AssemblyError(f"expected an integer, got {text!r}")
    return int(text, 0 if text.lstrip("+-").lower().startswith(("0x", "0b")) else 10)


def register(text):
    match = re.fullmatch(r"[Rr]([0-9]+)", text)
    if not match or not 0 <= int(match[1]) < 16:
        raise AssemblyError(f"expected R0 through R15, got {text!r}")
    return int(match[1])


def immediate(value):
    if not -8192 <= value <= 8191:
        raise AssemblyError(f"signed 14-bit immediate out of range: {value}")
    return value & 0x3FFF


def memory_operand(text):
    match = re.fullmatch(r"\[\s*([Rr][0-9]+)\s*(?:([+-])\s*(\S+))?\s*\]", text)
    if not match:
        raise AssemblyError(f"expected [Rn], [Rn + offset], or [Rn - offset], got {text!r}")
    offset = number(match[3]) if match[3] else 0
    if offset < 0:
        raise AssemblyError("use only one sign in a memory offset")
    return register(match[1]), offset * (-1 if match[2] == "-" else 1)


def assemble(source, max_words=IMEM_WORDS):
    labels, instructions = {}, []
    for line_no, raw in enumerate(source.splitlines(), 1):
        line = re.split(r";|//|#", raw, maxsplit=1)[0].strip()
        while match := re.match(r"^([A-Za-z_][A-Za-z_0-9]*):", line):
            name = match[1]
            if name in labels:
                raise AssemblyError(f"line {line_no}: duplicate label {name!r}")
            labels[name] = len(instructions)
            line = line[match.end() :].strip()
        if line:
            instructions.append((line_no, line))
    if not instructions:
        raise AssemblyError("program is empty")
    if len(instructions) > max_words:
        raise AssemblyError(
            f"program has {len(instructions)} instructions; capacity is {max_words}"
        )

    words = []
    for pc, (line_no, line) in enumerate(instructions):
        try:
            fields = line.split(None, 1)
            op = fields[0].upper()
            if op not in OPCODES:
                raise AssemblyError(f"unknown instruction {op!r}")
            operands = [item.strip() for item in fields[1].split(",")] if len(fields) > 1 else []
            count = OPERAND_COUNTS.get(op, 3)
            if len(operands) != count or any(not operand for operand in operands):
                raise AssemblyError(f"{op} expects {count} operands")
            rd = a = b = imm = 0
            if op == "TID":
                rd = register(operands[0])
            elif op == "ADDI":
                rd, a = map(register, operands[:2])
                imm = immediate(number(operands[2]))
            elif op == "LD_SHARED":
                rd = register(operands[0])
                a, offset = memory_operand(operands[1])
                imm = immediate(offset)
            elif op == "ST_SHARED":
                a, offset = memory_operand(operands[0])
                b = register(operands[1])
                imm = immediate(offset)
            elif op in ("BEQ", "SSY"):
                if op == "BEQ":
                    a, b = map(register, operands[:2])
                target = operands[-1]
                if target in labels:
                    displacement = labels[target] - pc - 1
                elif re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", target):
                    raise AssemblyError(f"undefined label {target!r}")
                else:
                    displacement = number(target)
                imm = immediate(displacement)
            elif count == 3:
                rd, a, b = map(register, operands)
                if op == "TMUL" and max(rd, a, b) > 12:
                    raise AssemblyError("TMUL tile bases must be R0 through R12")
            words.append((OPCODES[op] << 26) | (rd << 22) | (a << 18) | (b << 14) | imm)
        except AssemblyError as error:
            raise AssemblyError(f"line {line_no}: {error}") from error
    return words


def write_image(path, words):
    if not 0 < len(words) <= IMEM_WORDS:
        raise AssemblyError(f"expected 1 through {IMEM_WORDS} instructions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{word:08x}\n" for word in words + [EXIT_WORD] * (IMEM_WORDS - len(words)))
    )
