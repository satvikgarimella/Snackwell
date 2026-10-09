"""Exact integer arithmetic oracle for binary32 nearest-even test vectors.

Finite values are represented as integer * 2**exponent, then rounded once.
No host floating-point arithmetic or HDL implementation is used by the oracle.
"""
import random

FMUL, FADD, FSUB = 0x0C, 0x0D, 0x0E


def finite(bits):
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    if exponent:
        mantissa |= 1 << 23
    return (-mantissa if bits >> 31 else mantissa), (exponent - 150 if exponent else -149)


def pack(number, power, zero_sign=0):
    if number == 0:
        return zero_sign << 31
    sign = (number < 0) << 31
    number = abs(number)
    quantum = max(number.bit_length() - 1 + power - 23, -149)
    shift = quantum - power
    if shift > 0:
        significand, remainder = divmod(number, 1 << shift)
        midpoint = 1 << (shift - 1)
        significand += remainder > midpoint or (remainder == midpoint and significand & 1)
    else:
        significand = number << -shift
    if significand >= 1 << 24:
        significand >>= 1
        quantum += 1
    if significand < 1 << 23:
        return sign | significand
    exponent = quantum + 150
    if exponent >= 255:
        return sign | 0x7F800000
    return sign | exponent << 23 | (significand & 0x7FFFFF)


def reference(opcode, a, b):
    if opcode == FSUB:
        b ^= 1 << 31
    a_mag, b_mag = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    if a_mag > 0x7F800000 or b_mag > 0x7F800000:
        return 0x7FC00000
    sign = (a ^ b) >> 31
    if opcode == FMUL:
        if (a_mag == 0x7F800000 and b_mag == 0) or (b_mag == 0x7F800000 and a_mag == 0):
            return 0x7FC00000
        if a_mag == 0x7F800000 or b_mag == 0x7F800000:
            return sign << 31 | 0x7F800000
        na, ea = finite(a)
        nb, eb = finite(b)
        return pack(na * nb, ea + eb, sign)
    if a_mag == 0x7F800000 and b_mag == 0x7F800000 and sign:
        return 0x7FC00000
    if a_mag == 0x7F800000:
        return a
    if b_mag == 0x7F800000:
        return b
    na, ea = finite(a)
    nb, eb = finite(b)
    common = min(ea, eb)
    return pack((na << (ea - common)) + (nb << (eb - common)), common, (a & b) >> 31)


def write_vectors(path):
    rng = random.Random(0x534E4143)
    edges = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
             0x00800000, 0x80800000, 0x00800001, 0x3F800000, 0xBF800000,
             0x3F800001, 0x3F800002, 0x3FFFFFFF, 0x40000000, 0x40200000,
             0x33800000, 0xB3800000, 0x33000000, 0x7F7FFFFF, 0xFF7FFFFF,
             0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001]
    lines = []
    for op in (FMUL, FADD, FSUB):
        pairs = [(a, b) for a in edges for b in edges]
        pairs += [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(4096)]
        # Close magnitudes stress cancellation and normalization.
        for _ in range(512):
            a = rng.randrange(0x00800000, 0x7F7FFF00)
            b = a + rng.randrange(0, 256)
            pairs.append((a, b ^ (0x80000000 if op == FADD else 0)))
        while len(pairs) % 4:
            pairs.append((0, 0))
        for offset in range(0, len(pairs), 4):
            words = [op, 15]
            for a, b in pairs[offset:offset + 4]:
                words += [a, b, reference(op, a, b)]
            lines.append(" ".join(f"{v:08x}" for v in words))
        for mask in range(16):
            words = [op, mask]
            for lane in range(4):
                a, b = edges[lane + 7], edges[lane + 15]
                words += [a, b, reference(op, a, b)]
            lines.append(" ".join(f"{v:08x}" for v in words))
    path.write_text("\n".join(lines) + "\n")
    return len(lines)
