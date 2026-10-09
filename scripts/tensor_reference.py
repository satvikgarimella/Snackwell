"""Deterministic matrix vectors with sequential, non-fused binary32 MACs."""
import random
from fp32_reference import FADD, FMUL, pack, reference


def matmul(a, b):
    result = []
    for row in range(4):
        for col in range(4):
            accumulator = 0
            for k in range(4):
                product = reference(FMUL, a[4 * row + k], b[4 * k + col])
                accumulator = reference(FADD, accumulator, product)
            result.append(accumulator)
    return result


def write_tensor_vectors(path):
    rng = random.Random(0x54454E53)
    zero = [0] * 16
    identity = [pack(int(row == col), 0) for row in range(4) for col in range(4)]
    ascending = [pack(i + 1, 0) for i in range(16)]
    descending = [pack(16 - i, 0) for i in range(16)]
    cases = [(zero, zero), (zero, ascending), (ascending, zero),
             (identity, ascending), (ascending, identity), (ascending, ascending),
             (ascending, descending), (descending, ascending)]
    # Basis matrices detect swapped rows/columns and misaligned tokens.
    for index in range(16):
        basis = zero.copy()
        basis[index] = pack(1, 0)
        cases += [(basis, ascending), (ascending, basis)]
    cases.append(([pack((-1)**i * (i + 1), -2) for i in range(16)], descending))
    for _ in range(64):
        cases.append(([pack(rng.randrange(-128, 129), -3) for _ in range(16)],
                      [pack(rng.randrange(-128, 129), -3) for _ in range(16)]))
    for _ in range(32):
        cases.append(([pack(rng.randrange(-1024, 1025), rng.randrange(-140, 110)) for _ in range(16)],
                      [pack(rng.randrange(-1024, 1025), rng.randrange(-140, 110)) for _ in range(16)]))
    edges = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
             0x3f800001, 0xbf800001, 0x33800000, 0x7f7fffff, 0xff7fffff,
             0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001]
    for _ in range(32):
        cases.append(([rng.choice(edges) for _ in range(16)],
                      [rng.choice(edges) for _ in range(16)]))
    for _ in range(16):
        cases.append(([rng.getrandbits(32) for _ in range(16)],
                      [rng.getrandbits(32) for _ in range(16)]))
    lines = []
    for index, (a, b) in enumerate(cases):
        words = [index, *a, *b, *matmul(a, b)]
        lines.append(" ".join(f"{word:08x}" for word in words))
    path.write_text("\n".join(lines) + "\n")
    return len(cases)
