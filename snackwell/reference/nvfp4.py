"""NVFP4 encoding, finite host quantization, and independent FP32 matrix oracle.

The oracle decodes values arithmetically, then uses the integer binary32
reference. It does not reproduce the RTL's fixed-point multiplier circuit.
"""

import math
import random
from dataclasses import dataclass

from snackwell.numeric import bits, value
from snackwell.reference.fp32 import FADD, FMUL, reference


def e2m1(code):
    if type(code) is not int or not 0 <= code < 16:
        raise ValueError("E2M1 code must be a nibble")
    exponent, mantissa = (code >> 1) & 3, code & 1
    magnitude = mantissa / 2 if exponent == 0 else (1 + mantissa / 2) * 2 ** (exponent - 1)
    return math.copysign(magnitude, -1 if code & 8 else 1)


def ue4m3(code):
    if type(code) is not int or not 0 <= code < 127:
        raise ValueError("UE4M3 scale must be finite, unsigned, with zero padding")
    exponent, mantissa = code >> 3, code & 7
    return mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)


def scaled(code, scale):
    return e2m1(code) * ue4m3(scale)


def product(a, scale_a, b, scale_b):
    if not 0 <= scale_a < 127 or not 0 <= scale_b < 127:
        return 0x7FC00000
    return bits(scaled(a, scale_a) * scaled(b, scale_b))


def encode_e2m1(number):
    if not math.isfinite(number):
        raise ValueError("quantization requires finite input")
    magnitude = min(abs(number), 6)
    code = min(range(8), key=lambda c: (abs(magnitude - e2m1(c)), c & 1))
    return code | (8 if math.copysign(1, number) < 0 else 0)


def encode_ue4m3(number):
    if not math.isfinite(number) or number < 0:
        raise ValueError("scale must be finite and nonnegative")
    number = min(number, 448)
    return min(range(127), key=lambda c: (abs(number - ue4m3(c)), c & 1))


@dataclass
class QuantizedVectors:
    """Rows of A or columns of B, each divided into consecutive groups of 16."""

    codes: list[list[int]]
    scales: list[list[int]]
    tensor_scale: int  # binary32 dequantization factor

    def group_words(self, first_vector, group):
        vectors = self.codes[first_vector : first_vector + 4]
        if len(vectors) != 4 or group < 0 or any(16 * (group + 1) > len(v) for v in vectors):
            raise ValueError("a tile needs four complete 16-element vectors")
        words = []
        for index, vector in enumerate(vectors, first_vector):
            codes = vector[group * 16 : (group + 1) * 16]
            packed = sum(code << (4 * k) for k, code in enumerate(codes))
            words.extend((packed & 0xFFFFFFFF, packed >> 32, self.scales[index][group]))
        return words


def quantize(vectors, *, tensor_scale=None):
    vectors = [[value(bits(x)) for x in vector] for vector in vectors]
    if not vectors or not vectors[0] or len(vectors[0]) % 16:
        raise ValueError("vectors must contain complete groups of 16")
    if any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("vectors must have the same length")
    if not all(math.isfinite(x) for v in vectors for x in v):
        raise ValueError("quantization requires finite input")
    maximum = max(abs(x) for v in vectors for x in v)
    if tensor_scale is None:
        # Clamp only to the smallest positive binary32 value; zero tensors use 1.
        tensor_scale = max(maximum / (448 * 6), value(1)) if maximum else 1.0
    tensor_scale = value(bits(tensor_scale))
    if not math.isfinite(tensor_scale) or tensor_scale <= 0:
        raise ValueError("tensor scale must be a finite positive binary32 value")
    codes, scales = [], []
    for vector in vectors:
        row, row_scales = [], []
        for start in range(0, len(vector), 16):
            block = vector[start : start + 16]
            scale = encode_ue4m3(max(abs(x) for x in block) / (6 * tensor_scale))
            denominator = ue4m3(scale) * tensor_scale
            row.extend(
                encode_e2m1(x / denominator if denominator else math.copysign(0, x)) for x in block
            )
            row_scales.append(scale)
        codes.append(row)
        scales.append(row_scales)
    return QuantizedVectors(codes, scales, bits(tensor_scale))


def matmul(a, b, initial=None, *, apply_tensor_scale=True):
    """A rows times B columns, increasing k, with one binary32 add per product.

    Initial accumulators use the block-scaled units, before the tensor epilogue.
    """
    if not a.codes or not b.codes or any(len(v) != len(a.codes[0]) for v in a.codes + b.codes):
        raise ValueError("matrix inner dimensions must agree")
    rows, columns, result = len(a.codes), len(b.codes), []
    if initial is not None and len(initial) != rows * columns:
        raise ValueError("incorrect initial accumulator shape")
    alpha = reference(FMUL, a.tensor_scale, b.tensor_scale)
    for row in range(rows):
        for column in range(columns):
            accumulator = 0 if initial is None else initial[row * columns + column]
            for k, (aa, bb) in enumerate(zip(a.codes[row], b.codes[column], strict=True)):
                term = product(aa, a.scales[row][k // 16], bb, b.scales[column][k // 16])
                accumulator = reference(FADD, accumulator, term)
            result.append(
                reference(FMUL, accumulator, alpha) if apply_tensor_scale else accumulator
            )
    return result


def write_vectors(path):
    # Every element and scale byte, including all rejected scale encodings.
    cases = [(a, s, 2, 0x38) for a in range(16) for s in range(256)]
    cases += [(2, 0x38, b, s) for b in range(16) for s in range(256)]
    boundaries = (0, 1, 7, 8, 0x37, 0x38, 0x39, 0x77, 0x78, 0x7E)
    cases += [
        (a, sa, b, sb)
        for a in range(16)
        for b in range(16)
        for sa in boundaries
        for sb in boundaries
    ]
    rng = random.Random(0x4E564634)
    cases += [
        (rng.randrange(16), rng.randrange(127), rng.randrange(16), rng.randrange(127))
        for _ in range(4096)
    ]
    path.write_text(
        "".join(
            f"{a:x} {sa:02x} {b:x} {sb:02x} {product(a, sa, b, sb):08x}\n" for a, sa, b, sb in cases
        )
    )
    return len(cases)
