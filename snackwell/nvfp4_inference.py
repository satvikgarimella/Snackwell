"""Versioned host quantization policy for native NVFP4 linear projections.

Each activation row has its own FP32 factor, independent of prompt batching.
Each complete weight matrix has one factor, independent of launch tiling.
Group scales always cover sixteen consecutive K elements, zero-padded at end.
Attention QK/PV and all other operations retain the FP32 path.
"""

from functools import lru_cache

from snackwell.numeric import value
from snackwell.reference.nvfp4 import QuantizedVectors, quantize

POLICY = "nvfp4-linear-row-v1"
PRECISIONS = ("fp32", "nvfp4")


def precision_policy(precision):
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    return POLICY if precision == "nvfp4" else "fp32"


def activation_rows(words, rows, inner):
    padded = (inner + 15) // 16 * 16
    return [
        quantize([[value(x) for x in words[r * inner : (r + 1) * inner]] + [0] * (padded - inner)])
        for r in range(rows)
    ]


@lru_cache(maxsize=48)
def weight_columns(words, columns, inner):
    padded = (inner + 15) // 16 * 16
    return quantize(
        [
            [value(words[k * columns + c]) for k in range(inner)] + [0] * (padded - inner)
            for c in range(columns)
        ]
    )


def tile(vectors, first, count, start, inner):
    """Slice already-quantized vectors without choosing new scales."""
    return QuantizedVectors(
        [v[start : start + inner] for v in vectors.codes[first : first + count]],
        [v[start // 16 : (start + inner) // 16] for v in vectors.scales[first : first + count]],
        vectors.tensor_scale,
    )
