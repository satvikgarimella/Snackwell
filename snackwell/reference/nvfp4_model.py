"""Independent numerical model for mixed NVFP4/FP32 inference.

No assembly, simulator outputs, launch layouts or runtime methods are used by
forward(). It follows the documented FP32 polynomial/reduction order so small
nonlinear approximation differences cannot cross a quantization boundary.
The original double-precision FP32 model remains the quality baseline.
"""

import math
from functools import lru_cache

from snackwell.character_model import ReferenceState, block_prefix, feedforward_width
from snackwell.numeric import bits, value
from snackwell.reference.fp32 import FADD, FMUL, FSUB, reference
from snackwell.reference.native import native_library, nvfp4_product, tensor_product
from snackwell.reference.nvfp4 import QuantizedVectors, quantize


@lru_cache(maxsize=1)
def arithmetic():
    library = native_library()
    return library.snackwell_fp32 if library is not None else reference


def add(a, b):
    return arithmetic()(FADD, a, b)


def sub(a, b):
    return arithmetic()(FSUB, a, b)


def mul(a, b):
    return arithmetic()(FMUL, a, b)


def total(words):
    result = 0
    for word in words:
        result = add(result, word)
    return result


def reciprocal(x):
    y = (0x7EF311C3 - x) & 0xFFFFFFFF
    for _ in range(4):
        y = add(y, mul(y, sub(bits(1), mul(x, y))))
    return y


def exp_negative(x):
    if value(x) < -80:
        return 0
    scaled = mul(x, bits(1 / 32))
    result = bits(1 / 720)
    for coefficient in (1 / 120, 1 / 24, 1 / 6, 0.5, 1, 1):
        result = add(mul(result, scaled), bits(coefficient))
    for _ in range(5):
        result = mul(result, result)
    return result


def softmax(row):
    maximum = max(row, key=value)
    exps = [exp_negative(sub(x, maximum)) for x in row]
    inverse = reciprocal(total(exps))
    return [mul(x, inverse) for x in exps]


def gelu(x):
    magnitude = bits(min(abs(value(x)), 10))
    cube_term = mul(mul(magnitude, magnitude), bits(0.044715))
    exponent = mul(mul(add(cube_term, bits(1)), magnitude), bits(-2 * math.sqrt(2 / math.pi)))
    tail = exp_negative(exponent)
    positive = mul(x, reciprocal(add(bits(1), tail)))
    return mul(positive, tail) if value(x) < 0 else positive


def normalize(x, gamma, beta, epsilon):
    width, result = len(gamma), []
    for start in range(0, len(x), width):
        row = x[start : start + width]
        mean = mul(total(row), bits(1 / width))
        centered = [sub(word, mean) for word in row]
        variance = add(
            mul(total(mul(word, word) for word in centered), bits(1 / width)), bits(epsilon)
        )
        inverse = (0x5F3759DF - (variance >> 1)) & 0xFFFFFFFF
        half = mul(variance, bits(0.5))
        for _ in range(4):
            inverse = mul(inverse, sub(bits(1.5), mul(half, mul(inverse, inverse))))
        result.extend(
            add(mul(mul(word, inverse), g), b)
            for word, g, b in zip(centered, gamma, beta, strict=True)
        )
    return result


@lru_cache(maxsize=48)
def quantized_weights(weights, inner, columns):
    # Construct columns independently of the runtime packing and tiling code.
    vectors = [[value(weights[k * columns + c]) for k in range(inner)] for c in range(columns)]
    return quantize([v + [0] * (-inner % 16) for v in vectors])


def linear(x, weights, bias, inner, columns):
    right = quantized_weights(tuple(weights), inner, columns)
    result = []
    for start in range(0, len(x), inner):
        left = quantize([[value(w) for w in x[start : start + inner]] + [0] * (-inner % 16)])
        result.extend(add(word, bias[c]) for c, word in enumerate(nvfp4_product(left, right)))
    return result


def packed_product(arrays, params):
    """Per-launch oracle: unpack the submitted words, check exact products/epilogue."""
    packed_a, packed_b, scale_a, scale_b = arrays[:4]
    columns, inner = params["n"], params["k"]
    groups = inner // 16

    def unpack(words, count):
        codes, scales = [], []
        for vector in range(count):
            row, row_scales = [], []
            for group in range(groups):
                index = ((vector // 4 * groups + group) * 4 + vector % 4) * 3
                lo, hi, scale = words[index : index + 3]
                packed = lo | hi << 32
                row.extend((packed >> (4 * k)) & 15 for k in range(16))
                row_scales.append(scale)
            codes.append(row)
            scales.append(row_scales)
        return QuantizedVectors(codes, scales, bits(1))

    initial = arrays[4] if params["accumulate"] else None
    result = nvfp4_product(
        unpack(packed_a, 4), unpack(packed_b, columns), initial, apply_tensor_scale=False
    )
    if params["final"]:
        result = [
            mul(word, mul(scale_a[i // columns], scale_b[0])) for i, word in enumerate(result)
        ]
        if params["bias"]:
            result = [add(word, arrays[-1][i % columns]) for i, word in enumerate(result)]
    return result


def forward(model, ids, *, previous=None, return_state=False):
    config = model["config"]
    width, context, vocabulary = config["width"], len(ids), len(config["vocabulary"])
    if not 1 <= context <= config["context"] or any(
        type(i) is not int or not 0 <= i < vocabulary for i in ids
    ):
        raise ValueError("invalid token input")
    weights = {name: [bits(w) for w in data] for name, data in model["weights"].items()}
    layers, heads = config.get("layers", 1), config.get("heads", 1)
    start = len(previous.ids) if previous else 0
    if previous and (
        tuple(ids[:-1]) != previous.ids or context != start + 1 or len(previous.layers) != layers
    ):
        raise ValueError("reference history does not match this prefix")
    rows, head_width = context - start, width // heads
    attention_columns = max(4, 1 << (context - 1).bit_length())
    pending = []

    def norm(x, suffix, prefix=""):
        return normalize(
            x,
            weights[prefix + "gamma" + suffix],
            weights[prefix + "beta" + suffix],
            config["epsilon"],
        )

    def project(x, name, inner, columns):
        return linear(x, weights[name[0]], weights[name[1]], inner, columns)

    x = [
        add(
            weights["token_embedding"][token * width + c],
            weights["position_embedding"][(start + r) * width + c],
        )
        for r, token in enumerate(ids[start:])
        for c in range(width)
    ]
    observed = {"embedding": x}
    for layer in range(layers):
        prefix = block_prefix(layer, layers)
        z = norm(x, "1", prefix)
        q, keys, values = [
            project(z, (prefix + "w" + name, prefix + "b" + name), width, width)
            for name in ("q", "k", "v")
        ]
        if previous:
            old_k, old_v = previous.layers[layer]
            keys, values = list(old_k) + keys, list(old_v) + values
        pending.append((tuple(keys), tuple(values)))
        padded_keys = keys + [0] * ((attention_columns - context) * width)
        padded_values = values + [0] * ((attention_columns - context) * width)
        attended = [0] * (rows * width)
        for head in range(heads):
            offset = head * head_width

            def features(data, count):
                return [
                    data[r * width + offset + j] for r in range(count) for j in range(head_width)
                ]

            scores = tensor_product(
                features(q, rows),
                features(padded_keys, attention_columns),
                rows,
                attention_columns,
                head_width,
                True,
            )
            probabilities = []
            for r in range(rows):
                probabilities.extend(
                    softmax(
                        [
                            mul(scores[r * attention_columns + c], bits(1 / math.sqrt(head_width)))
                            if c <= start + r
                            else 0xFF800000
                            for c in range(attention_columns)
                        ]
                    )
                )
            product = tensor_product(
                probabilities,
                features(padded_values, attention_columns),
                rows,
                head_width,
                attention_columns,
            )
            for r in range(rows):
                attended[r * width + offset : r * width + offset + head_width] = product[
                    r * head_width : (r + 1) * head_width
                ]
        projection = project(attended, (prefix + "wo", prefix + "bo"), width, width)
        residual = [add(a, b) for a, b in zip(x, projection, strict=True)]
        hidden_width = feedforward_width(config)
        hidden = [
            gelu(word)
            for word in project(
                norm(residual, "2", prefix), (prefix + "w1", prefix + "b1"), width, hidden_width
            )
        ]
        projection = project(hidden, (prefix + "w2", prefix + "b2"), hidden_width, width)
        x = [add(a, b) for a, b in zip(residual, projection, strict=True)]
        observed[prefix + "block"] = x
    normalized = norm(x, "_final")
    logits = project(normalized, ("lm_weight", "lm_bias"), width, vocabulary)
    observed.update(
        block=x,
        final_norm=normalized,
        logits=logits,
        probabilities=[
            word
            for r in range(rows)
            for word in softmax(logits[r * vocabulary : (r + 1) * vocabulary])
        ],
    )
    # Expose exact binary32 numbers through the same interface as the FP32 oracle.
    observed = {name: [value(word) for word in data] for name, data in observed.items()}
    observed["predictions"] = [
        max(range(vocabulary), key=lambda c: value(logits[r * vocabulary + c])) for r in range(rows)
    ]
    if previous:
        observed = {name: list(previous.observed[name]) + data for name, data in observed.items()}
    state = ReferenceState(
        tuple(ids), tuple(pending), {name: tuple(data) for name, data in observed.items()}
    )
    return (observed, state) if return_state else observed
