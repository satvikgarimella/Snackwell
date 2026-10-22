"""A pre-LayerNorm, single-head transformer block built from SM kernels.

The host allocates memory and emits programs. Every projection, normalization,
attention operation, activation and residual addition executes on the SM.
"""

import math
import random

from snackwell.kernels.inference import (
    Case,
    Layout,
    Stage,
    elementwise,
    layernorm,
    scale_mask,
    softmax,
    tiled_matmul,
)
from snackwell.numeric import bits, value


def build_transformer(rows=4, width=4, inputs=None):
    if rows not in (4, 8, 16) or width not in (4, 8, 16):
        raise ValueError("supported dimensions are 4, 8, and 16 where memory permits")
    inputs = {} if inputs is None else inputs
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a JSON object")
    hidden = 2 * width
    counts = {"x": rows * width}
    for name in ("q", "k", "v", "o"):
        counts["w" + name], counts["b" + name] = width * width, width
    counts.update(
        w1=width * hidden,
        b1=hidden,
        w2=hidden * width,
        b2=width,
        gamma1=width,
        beta1=width,
        gamma2=width,
        beta2=width,
    )
    allowed = set(counts) | {"causal", "epsilon"}
    if set(inputs) - allowed:
        raise ValueError(f"unexpected transformer input keys: {sorted(set(inputs) - allowed)}")
    causal = inputs.get("causal", True)
    if type(causal) is not bool:
        raise ValueError("causal must be true or false")
    epsilon = value(bits(inputs.get("epsilon", 1e-5)))
    if not math.isfinite(epsilon) or epsilon < value(0x00800000):
        raise ValueError("epsilon must be positive normal FP32")

    layout, stages, rng = Layout(), [], random.Random(0x534E4143)
    for name, count in counts.items():
        if name.startswith("gamma"):
            default = [1.0] * count
        elif name.startswith("b"):
            default = [0.0] * count
        else:
            scale = 1.0 if name == "x" else 0.5 / math.sqrt(hidden if name == "w2" else width)
            default = [rng.uniform(-scale, scale) for _ in range(count)]
        data = inputs.get(name, default)
        if not isinstance(data, list) or len(data) != count:
            raise ValueError(f"{name} requires exactly {count} values")
        words = [bits(x) for x in data]
        if not all(math.isfinite(value(word)) for word in words):
            raise ValueError(f"{name} requires finite FP32 inputs")
        layout.allocate(name, count, words)

    # These scratch regions are reused only after their previous values die.
    # Per-stage memory guards still cover every other word, including weights.
    for name in ("norm", "q", "k", "v", "attended", "projected", "residual", "out"):
        layout.allocate(name, rows * width)
    layout.allocate("scores", rows * rows)
    layout.allocate("hidden", rows * hidden)

    def add(name, source, operation, ins, out, **params):
        stages.append(Stage(name, source, operation, tuple(ins), out, params))

    def norm(name, x, number):
        gamma, beta = f"gamma{number}", f"beta{number}"
        add(
            name,
            layernorm(layout, rows, width, epsilon, x, "norm", gamma, beta),
            "layernorm",
            (x, gamma, beta),
            "norm",
            rows=rows,
            cols=width,
            epsilon=epsilon,
        )

    def linear(name, x, weight, bias, out, inner, columns):
        add(
            name + "_matmul",
            tiled_matmul(layout, rows, columns, inner, x, weight, out),
            "matmul",
            (x, weight),
            out,
            m=rows,
            n=columns,
            k=inner,
        )
        add(
            name + "_bias",
            elementwise(layout, "bias_add", rows, columns, out, out, bias),
            "bias_add",
            (out, bias),
            out,
            rows=rows,
            cols=columns,
        )

    norm("layernorm1", "x", 1)
    for name in ("q", "k", "v"):
        linear(name, "norm", "w" + name, "b" + name, name, width, width)
    add(
        "attention_qk",
        tiled_matmul(layout, rows, rows, width, "q", "k", "scores", True),
        "matmul",
        ("q", "k"),
        "scores",
        m=rows,
        n=rows,
        k=width,
        transpose_b=True,
    )
    add(
        "attention_scale_mask",
        scale_mask(layout, rows, width, causal),
        "scale_mask",
        ("scores",),
        "scores",
        rows=rows,
        cols=rows,
        width=width,
        causal=causal,
    )
    add(
        "attention_softmax",
        softmax(layout, rows, rows, "scores", "scores"),
        "softmax",
        ("scores",),
        "scores",
        rows=rows,
        cols=rows,
    )
    add(
        "attention_pv",
        tiled_matmul(layout, rows, width, rows, "scores", "v", "attended"),
        "matmul",
        ("scores", "v"),
        "attended",
        m=rows,
        n=width,
        k=rows,
    )
    linear("attention_output", "attended", "wo", "bo", "projected", width, width)
    add(
        "attention_residual",
        elementwise(layout, "residual_add", rows, width, "x", "residual", "projected"),
        "residual_add",
        ("x", "projected"),
        "residual",
        rows=rows,
        cols=width,
    )
    norm("layernorm2", "residual", 2)
    linear("ffn_up", "norm", "w1", "b1", "hidden", width, hidden)
    add(
        "ffn_gelu",
        elementwise(layout, "gelu", rows, hidden, "hidden", "hidden"),
        "gelu",
        ("hidden",),
        "hidden",
        rows=rows,
        cols=hidden,
    )
    linear("ffn_down", "hidden", "w2", "b2", "projected", hidden, width)
    add(
        "ffn_residual",
        elementwise(layout, "residual_add", rows, width, "residual", "out", "projected"),
        "residual_add",
        ("residual", "projected"),
        "out",
        rows=rows,
        cols=width,
    )
    return Case("transformer", layout, stages, rows, width)
