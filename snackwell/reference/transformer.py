"""Independent mathematical transformer block, used only for verification.

This starts from the original input and parameters, never intermediate SM
outputs. Python doubles and math functions provide a separate numerical path.
"""

import math

from snackwell.numeric import value
from snackwell.reference.inference import gelu_tanh, stable_softmax


def transformer_reference(case):
    n, d, h = case.rows, case.cols, case.cols * 2
    original = case.layout.memory

    def tensor(name):
        start, count = case.layout.regions[name]
        return [value(w) for w in original[start : start + count]]

    epsilon = next(s.params["epsilon"] for s in case.stages if s.name == "layernorm1")
    causal = next(s.params["causal"] for s in case.stages if s.name == "attention_scale_mask")

    def norm(x, number):
        gamma, beta = tensor(f"gamma{number}"), tensor(f"beta{number}")
        result = []
        for r in range(n):
            row = x[r * d : (r + 1) * d]
            mean = math.fsum(row) / d
            variance = math.fsum((a - mean) ** 2 for a in row) / d
            inv = 1 / math.sqrt(variance + epsilon)
            result.extend((a - mean) * inv * gamma[c] + beta[c] for c, a in enumerate(row))
        return result

    def linear(x, weight, bias, inner, columns):
        w, b = tensor(weight), tensor(bias)
        return [
            math.fsum(x[r * inner + k] * w[k * columns + c] for k in range(inner)) + b[c]
            for r in range(n)
            for c in range(columns)
        ]

    x = tensor("x")
    normalized = norm(x, 1)
    q, k, v = [linear(normalized, "w" + name, "b" + name, d, d) for name in ("q", "k", "v")]
    attended = []
    for r in range(n):
        scores = [
            math.fsum(q[r * d + j] * k[c * d + j] for j in range(d)) / math.sqrt(d)
            if not causal or c <= r
            else -math.inf
            for c in range(n)
        ]
        probabilities = stable_softmax(scores)
        attended.extend(
            math.fsum(probabilities[c] * v[c * d + j] for c in range(n)) for j in range(d)
        )
    projected = linear(attended, "wo", "bo", d, d)
    residual = [a + b for a, b in zip(x, projected)]
    normalized = norm(residual, 2)
    hidden = [gelu_tanh(a) for a in linear(normalized, "w1", "b1", d, h)]
    projected = linear(hidden, "w2", "b2", h, d)
    return [a + b for a, b in zip(residual, projected)]
