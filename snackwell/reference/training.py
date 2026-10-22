"""Independent integer-FP32 stage oracles and double-precision backpropagation."""

import math

from snackwell.numeric import bits, value
from snackwell.reference.fp32 import FADD, FMUL, FSUB, reference
from snackwell.reference.inference import gelu_tanh


def gelu_derivative(x):
    if abs(x) >= 10:
        return 1.0 if x > 0 else 0.0
    k, a = math.sqrt(2 / math.pi), 0.044715
    t = math.tanh(k * (x + a * x**3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * a * x * x)


def expected_stage(kind, arrays, params):
    x, b = arrays[:2]
    if kind == "gelu_backward":
        return [gelu_derivative(value(a)) * value(g) for a, g in zip(x, b)], False
    if kind == "column_sum":
        rows, cols = params["rows"], params["cols"]
        out = list(b)
        for r in range(rows):
            for c in range(cols):
                out[c] = reference(FADD, out[c], x[r * cols + c])
        return out, True
    if kind == "add":
        return [reference(FADD, a, g) for a, g in zip(x, b)], True
    scale = bits(params["scale"])
    if kind == "scale":
        return [reference(FMUL, a, scale) for a in x], True
    if kind == "mse_gradient":
        return [reference(FMUL, reference(FSUB, a, t), scale) for a, t in zip(x, b)], True
    if kind == "sgd":
        return [reference(FSUB, a, reference(FMUL, g, scale)) for a, g in zip(x, b)], True
    if kind == "squared_loss":
        total = arrays[2][0]
        for a, t in zip(x, b):
            delta = reference(FSUB, a, t)
            total = reference(FADD, total, reference(FMUL, delta, delta))
        return [reference(FMUL, total, scale)], True
    raise ValueError(kind)


def forward(parameters, x, target, rows, inputs, hidden, outputs):
    """Use original inputs/parameters only; never consume RTL intermediates."""
    p = parameters
    z = [
        math.fsum(x[r * inputs + i] * p["w1"][i * hidden + h] for i in range(inputs)) + p["b1"][h]
        for r in range(rows)
        for h in range(hidden)
    ]
    a = [gelu_tanh(v) for v in z]
    y = [
        math.fsum(a[r * hidden + h] * p["w2"][h * outputs + o] for h in range(hidden)) + p["b2"][o]
        for r in range(rows)
        for o in range(outputs)
    ]
    loss = math.fsum((a - b) ** 2 for a, b in zip(y, target)) / (2 * rows * outputs)
    return dict(z=z, a=a, y=y, loss=[loss])


def backward(parameters, x, target, rows, inputs, hidden, outputs, learning_rate):
    state = forward(parameters, x, target, rows, inputs, hidden, outputs)
    p, a, z, y = parameters, state["a"], state["z"], state["y"]
    dy = [(a - b) / (rows * outputs) for a, b in zip(y, target)]
    dw2 = [
        math.fsum(a[r * hidden + h] * dy[r * outputs + o] for r in range(rows))
        for h in range(hidden)
        for o in range(outputs)
    ]
    db2 = [math.fsum(dy[r * outputs + o] for r in range(rows)) for o in range(outputs)]
    da = [
        math.fsum(dy[r * outputs + o] * p["w2"][h * outputs + o] for o in range(outputs))
        for r in range(rows)
        for h in range(hidden)
    ]
    dz = [g * gelu_derivative(v) for g, v in zip(da, z)]
    dw1 = [
        math.fsum(x[r * inputs + i] * dz[r * hidden + h] for r in range(rows))
        for i in range(inputs)
        for h in range(hidden)
    ]
    db1 = [math.fsum(dz[r * hidden + h] for r in range(rows)) for h in range(hidden)]
    dx = [
        math.fsum(dz[r * hidden + h] * p["w1"][i * hidden + h] for h in range(hidden))
        for r in range(rows)
        for i in range(inputs)
    ]
    gradients = dict(w1=dw1, b1=db1, w2=dw2, b2=db2)
    updated = {k: [v - learning_rate * g for v, g in zip(p[k], gradients[k])] for k in p}
    return dict(**state, dy=dy, da=da, dz=dz, dx=dx, gradients=gradients, updated=updated)
