"""Independent mathematical and integer-FP32 checks for pretrained kernels."""

import math

from snackwell.numeric import bits, value
from snackwell.reference.fp32 import FADD, FMUL, reference


def expected(kind, arrays, params):
    x = arrays[0]
    if kind in ("sum", "squares", "max"):
        total = arrays[1][0]
        for item in x:
            if kind == "max":
                if value(item) > value(total):
                    total = item
            else:
                term = reference(FMUL, item, item) if kind == "squares" else item
                total = reference(FADD, total, term)
        return [total], True
    if kind in ("rms_inverse", "reciprocal"):
        if kind == "rms_inverse":
            v = reference(
                FADD, reference(FMUL, x[0], bits(1 / params["width"])), bits(params["epsilon"])
            )
            return [1 / math.sqrt(value(v))], False
        return [1 / value(x[0])], False
    if kind in ("add", "multiply"):
        op = FADD if kind == "add" else FMUL
        return [reference(op, a, b) for a, b in zip(x, arrays[1])], True
    if kind in ("scale", "norm"):
        scaled = [reference(FMUL, a, arrays[1][0]) for a in x]
        if kind == "norm":
            scaled = [reference(FMUL, a, b) for a, b in zip(scaled, arrays[2])]
        return scaled, True
    if kind == "rope":
        return [
            reference(FADD, reference(FMUL, a, c), reference(FMUL, b, s))
            for a, b, c, s in zip(*arrays)
        ], True
    if kind == "exp":
        maximum, temperature_inverse = value(arrays[1][0]), value(arrays[2][0])
        return [math.exp((value(a) - maximum) * temperature_inverse) for a in x], False
    if kind == "swiglu":
        out = []
        for a, b in zip(x, arrays[1]):
            a, b = value(a), value(b)
            exponential = math.exp(-abs(a))
            sigmoid = (1 if a >= 0 else exponential) / (1 + exponential)
            out.append(a * sigmoid * b)
        return out, False
    if kind == "argmax":
        best, index = arrays[1]
        for i, word in enumerate(x, params["offset"]):
            if value(word) > value(best):
                best, index = word, i
        return [best, index], True
    if kind == "categorical":
        total, selected, last = arrays[1]
        threshold = value(arrays[2][0])
        for i, word in enumerate(x, params["offset"]):
            if not math.isfinite(value(word)) or value(word) < 0:
                raise ValueError("sampling requires finite nonnegative weights")
            total = reference(FADD, total, word)
            if value(word) > 0:
                last = i
            if selected == 0xFFFFFFFF and value(total) > threshold:
                selected = i
        return [total, selected, last], True
    if kind == "lookup":
        index = x[0] * params["columns"]
        return arrays[1][index : index + params["columns"]], True
    raise ValueError(kind)
