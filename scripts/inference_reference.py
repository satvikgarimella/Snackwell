"""Independent references for generated kernels, not used to execute programs."""
import math

from fp32_reference import FADD, FSUB, FMUL, reference
from inference_kernels import bits, value


def tensor_product(a, b, m, n, k, transpose_b=False):
    """Match the documented two-level FP32 sum: four-term tiles, then tile sums."""
    out = []
    for r in range(m):
        for c in range(n):
            total = 0
            for base in range(0, k, 4):
                partial = 0
                for j in range(base, base+4):
                    right = b[c*k+j] if transpose_b else b[j*n+c]
                    partial = reference(FADD, partial, reference(FMUL, a[r*k+j], right))
                total = reference(FADD, total, partial)
            out.append(total)
    return out


def stable_softmax(row):
    maximum = max(row)
    if not math.isfinite(maximum):
        raise ValueError("softmax requires finite or -inf values, with at least one finite value per row")
    exps = [math.exp(x-maximum) for x in row]
    denominator = math.fsum(exps)
    return [x/denominator for x in exps]


def gelu_tanh(x):
    if abs(x) >= 10:
        return x if x > 0 else 0.0
    return 0.5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+0.044715*x*x*x)))


def expected_stage(stage, before, layout):
    arrays = [before[layout.regions[name][0]:sum(layout.regions[name])] for name in stage.inputs]
    x = arrays[0]
    p, kind = stage.params, stage.operation
    cols = p.get("cols", 0)
    if kind == "matmul":
        return tensor_product(*arrays, p["m"], p["n"], p["k"], p.get("transpose_b", False)), True
    if kind in ("bias_add", "residual_add"):
        return [reference(FADD, a, arrays[1][i % cols if kind == "bias_add" else i])
                for i, a in enumerate(x)], True
    if kind == "relu":
        # Ordered comparison leaves NaNs and signed zero unchanged.
        return [0 if value(a) < 0 else a for a in x], True
    if kind == "row_sum":
        out = []
        for r in range(p["rows"]):
            total = 0
            for a in x[r*cols:(r+1)*cols]:
                total = reference(FADD, total, a)
            out.append(total)
        return out, True
    if kind == "row_max":
        out = []
        for r in range(p["rows"]):
            row = x[r*cols:(r+1)*cols]
            total = row[0]
            for a in row[1:]:
                if value(total) < value(a):
                    total = a
            out.append(total)
        return out, True
    if kind == "scale_mask":
        scale = bits(1/math.sqrt(p["width"]))
        return [0xFF800000 if p["causal"] and c > r else reference(FMUL, x[r*cols+c], scale)
                for r in range(p["rows"]) for c in range(cols)], True
    if kind == "gelu":
        return [gelu_tanh(value(a)) for a in x], False
    if kind == "softmax":
        return [y for r in range(p["rows"]) for y in stable_softmax(
            [value(a) for a in x[r*cols:(r+1)*cols]])], False
    if kind == "layernorm":
        out = []
        for r in range(p["rows"]):
            row = x[r*cols:(r+1)*cols]
            mean = 0
            for a in row:
                mean = reference(FADD, mean, a)
            mean = reference(FMUL, mean, bits(1/cols))
            centered = [reference(FSUB, a, mean) for a in row]
            variance = 0
            for a in centered:
                variance = reference(FADD, variance, reference(FMUL, a, a))
            variance = reference(FADD, reference(FMUL, variance, bits(1/cols)), bits(p["epsilon"]))
            if not math.isfinite(value(variance)):
                raise ValueError("LayerNorm variance overflowed FP32; reduce input magnitudes")
            inv = 1/math.sqrt(value(variance))
            out.extend(value(a)*inv*value(arrays[1][c])+value(arrays[2][c]) for c, a in enumerate(centered))
        return out, False
    raise ValueError(kind)


def attention_reference(case):
    """Mathematical attention from original Q/K/V; independent of stage outputs."""
    layout, n, d = case.layout, case.rows, case.cols
    arrays = [[value(w) for w in layout.memory[start:start+count]]
              for start, count in [layout.regions[key] for key in ("q", "k", "v")]]
    q, k, v = arrays
    causal = case.stages[1].params["causal"]
    out = []
    for r in range(n):
        scores = [math.fsum(q[r*d+j]*k[c*d+j] for j in range(d))/math.sqrt(d)
                  if not causal or c <= r else -math.inf for c in range(n)]
        probabilities = stable_softmax(scores)
        out.extend(math.fsum(probabilities[c]*v[c*d+j] for c in range(n)) for j in range(d))
    return out


TOLERANCES = {"gelu": (1e-4, 3e-6), "softmax": (1e-4, 2e-6),
              "layernorm": (3e-5, 3e-6), "attention": (2e-4, 2e-5)}


def verify_approx(actual_bits, expected, kind):
    rtol, atol = TOLERANCES[kind]
    errors = []
    for i, (word, ref) in enumerate(zip(actual_bits, expected)):
        actual = value(word)
        error = abs(actual-ref)
        if not math.isfinite(actual) or not math.isfinite(ref) or error > atol + rtol*abs(ref):
            raise ValueError(f"{kind} output {i}: got {actual:.9g}, expected {ref:.9g}; tolerance {atol}+{rtol}*abs(ref)")
        errors.append(error)
    if len(actual_bits) != len(expected):
        raise ValueError("output length mismatch")
    return {"max_absolute_error": max(errors, default=0), "rtol": rtol, "atol": atol}


def verify_stage(stage, before, after, layout):
    expected, exact = expected_stage(stage, before, layout)
    start, count = layout.regions[stage.output]
    if len(before) != 1024 or len(after) != 1024 or len(expected) != count:
        raise ValueError("invalid memory/output size")
    for i, (old, new) in enumerate(zip(before, after)):
        if not start <= i < start+count and old != new:
            raise ValueError(f"{stage.name}: memory guard changed at byte {i*4}")
    actual = after[start:start+count]
    if exact:
        for i, (got, ref) in enumerate(zip(actual, expected)):
            nan = (ref & 0x7FFFFFFF) > 0x7F800000
            if not ((got & 0x7FFFFFFF) > 0x7F800000 if nan else got == ref):
                raise ValueError(f"{stage.name} output {i}: got {got:08x}, expected {ref:08x}")
        return {"comparison": "FP32 bits (NaNs by classification)"}
    metrics = verify_approx(actual, expected, stage.operation)
    if stage.operation == "softmax":
        for row in range(stage.params["rows"]):
            values = [value(w) for w in actual[row*stage.params["cols"]:(row+1)*stage.params["cols"]]]
            if any(x < 0 or x > 1.000001 for x in values) or abs(math.fsum(values)-1) > 2e-5:
                raise ValueError("softmax probabilities must be nonnegative and sum to one")
    return {"comparison": "numerical tolerance", **metrics}
