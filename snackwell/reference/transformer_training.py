"""Double-precision reverse-mode reference, independent of RTL intermediates."""

import math

from snackwell.character_model import block_prefix, feedforward_width
from snackwell.numeric import value
from snackwell.reference.inference import gelu_tanh, stable_softmax
from snackwell.reference.training import gelu_derivative


def normalization(x, gamma, dy, epsilon):
    count = len(x)
    mean = math.fsum(x) / count
    variance = math.fsum((a - mean) ** 2 for a in x) / count
    inv = 1 / math.sqrt(variance + epsilon)
    normalized = [(a - mean) * inv for a in x]
    u = [a * b for a, b in zip(dy, gamma)]
    mean_u = math.fsum(u) / count
    mean_ux = math.fsum(a * b for a, b in zip(u, normalized)) / count
    return normalized, [inv * (g - mean_u - a * mean_ux) for g, a in zip(u, normalized)]


def expected_stage(kind, arrays, params):
    rows, cols = params["rows"], params["cols"]
    x = list(map(value, arrays[0]))
    if kind == "cross_entropy":
        out = []
        for r, target in enumerate(arrays[1]):
            row = x[r * cols : (r + 1) * cols]
            maximum = max(row)
            out.append(
                maximum - row[target] + math.log(math.fsum(math.exp(a - maximum) for a in row))
            )
        return out
    dy = list(map(value, arrays[1]))
    out, dg = [], []
    for r in range(rows):
        a, g = x[r * cols : (r + 1) * cols], dy[r * cols : (r + 1) * cols]
        if kind == "softmax_backward":
            dot = math.fsum(p * v for p, v in zip(a, g))
            out.extend(p * (v - dot) * params["scale"] for p, v in zip(a, g))
        else:
            normalized, dx = normalization(a, list(map(value, arrays[2])), g, params["epsilon"])
            out.extend(dx)
            dg.extend(p * v for p, v in zip(normalized, g))
    return out + dg


class Node:
    def __init__(self, values, propagate=None):
        self.values = list(values)
        self.grad = [0.0] * len(self.values)
        self.propagate = propagate

    def accumulate(self, gradient):
        if len(gradient) != len(self.grad):
            raise ValueError("reference gradient shape mismatch")
        self.grad = [a + b for a, b in zip(self.grad, gradient)]


class Tape:
    def __init__(self):
        self.nodes = []

    def record(self, values, propagate):
        node = Node(values, propagate)
        self.nodes.append(node)
        return node

    def add(self, a, b):
        def backward(g):
            a.accumulate(g)
            b.accumulate(g)

        return self.record([x + y for x, y in zip(a.values, b.values)], backward)

    def gather(self, source, indices):
        def backward(g):
            gradient = [0.0] * len(source.values)
            for index, v in zip(indices, g):
                gradient[index] += v
            source.accumulate(gradient)

        return self.record([source.values[i] for i in indices], backward)

    def matmul(self, a, b, m, n, k):
        values = [
            math.fsum(a.values[r * k + j] * b.values[j * n + c] for j in range(k))
            for r in range(m)
            for c in range(n)
        ]

        def backward(g):
            a.accumulate(
                [
                    math.fsum(g[r * n + c] * b.values[j * n + c] for c in range(n))
                    for r in range(m)
                    for j in range(k)
                ]
            )
            b.accumulate(
                [
                    math.fsum(a.values[r * k + j] * g[r * n + c] for r in range(m))
                    for j in range(k)
                    for c in range(n)
                ]
            )

        return self.record(values, backward)

    def bias(self, x, bias, cols):
        def backward(g):
            x.accumulate(g)
            bias.accumulate([math.fsum(g[c::cols]) for c in range(cols)])

        return self.record([v + bias.values[i % cols] for i, v in enumerate(x.values)], backward)

    def norm(self, x, gamma, beta, cols, epsilon):
        values = []
        for start in range(0, len(x.values), cols):
            a, _ = normalization(x.values[start : start + cols], gamma.values, [0] * cols, epsilon)
            values.extend(v * g + b for v, g, b in zip(a, gamma.values, beta.values))

        def backward(g):
            dx, dg, db = [], [0.0] * cols, [0.0] * cols
            for start in range(0, len(g), cols):
                row = g[start : start + cols]
                a, derivative = normalization(
                    x.values[start : start + cols], gamma.values, row, epsilon
                )
                dx.extend(derivative)
                for c in range(cols):
                    dg[c] += row[c] * a[c]
                    db[c] += row[c]
            x.accumulate(dx)
            gamma.accumulate(dg)
            beta.accumulate(db)

        return self.record(values, backward)

    def gelu(self, x):
        return self.record(
            [gelu_tanh(v) for v in x.values],
            lambda g: x.accumulate([v * gelu_derivative(a) for a, v in zip(x.values, g)]),
        )

    def causal_softmax(self, scores, rows, scale):
        p = [
            v
            for r in range(rows)
            for v in stable_softmax(
                [scores.values[r * rows + c] * scale if c <= r else -math.inf for c in range(rows)]
            )
        ]

        def backward(g):
            dx = []
            for r in range(rows):
                a, b = p[r * rows : (r + 1) * rows], g[r * rows : (r + 1) * rows]
                dot = math.fsum(x * y for x, y in zip(a, b))
                dx.extend(x * (y - dot) * scale for x, y in zip(a, b))
            scores.accumulate(dx)

        return self.record(p, backward)


def reference(model, ids, targets, rate=None):
    """Build the complete character transformer and differentiate its mean CE loss."""
    config = model["config"]
    hidden = feedforward_width(config)
    n, d, v = len(ids), config["width"], len(config["vocabulary"])
    layers, heads = config.get("layers", 1), config.get("heads", 1)
    p = {name: Node(data) for name, data in model["weights"].items()}
    tape, observed = Tape(), {}

    def note(name, node):
        observed[name] = node
        return node

    def norm(name, x, suffix, prefix=""):
        return note(
            name,
            tape.norm(
                x, p[prefix + "gamma" + suffix], p[prefix + "beta" + suffix], d, config["epsilon"]
            ),
        )

    def linear(name, x, weight, bias, inner, cols):
        return note(name, tape.bias(tape.matmul(x, p[weight], n, cols, inner), p[bias], cols))

    x = note(
        "embedding",
        tape.add(
            tape.gather(p["token_embedding"], [i * d + j for i in ids for j in range(d)]),
            tape.gather(p["position_embedding"], list(range(n * d))),
        ),
    )
    for layer in range(layers):
        prefix = block_prefix(layer, layers)
        z = norm(prefix + "norm1", x, "1", prefix)
        q, k, values = [
            linear(prefix + name, z, prefix + "w" + name, prefix + "b" + name, d, d)
            for name in ("q", "k", "v")
        ]
        products = []
        h = d // heads
        for head in range(heads):
            indexes = [r * d + head * h + j for r in range(n) for j in range(h)]
            query, key, val = [tape.gather(t, indexes) for t in (q, k, values)]
            kt = tape.gather(key, [r * h + c for c in range(h) for r in range(n)])
            probabilities = note(
                prefix + f"head{head}_probabilities",
                tape.causal_softmax(tape.matmul(query, kt, n, n, h), n, 1 / math.sqrt(h)),
            )
            products.append(tape.matmul(probabilities, val, n, h, n))

        # Merge heads as a gather over a differentiable concatenation.
        def unmerge(g, products=tuple(products), h=h):
            for head, node in enumerate(products):
                node.accumulate([g[r * d + head * h + j] for r in range(n) for j in range(h)])

        attended = note(
            prefix + "attended",
            tape.record(
                [
                    products[head].values[r * h + j]
                    for r in range(n)
                    for head in range(heads)
                    for j in range(h)
                ],
                unmerge,
            ),
        )
        projected = linear(
            prefix + "attention_projection", attended, prefix + "wo", prefix + "bo", d, d
        )
        residual = note(prefix + "residual", tape.add(x, projected))
        z = norm(prefix + "norm2", residual, "2", prefix)
        z = linear(prefix + "ff1", z, prefix + "w1", prefix + "b1", d, hidden)
        z = note(prefix + "gelu", tape.gelu(z))
        z = linear(prefix + "ff2", z, prefix + "w2", prefix + "b2", hidden, d)
        x = note(prefix + "block", tape.add(residual, z))
    x = norm("final_norm", x, "_final")
    logits = linear("logits", x, "lm_weight", "lm_bias", d, v)
    probabilities = [
        p for r in range(n) for p in stable_softmax(logits.values[r * v : (r + 1) * v])
    ]
    losses = []
    for r, target in enumerate(targets):
        row = logits.values[r * v : (r + 1) * v]
        maximum = max(row)
        losses.append(
            maximum - row[target] + math.log(math.fsum(math.exp(a - maximum) for a in row))
        )
    logits.accumulate(
        [(p - int(i % v == targets[i // v])) / n for i, p in enumerate(probabilities)]
    )
    for node in reversed(tape.nodes):
        node.propagate(node.grad)
    output = {name: node.values for name, node in observed.items()}
    output.update(probabilities=probabilities, loss=[math.fsum(losses) / n])
    gradients = {name: node.grad for name, node in p.items()}
    result = dict(
        observed=output, gradients=gradients, embedding_gradient=observed["embedding"].grad
    )
    if rate is not None:
        result["updated"] = {
            name: [x - rate * g for x, g in zip(node.values, node.grad)] for name, node in p.items()
        }
    return result
