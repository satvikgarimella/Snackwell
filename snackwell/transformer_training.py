"""Full causal character-transformer forward, backward and SGD on the SM."""

import math
import time

from snackwell.character_model import block_prefix, feedforward_width, validate_checkpoint
from snackwell.model_runtime import insert_rectangle, rectangle
from snackwell.numeric import bits, value
from snackwell.reference.inference import verify_approx
from snackwell.reference.transformer_training import reference
from snackwell.training import learning_rate_word, write_json
from snackwell.training_runtime import transpose


def validate_example(model, ids, targets):
    model = validate_checkpoint(model)
    c = model["config"]
    if (
        not isinstance(ids, list)
        or len(ids) != c["context"]
        or not isinstance(targets, list)
        or len(targets) != len(ids)
    ):
        raise ValueError("training requires one full context of input and target IDs")
    if any(type(i) is not int or not 0 <= i < len(c["vocabulary"]) for i in ids + targets):
        raise ValueError("input or target ID outside vocabulary")
    return model


def _forward(runtime, model, ids, targets):
    c = model["config"]
    hidden = feedforward_width(c)
    n, d, v = len(ids), c["width"], len(c["vocabulary"])
    layers, heads = c.get("layers", 1), c.get("heads", 1)
    p = {name: list(map(bits, data)) for name, data in model["weights"].items()}
    observed, blocks = {}, []

    def note(name, data):
        observed[name] = data
        return data

    def norm(name, x, suffix, prefix=""):
        return note(
            name,
            runtime.rows(
                name,
                "layernorm",
                x,
                n,
                d,
                gamma=p[prefix + "gamma" + suffix],
                beta=p[prefix + "beta" + suffix],
                epsilon=c["epsilon"],
            ),
        )

    def linear(name, x, weight, bias, inner, columns):
        return note(name, runtime.matmul(name, x, p[weight], n, columns, inner, bias=p[bias]))

    x = note("embedding", runtime.lookup(ids, p["token_embedding"], p["position_embedding"], d, v))
    for layer in range(layers):
        prefix = block_prefix(layer, layers)
        z = norm(prefix + "norm1", x, "1", prefix)
        q, k, values = [
            linear(prefix + name, z, prefix + "w" + name, prefix + "b" + name, d, d)
            for name in ("q", "k", "v")
        ]
        attended, attention = [0] * (n * d), []
        h = d // heads
        for head in range(heads):
            query, key, val = [rectangle(t, d, 0, n, head * h, h) for t in (q, k, values)]
            name = prefix + f"head{head}_"
            scores = runtime.matmul(name + "qk", query, key, n, n, h, transpose_b=True)
            scores = runtime.attention_scale(scores, n, h)
            probabilities = note(
                name + "probabilities", runtime.rows(name + "softmax", "softmax", scores, n, n)
            )
            product = runtime.matmul(name + "pv", probabilities, val, n, h, n)
            insert_rectangle(attended, d, 0, n, head * h, h, product)
            attention.append(dict(q=query, k=key, v=val, p=probabilities))
        note(prefix + "attended", attended)
        projected = linear(
            prefix + "attention_projection", attended, prefix + "wo", prefix + "bo", d, d
        )
        residual = note(
            prefix + "residual",
            runtime.arithmetic(prefix + "attention_residual", "add", x, projected),
        )
        z = norm(prefix + "norm2", residual, "2", prefix)
        preactivation = linear(prefix + "ff1", z, prefix + "w1", prefix + "b1", d, hidden)
        activation = note(
            prefix + "gelu", runtime.rows(prefix + "gelu", "gelu", preactivation, n, hidden)
        )
        projected = linear(prefix + "ff2", activation, prefix + "w2", prefix + "b2", hidden, d)
        blocks.append(dict(x=x, attention=attention))
        x = note(
            prefix + "block", runtime.arithmetic(prefix + "ff_residual", "add", residual, projected)
        )
    normalized = norm("final_norm", x, "_final")
    padded = max(4, 1 << (v - 1).bit_length())
    lm_weight = [
        w for r in range(d) for w in p["lm_weight"][r * v : (r + 1) * v] + [0] * (padded - v)
    ]
    logits = runtime.matmul(
        "logits", normalized, lm_weight, n, padded, d, bias=p["lm_bias"] + [0] * (padded - v)
    )
    note("logits", rectangle(logits, padded, 0, n, 0, v))
    masked = runtime.rows("vocabulary_mask", "vocabulary_mask", logits, n, padded, vocabulary=v)
    probabilities = runtime.rows("probabilities", "softmax", masked, n, padded)
    note("probabilities", rectangle(probabilities, padded, 0, n, 0, v))
    note("loss", runtime.cross_entropy(masked, targets, n, padded))
    return observed, blocks, p, lm_weight, probabilities, padded


def _backward(runtime, model, ids, targets, state):
    o, blocks, p, lm_weight, probabilities, padded = state
    c = model["config"]
    hidden = feedforward_width(c)
    n, d, v = len(ids), c["width"], len(c["vocabulary"])
    layers, heads = c.get("layers", 1), c.get("heads", 1)
    gradients = {}

    def linear(name, x, dy, weight, bias, inner, columns, weights=None):
        weights = p[weight] if weights is None else weights
        dw = runtime.matmul("grad_" + weight, transpose(x, n, inner), dy, inner, columns, n)
        db = runtime.column_sum("grad_" + bias, dy, n, columns)
        gradients[weight], gradients[bias] = dw, db
        return runtime.matmul(name, dy, weights, n, inner, columns, transpose_b=True)

    def norm(name, x, dy, suffix, prefix=""):
        gamma, beta = prefix + "gamma" + suffix, prefix + "beta" + suffix
        dx, dg, db = runtime.norm_backward(name, x, dy, p[gamma], n, d, c["epsilon"])
        gradients[gamma], gradients[beta] = dg, db
        return dx

    dy = runtime.classifier_backward(probabilities, targets, padded)
    dy = linear(
        "grad_final_norm", o["final_norm"], dy, "lm_weight", "lm_bias", d, padded, lm_weight
    )
    gradients["lm_weight"] = rectangle(gradients["lm_weight"], padded, 0, d, 0, v)
    gradients["lm_bias"] = gradients["lm_bias"][:v]
    last = block_prefix(layers - 1, layers) + "block"
    dy = norm("grad_final_layernorm", o[last], dy, "_final")
    for layer in reversed(range(layers)):
        prefix = block_prefix(layer, layers)
        skip = dy
        da = linear(
            prefix + "grad_gelu", o[prefix + "gelu"], dy, prefix + "w2", prefix + "b2", hidden, d
        )
        dz = runtime.arithmetic(prefix + "grad_ff1", "gelu_backward", o[prefix + "ff1"], da)
        dz = linear(
            prefix + "grad_norm2", o[prefix + "norm2"], dz, prefix + "w1", prefix + "b1", d, hidden
        )
        dz = norm(prefix + "grad_layernorm2", o[prefix + "residual"], dz, "2", prefix)
        dy = runtime.arithmetic(prefix + "grad_residual", "add", skip, dz)
        skip = dy
        da = linear(
            prefix + "grad_attended", o[prefix + "attended"], dy, prefix + "wo", prefix + "bo", d, d
        )
        dq, dk, dv = ([0] * (n * d) for _ in range(3))
        h = d // heads
        for head, cached in enumerate(blocks[layer]["attention"]):
            name = prefix + f"grad_head{head}_"
            upstream = rectangle(da, d, 0, n, head * h, h)
            dp = runtime.matmul(name + "p", upstream, cached["v"], n, n, h, transpose_b=True)
            value_gradient = runtime.matmul(
                name + "v", transpose(cached["p"], n, n), upstream, n, h, n
            )
            ds = runtime.softmax_backward(name + "softmax", cached["p"], dp, n, n, 1 / math.sqrt(h))
            query_gradient = runtime.matmul(name + "q", ds, cached["k"], n, h, n)
            key_gradient = runtime.matmul(name + "k", transpose(ds, n, n), cached["q"], n, h, n)
            for dest, src in ((dq, query_gradient), (dk, key_gradient), (dv, value_gradient)):
                insert_rectangle(dest, d, 0, n, head * h, h, src)
        parts = [
            linear(
                prefix + "grad_norm1_" + name,
                o[prefix + "norm1"],
                g,
                prefix + "w" + name,
                prefix + "b" + name,
                d,
                d,
            )
            for name, g in (("q", dq), ("k", dk), ("v", dv))
        ]
        dz = runtime.arithmetic(prefix + "grad_norm1_qk", "add", parts[0], parts[1])
        dz = runtime.arithmetic(prefix + "grad_norm1_qkv", "add", dz, parts[2])
        dz = norm(prefix + "grad_layernorm1", blocks[layer]["x"], dz, "1", prefix)
        dy = runtime.arithmetic(prefix + "grad_input", "add", skip, dz)
    gradients["position_embedding"] = list(dy)
    gradients["token_embedding"] = runtime.embedding_backward(ids, dy, d, v)
    return gradients, dy


def _check(actual, expected, label):
    result = {}
    if set(actual) != set(expected):
        raise ValueError(f"{label}: missing or unexpected tensor names")
    for name, words in actual.items():
        try:
            result[name] = verify_approx(words, expected[name], "transformer_training")
        except ValueError as error:
            raise ValueError(f"{label} {name}: {error}") from error
    return result


def run_step(runtime, model, ids, targets, learning_rate=None):
    """Evaluate, or perform one complete sequence update; commit only checked weights."""
    (runtime.output / "result.json").unlink(missing_ok=True)
    model = validate_example(model, ids, targets)
    rate = None if learning_rate is None else value(learning_rate_word(learning_rate))
    began, start = time.monotonic(), len(runtime.reports)
    expected = reference(model, ids, targets, rate)
    state = _forward(runtime, model, ids, targets)
    observed, _, parameters, _, _, _ = state
    checks = dict(forward=_check(observed, expected["observed"], "forward"))
    tensors = dict(observed=observed)
    updated_model = None
    if rate is not None:
        gradients, dx = _backward(runtime, model, ids, targets, state)
        checks["gradients"] = _check(gradients, expected["gradients"], "gradient")
        checks["embedding_gradient"] = verify_approx(
            dx, expected["embedding_gradient"], "transformer_training"
        )
        updated = {
            name: runtime.arithmetic("update_" + name, "sgd", words, gradients[name], scale=rate)
            for name, words in parameters.items()
        }
        checks["updated"] = _check(updated, expected["updated"], "update")
        tensors.update(gradients=gradients, embedding_gradient=dx, updated=updated)
        updated_model = validate_checkpoint(
            dict(
                format=model["format"],
                config=model["config"],
                **({"tokenizer": model["tokenizer"]} if "tokenizer" in model else {}),
                weights={k: list(map(value, w)) for k, w in updated.items()},
            )
        )
    stages = runtime.reports[start:]
    result = dict(
        verified=True,
        loss=value(observed["loss"][0]),
        cycles=sum(s["cycles"] for s in stages),
        launches=len(stages),
        seconds=time.monotonic() - began,
        checks=checks,
        tensors=tensors,
        stages=stages,
        ids=ids,
        targets=targets,
    )
    write_json(runtime.output / "result.json", result)
    return result, updated_model
