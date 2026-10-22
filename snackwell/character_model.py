"""Validated portable FP32 checkpoint and independent character-model reference."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from snackwell.numeric import bits, value
from snackwell.reference.inference import gelu_tanh, stable_softmax

FORMAT = "snackwell-character-transformer-v1"
FORMAT_V2 = "snackwell-character-transformer-v2"
FORMAT_SUBWORD = "snackwell-subword-transformer-v1"


def block_prefix(layer, layers):
    return "" if layers == 1 else f"block{layer}_"


def feedforward_width(config):
    return config.get("ffn_multiplier", 2) * config["width"]


def parameter_shapes(context, width, vocabulary, layers=1, ffn_multiplier=2):
    hidden = ffn_multiplier * width
    shapes = {"token_embedding": (vocabulary, width), "position_embedding": (context, width)}
    for layer in range(layers):
        prefix = block_prefix(layer, layers)
        for name in ("q", "k", "v", "o"):
            shapes[prefix + "w" + name] = (width, width)
            shapes[prefix + "b" + name] = (width,)
        shapes.update(
            {
                prefix + key: shape
                for key, shape in dict(
                    w1=(width, hidden),
                    b1=(hidden,),
                    w2=(hidden, width),
                    b2=(width,),
                    gamma1=(width,),
                    beta1=(width,),
                    gamma2=(width,),
                    beta2=(width,),
                ).items()
            }
        )
    shapes.update(gamma_final=(width,), beta_final=(width,))
    shapes.update(lm_weight=(width, vocabulary), lm_bias=(vocabulary,))
    return shapes


def validate_checkpoint(data):
    if not isinstance(data, dict) or data.get("format") not in (FORMAT, FORMAT_V2, FORMAT_SUBWORD):
        raise ValueError("unsupported model checkpoint format")
    c = data.get("config", {})
    keys = {"context", "width", "epsilon", "vocabulary"}
    subword = data["format"] == FORMAT_SUBWORD
    modern = data["format"] != FORMAT
    if modern:
        keys |= {"layers", "heads"}
        if isinstance(c, dict) and "ffn_multiplier" in c:
            keys.add("ffn_multiplier")
    if not isinstance(c, dict) or set(c) != keys:
        raise ValueError(f"checkpoint config requires {sorted(keys)}")
    n, d, vocab = c["context"], c["width"], c["vocabulary"]
    if subword:
        from snackwell.tokenizer import Tokenizer

        tokenizer = Tokenizer.restore(data.get("tokenizer"))
        if vocab != tokenizer.vocabulary:
            raise ValueError("checkpoint vocabulary does not match tokenizer")
    elif "tokenizer" in data:
        raise ValueError("character checkpoints cannot contain a subword tokenizer")
    if (
        type(n) is not int
        or n not in ((4, 8, 16, 32, 64, 128) if modern else (4, 8, 16))
        or type(d) is not int
        or d not in ((4, 8, 16, 32, 64, 128) if modern else (4, 8, 16))
    ):
        raise ValueError("unsupported model context or width")
    layers, heads = c.get("layers", 1), c.get("heads", 1)
    multiplier = c.get("ffn_multiplier", 2)
    if type(multiplier) is not int or multiplier not in (2, 4) or multiplier * d > 512:
        raise ValueError("feed-forward width must be 2x or 4x model width and at most 512")
    if (
        type(layers) is not int
        or layers not in (1, 2, 4)
        or type(heads) is not int
        or heads not in (1, 2, 4)
        or d % heads
        or d // heads < 4
    ):
        raise ValueError("model requires 1, 2 or 4 layers and 1, 2 or 4 heads of width >=4")
    if not subword and (
        not isinstance(vocab, list)
        or not 2 <= len(vocab) <= 128
        or any(not isinstance(x, str) or len(x) != 1 for x in vocab)
        or len(set(vocab)) != len(vocab)
    ):
        raise ValueError("vocabulary must contain 2..128 distinct characters")
    epsilon = value(bits(c["epsilon"]))
    if not math.isfinite(epsilon) or epsilon < value(0x00800000):
        raise ValueError("epsilon must be positive normal FP32")
    shapes = parameter_shapes(n, d, len(vocab), layers, multiplier)
    weights = data.get("weights", {})
    if not isinstance(weights, dict) or set(weights) != set(shapes):
        raise ValueError("missing or unexpected model parameter names")
    checked = {}
    for name, shape in shapes.items():
        values = weights[name]
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise ValueError(f"{name}: expected shape {shape}")
        words = [bits(x) for x in values]
        if any(not math.isfinite(value(w)) for w in words):
            raise ValueError(f"{name}: parameters must be finite FP32")
        checked[name] = [value(w) for w in words]
    return dict(data, config=dict(c, epsilon=epsilon), weights=checked)


def load_checkpoint(path):
    path = Path(path)
    raw = path.read_bytes()
    data = validate_checkpoint(json.loads(raw))
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def model_description(model):
    config = model["config"]
    feedforward = f"FFN width {feedforward_width(config)}, " if "ffn_multiplier" in config else ""
    unit = "subword tokens" if "tokenizer" in model else "characters"
    return (
        f"{config.get('layers', 1)} blocks, {config.get('heads', 1)} heads, "
        f"width {config['width']}, {feedforward}context {config['context']} {unit}; "
        f"checkpoint {model.get('sha256', 'unsaved')[:12]}"
    )


def encode_prompt(model, prompt, *, pad=True):
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must contain at least one character")
    vocab = model["config"]["vocabulary"]
    mapping = {ch: i for i, ch in enumerate(vocab)}
    unknown = sorted(set(prompt) - set(mapping))
    if unknown:
        raise ValueError(f"prompt contains characters outside the vocabulary: {unknown!r}")
    n = model["config"]["context"]
    if "tokenizer" in model:
        from snackwell.tokenizer import Tokenizer

        tokenizer = Tokenizer.restore(model["tokenizer"])
        ids = tokenizer.encode(prompt)
        if pad and len(ids) < n:
            if " " not in tokenizer.alphabet:
                raise ValueError(f"prompt must contain at least {n} tokens for this vocabulary")
            ids = [tokenizer.alphabet[" "]] * (n - len(ids)) + ids
        return ids[-n:]
    if pad and len(prompt) < n:
        if " " not in mapping:
            raise ValueError(f"prompt must contain at least {n} characters for this vocabulary")
        prompt = " " * (n - len(prompt)) + prompt
    return [mapping[ch] for ch in prompt[-n:]]


@dataclass(frozen=True)
class ReferenceState:
    """Independent double-precision history, never populated from RTL words."""

    ids: tuple
    layers: tuple
    observed: dict


def reference_forward(model, ids, *, previous=None, return_state=False):
    """Model math from checkpoint/IDs, optionally reusing its own causal history.

    The runtime binds this history to the same model and prefix as the RTL cache.
    Call without previous to independently recompute every row.
    """
    c, w = model["config"], model["weights"]
    context, d, v = len(ids), c["width"], len(c["vocabulary"])
    h = feedforward_width(c)
    if not 1 <= context <= c["context"] or any(type(i) is not int or not 0 <= i < v for i in ids):
        raise ValueError("invalid token input")
    start = len(previous.ids) if previous else 0
    layers, heads = c.get("layers", 1), c.get("heads", 1)
    if previous and (
        tuple(ids[:-1]) != previous.ids
        or context != start + 1
        or len(previous.layers) != layers
        or any(len(k) != start * d or len(value) != start * d for k, value in previous.layers)
    ):
        raise ValueError("reference history does not match this prefix")
    n = context - start
    pending_layers = []

    def norm(x, suffix, prefix=""):
        gamma, beta = w[prefix + "gamma" + suffix], w[prefix + "beta" + suffix]
        result = []
        for row in range(n):
            a = x[row * d : (row + 1) * d]
            mean = math.fsum(a) / d
            variance = math.fsum((z - mean) ** 2 for z in a) / d
            inv = 1 / math.sqrt(variance + c["epsilon"])
            result.extend((z - mean) * inv * gamma[col] + beta[col] for col, z in enumerate(a))
        return result

    def linear(x, weight, bias, inner, columns):
        return [
            math.fsum(x[r * inner + k] * w[weight][k * columns + col] for k in range(inner))
            + w[bias][col]
            for r in range(n)
            for col in range(columns)
        ]

    x = [
        w["token_embedding"][token * d + col] + w["position_embedding"][(start + r) * d + col]
        for r, token in enumerate(ids[start:])
        for col in range(d)
    ]
    observed = {"embedding": x}
    head_width = d // heads
    for layer in range(layers):
        prefix = block_prefix(layer, layers)
        z = norm(x, "1", prefix)
        q, k, values = [
            linear(z, prefix + "w" + name, prefix + "b" + name, d, d) for name in ("q", "k", "v")
        ]
        if previous:
            old_k, old_v = previous.layers[layer]
            k, values = list(old_k) + k, list(old_v) + values
        pending_layers.append((tuple(k), tuple(values)))
        attended = []
        for row in range(n):
            for head in range(heads):
                features = range(head * head_width, (head + 1) * head_width)
                scores = [
                    math.fsum(q[row * d + j] * k[col * d + j] for j in features)
                    / math.sqrt(head_width)
                    if col <= start + row
                    else -math.inf
                    for col in range(context)
                ]
                p = stable_softmax(scores)
                attended.extend(
                    math.fsum(p[col] * values[col * d + j] for col in range(context))
                    for j in features
                )
        projection = linear(attended, prefix + "wo", prefix + "bo", d, d)
        residual = [a + b for a, b in zip(x, projection)]
        hidden = [
            gelu_tanh(a)
            for a in linear(norm(residual, "2", prefix), prefix + "w1", prefix + "b1", d, h)
        ]
        projection = linear(hidden, prefix + "w2", prefix + "b2", h, d)
        x = [a + b for a, b in zip(residual, projection)]
        observed[prefix + "block"] = x
    block = x
    normalized = norm(block, "_final")
    logits = linear(normalized, "lm_weight", "lm_bias", d, v)
    observed.update(
        block=block,
        final_norm=normalized,
        logits=logits,
        probabilities=[p for r in range(n) for p in stable_softmax(logits[r * v : (r + 1) * v])],
    )
    observed["predictions"] = [max(range(v), key=lambda j: logits[r * v + j]) for r in range(n)]
    if previous:
        observed = {name: list(previous.observed[name]) + data for name, data in observed.items()}
    if return_state:
        state = ReferenceState(
            tuple(ids),
            tuple(pending_layers),
            {name: tuple(data) for name, data in observed.items()},
        )
        return observed, state
    return observed
