"""A two-layer GELU network trained by the SM, with independently checked steps."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from snackwell.numeric import bits, value
from snackwell.reference.inference import verify_approx
from snackwell.reference.training import backward, forward
from snackwell.training_runtime import transpose

FORMAT = "snackwell.ffn-sgd.v1"
DATASET = "nonlinear-regression-v1"


@dataclass(frozen=True)
class Shape:
    inputs: int = 8
    hidden: int = 16
    outputs: int = 4

    def __post_init__(self):
        if any(
            type(n) is not int or n < 4 or n > 128 or n % 4 for n in (self.inputs, self.outputs)
        ):
            raise ValueError("input/output widths must be multiples of four in 4..128")
        if type(self.hidden) is not int or self.hidden not in (4, 8, 16, 32, 64, 128, 256):
            raise ValueError("hidden width must be a power of two in 4..256")

    def sizes(self):
        return dict(
            w1=self.inputs * self.hidden,
            b1=self.hidden,
            w2=self.hidden * self.outputs,
            b2=self.outputs,
        )


def finite_words(words, count, name):
    if (
        not isinstance(words, list)
        or len(words) != count
        or any(
            type(w) is not int or not 0 <= w <= 0xFFFFFFFF or not math.isfinite(value(w))
            for w in words
        )
    ):
        raise ValueError(f"{name} must contain {count} finite FP32 words")


def learning_rate_word(rate):
    try:
        word = bits(rate)
    except (ValueError, OverflowError) as error:
        raise ValueError("learning rate must be finite and positive in FP32") from error
    if not math.isfinite(value(word)) or value(word) <= 0:
        raise ValueError("learning rate must be finite and positive in FP32")
    return word


def validate_parameters(parameters, shape):
    if not isinstance(parameters, dict) or set(parameters) != set(shape.sizes()):
        raise ValueError("parameters must contain exactly w1, b1, w2, b2")
    for name, count in shape.sizes().items():
        finite_words(parameters[name], count, name)


def validate_batch(parameters, x, target, shape):
    validate_parameters(parameters, shape)
    rows, remainder = divmod(len(x), shape.inputs)
    if remainder or rows < 4 or rows % 4:
        raise ValueError("training/evaluation batches require a positive multiple of four rows")
    finite_words(x, rows * shape.inputs, "inputs")
    finite_words(target, rows * shape.outputs, "targets")
    return rows


def initialize(shape, seed):
    rng = random.Random(seed)
    return dict(
        w1=[
            bits(rng.uniform(-1, 1) * math.sqrt(3 / shape.inputs))
            for _ in range(shape.inputs * shape.hidden)
        ],
        b1=[0] * shape.hidden,
        w2=[
            bits(rng.uniform(-1, 1) * math.sqrt(3 / shape.hidden))
            for _ in range(shape.hidden * shape.outputs)
        ],
        b2=[0] * shape.outputs,
    )


def _forward(runtime, p, x, target, rows, shape):
    z = runtime.matmul("linear1", x, p["w1"], rows, shape.hidden, shape.inputs, bias=p["b1"])
    a = runtime.rows("gelu", "gelu", z, rows, shape.hidden)
    y = runtime.matmul("linear2", a, p["w2"], rows, shape.outputs, shape.hidden, bias=p["b2"])
    return dict(z=z, a=a, y=y, loss=runtime.loss(y, target))


def _check(actual, expected):
    checks = {}
    for name, tensor in actual.items():
        try:
            checks[name] = (
                _check(tensor, expected[name])
                if isinstance(tensor, dict)
                else verify_approx(tensor, expected[name], "training")
            )
        except ValueError as error:
            raise ValueError(f"training {name}: {error}") from error
    return checks


def write_json(path, document):
    """Replace a completed record atomically; interrupted writes cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _record(runtime, start, tensors, checks, began, name):
    launches = runtime.reports[start:]
    result = dict(
        verified=True,
        loss=value(tensors["loss"][0]),
        cycles=sum(r["cycles"] for r in launches),
        launches=len(launches),
        seconds=time.monotonic() - began,
        checks=checks,
        tensors=tensors,
        stages=launches,
    )
    write_json(runtime.output / name, result)
    return result


def evaluate(runtime, parameters, x, target, shape):
    (runtime.output / "evaluation.json").unlink(missing_ok=True)
    rows = validate_batch(parameters, x, target, shape)
    began, start = time.monotonic(), len(runtime.reports)
    tensors = _forward(runtime, parameters, x, target, rows, shape)
    expected = forward(
        {k: list(map(value, v)) for k, v in parameters.items()},
        list(map(value, x)),
        list(map(value, target)),
        rows,
        **asdict(shape),
    )
    return _record(runtime, start, tensors, _check(tensors, expected), began, "evaluation.json")


def train_step(runtime, parameters, x, target, shape, learning_rate):
    """Return new weights only after every launch and the entire graph pass checks.

    The caller's parameters are never mutated. All gradients use the old weights;
    input gradients are included so this layer can later be composed with others.
    """
    (runtime.output / "step.json").unlink(missing_ok=True)
    rows = validate_batch(parameters, x, target, shape)
    rate = value(learning_rate_word(learning_rate))
    began, start = time.monotonic(), len(runtime.reports)
    expected = backward(
        {k: list(map(value, v)) for k, v in parameters.items()},
        list(map(value, x)),
        list(map(value, target)),
        rows,
        **asdict(shape),
        learning_rate=rate,
    )
    t = _forward(runtime, parameters, x, target, rows, shape)
    dy = runtime.vector("loss_gradient", "mse_gradient", t["y"], target, 1 / len(target))
    dw2 = runtime.matmul(
        "grad_w2", transpose(t["a"], rows, shape.hidden), dy, shape.hidden, shape.outputs, rows
    )
    db2 = runtime.column_sum("grad_b2", dy, rows, shape.outputs)
    da = runtime.matmul(
        "grad_activation", dy, parameters["w2"], rows, shape.hidden, shape.outputs, transpose_b=True
    )
    dz = runtime.vector("gelu_backward", "gelu_backward", t["z"], da)
    dw1 = runtime.matmul(
        "grad_w1", transpose(x, rows, shape.inputs), dz, shape.inputs, shape.hidden, rows
    )
    db1 = runtime.column_sum("grad_b1", dz, rows, shape.hidden)
    dx = runtime.matmul(
        "grad_input", dz, parameters["w1"], rows, shape.inputs, shape.hidden, transpose_b=True
    )
    gradients = dict(w1=dw1, b1=db1, w2=dw2, b2=db2)
    updated = {
        k: runtime.vector(f"update_{k}", "sgd", parameters[k], gradients[k], rate)
        for k in parameters
    }
    t.update(dy=dy, da=da, dz=dz, dx=dx, gradients=gradients, updated=updated)
    return _record(runtime, start, t, _check(t, expected), began, "step.json")


def dataset(seed):
    """Disjoint deterministic train/held-out inputs; targets are host-created data."""
    rng = random.Random(seed + 1000)
    records = []
    for _ in range(96):
        x = [value(bits(rng.uniform(-1, 1))) for _ in range(8)]
        y = [
            math.sin(x[0]) + 0.25 * x[1] * x[2],
            0.5 * (x[2] + x[3]) ** 2 - 0.3,
            x[4] * math.tanh(x[5]),
            0.5 * x[6] - 0.25 * x[7] + 0.3 * x[0] * x[4],
        ]
        records.append(([bits(v) for v in x], [bits(v) for v in y]))

    def flatten(samples):
        return ([w for x, _ in samples for w in x], [w for _, y in samples for w in y])

    return flatten(records[:64]), flatten(records[64:])


def batch(data, seed, step, count=16):
    x, y = data
    indices = random.Random(f"{seed}:{step}").sample(range(len(x) // 8), count)
    return (
        [w for i in indices for w in x[i * 8 : (i + 1) * 8]],
        [w for i in indices for w in y[i * 4 : (i + 1) * 4]],
    )


def checkpoint(parameters, shape, step, seed, learning_rate):
    result = dict(
        format=FORMAT,
        dataset=DATASET,
        shape=asdict(shape),
        parameter_words=parameters,
        completed_steps=step,
        seed=seed,
        learning_rate=value(learning_rate_word(learning_rate)),
        batch_size=16,
    )
    validate_checkpoint(result)
    return result


def validate_checkpoint(document):
    required = {
        "format",
        "dataset",
        "shape",
        "parameter_words",
        "completed_steps",
        "seed",
        "learning_rate",
        "batch_size",
    }
    if not isinstance(document, dict) or not required <= document.keys():
        raise ValueError("incomplete Snackwell training checkpoint")
    if document["format"] != FORMAT or document["dataset"] != DATASET:
        raise ValueError("expected a Snackwell GELU-network SGD checkpoint")
    if not isinstance(document["shape"], dict) or set(document["shape"]) != {
        "inputs",
        "hidden",
        "outputs",
    }:
        raise ValueError("checkpoint shape requires inputs, hidden and outputs")
    shape = Shape(**document["shape"])
    if shape != Shape() or document.get("batch_size") != 16:
        raise ValueError("the regression demo uses shape 8->16->4 and batch size 16")
    if type(document["completed_steps"]) is not int or document["completed_steps"] < 0:
        raise ValueError("invalid completed step count")
    if type(document["seed"]) is not int:
        raise ValueError("seed must be an integer")
    validate_parameters(document["parameter_words"], shape)
    learning_rate_word(document["learning_rate"])
    return shape


def load_checkpoint(path):
    document = json.loads(Path(path).read_text())
    validate_checkpoint(document)
    return document
