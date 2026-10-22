"""Deterministic text windows, initialization and resumable RTL training state."""

import hashlib
import math
import random
from pathlib import Path

from snackwell.character_model import FORMAT_V2, parameter_shapes, validate_checkpoint


def initialize(config, seed):
    rng = random.Random(seed)
    weights = {}
    for name, shape in parameter_shapes(
        config["context"],
        config["width"],
        len(config["vocabulary"]),
        config["layers"],
        config.get("ffn_multiplier", 2),
    ).items():
        if "gamma" in name:
            data = [1.0] * math.prod(shape)
        elif len(shape) == 1:
            data = [0.0] * math.prod(shape)
        else:
            scale = 0.1 if "embedding" in name else 1 / math.sqrt(shape[0])
            data = [rng.gauss(0, scale) for _ in range(math.prod(shape))]
        weights[name] = data
    return validate_checkpoint(dict(format=FORMAT_V2, config=config, weights=weights))


class Corpus:
    def __init__(self, path, context, vocabulary=None, tokenizer=None):
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        train_end, valid_end = int(len(text) * 0.9), int(len(text) * 0.95)
        self.vocabulary = sorted(set(text[:train_end])) if vocabulary is None else vocabulary
        if tokenizer is not None:
            from snackwell.tokenizer import Tokenizer

            codec = Tokenizer.restore(tokenizer)
            if self.vocabulary != codec.vocabulary:
                raise ValueError("training vocabulary does not match tokenizer")
            encode = codec.encode
        elif not 2 <= len(self.vocabulary) <= 128 or any(len(t) != 1 for t in self.vocabulary):
            raise ValueError("training requires 2..128 distinct characters")
        else:
            mapping = {ch: i for i, ch in enumerate(self.vocabulary)}
            unknown = set(text) - mapping.keys()
            if unknown:
                raise ValueError(
                    f"corpus contains characters outside the training vocabulary: {sorted(unknown)!r}"
                )

            def encode(text):
                return [mapping[ch] for ch in text]

        self.train = encode(text[:train_end])
        self.validation = encode(text[train_end:valid_end])
        self.test = encode(text[valid_end:])
        if min(map(len, (self.train, self.validation, self.test))) <= context:
            raise ValueError("each corpus split must contain more tokens than the context")
        self.context, self.sha256 = context, hashlib.sha256(raw).hexdigest()

    def window(self, split, seed, index):
        data = getattr(self, split)
        start = random.Random(f"{split}:{seed}:{index}").randrange(len(data) - self.context)
        return data[start : start + self.context], data[start + 1 : start + self.context + 1]


def checkpoint(model, corpus, seed, rate, completed_steps):
    return dict(
        format=model["format"],
        config=model["config"],
        weights=model["weights"],
        **({"tokenizer": model["tokenizer"]} if "tokenizer" in model else {}),
        training=dict(
            kind="snackwell-rtl-transformer-sgd-v1",
            dataset_sha256=corpus.sha256,
            seed=seed,
            learning_rate=rate,
            completed_steps=completed_steps,
            batch_sequences=1,
            loss="mean next-token cross entropy",
        ),
    )


def resume_state(model, corpus):
    state = model.get("training", {})
    if state.get("kind") != "snackwell-rtl-transformer-sgd-v1":
        raise ValueError(
            "--resume needs an RTL transformer-training checkpoint; use --checkpoint for initial weights"
        )
    if state.get("dataset_sha256") != corpus.sha256:
        raise ValueError("resume corpus does not match the checkpoint")
    if (
        type(state.get("completed_steps")) is not int
        or state["completed_steps"] < 0
        or type(state.get("seed")) is not int
    ):
        raise ValueError("invalid training step or seed")
    return state
