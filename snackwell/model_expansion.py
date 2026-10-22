"""Add trainable feed-forward units without changing the starting model function."""

import math
import random

from snackwell.character_model import (
    FORMAT_V2,
    block_prefix,
    feedforward_width,
    validate_checkpoint,
)


def expand_feedforward(model, multiplier, seed):
    model = validate_checkpoint(model)
    config = dict(model["config"])
    width, previous = config["width"], feedforward_width(config)
    if type(multiplier) is not int or multiplier not in (2, 4):
        raise ValueError("feed-forward multiplier must be 2 or 4")
    hidden = multiplier * width
    if not previous < hidden <= 512:
        raise ValueError("expansion must increase feed-forward width, to at most 512")
    config.update(
        ffn_multiplier=multiplier, layers=config.get("layers", 1), heads=config.get("heads", 1)
    )
    weights = {name: list(data) for name, data in model["weights"].items()}
    rng = random.Random(seed)
    for layer in range(config["layers"]):
        prefix = block_prefix(layer, config["layers"])
        old = weights[prefix + "w1"]
        weights[prefix + "w1"] = [
            value
            for row in range(width)
            for value in old[row * previous : (row + 1) * previous]
            + [rng.gauss(0, 1 / math.sqrt(width)) for _ in range(hidden - previous)]
        ]
        weights[prefix + "b1"] += [0.0] * (hidden - previous)
        # New units initially contribute zero. Their output weights can learn
        # immediately; gradients then reach their independently initialized inputs.
        weights[prefix + "w2"] += [0.0] * ((hidden - previous) * width)
    return validate_checkpoint(
        dict(
            model,
            format=model["format"] if "tokenizer" in model else FORMAT_V2,
            config=config,
            weights=weights,
        )
    )
