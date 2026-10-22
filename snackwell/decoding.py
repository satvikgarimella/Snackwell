"""Reproducible random inputs for character selection on the SM."""

import math
import random

DEFAULT_TEMPERATURE = 0.7


def validate_temperature(temperature):
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0.1 <= temperature <= 2.0
    ):
        raise ValueError("temperature must be finite and between 0.1 and 2.0")
    return temperature


class Decoder:
    def __init__(self, mode="sample", seed=None, temperature=DEFAULT_TEMPERATURE):
        if mode not in ("sample", "greedy"):
            raise ValueError("decoding mode must be sample or greedy")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an unsigned 32-bit integer")
        self.mode = mode
        self.temperature = validate_temperature(temperature)
        self.seed = random.SystemRandom().getrandbits(32) if seed is None else seed
        self.random = random.Random(self.seed)

    def draw(self):
        # Exactly representable in FP32, always below 1 even after rounding.
        return self.random.getrandbits(24) / 2**24 if self.mode == "sample" else None

    def describe(self):
        return dict(
            mode=self.mode,
            seed=self.seed if self.mode == "sample" else None,
            temperature=self.temperature if self.mode == "sample" else None,
        )
