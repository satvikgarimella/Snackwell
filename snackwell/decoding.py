"""Prompt boundaries and reproducible inputs for token selection on the SM."""

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


def validate_min_p(min_p):
    if type(min_p) not in (int, float) or not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError("min-p must be finite and between 0 and 1")
    return min_p


def prepare_prompt(model, prompt, *, cache=True, heal=True):
    """Reconsider an extendable final BPE token without changing the user's text.

    For example, a trailing space may be the start of a token such as ' king'.
    Back up one token and require its replacement to start with the same text.
    A one-token prompt has no preceding state to predict from, so leave it alone.
    """
    from snackwell.character_model import encode_prompt

    ids = encode_prompt(model, prompt, pad=not cache)
    if not heal or "tokenizer" not in model:
        return ids, ""
    unpadded = encode_prompt(model, prompt, pad=False)
    if len(unpadded) > 1:
        vocabulary = model["config"]["vocabulary"]
        prefix = vocabulary[unpadded[-1]]
        if any(token.startswith(prefix) and len(token) > len(prefix) for token in vocabulary):
            return encode_prompt(model, prompt[: -len(prefix)], pad=not cache), prefix
    return ids, ""


class Decoder:
    def __init__(self, mode="sample", seed=None, temperature=DEFAULT_TEMPERATURE, *, min_p=0.0):
        if mode not in ("sample", "greedy"):
            raise ValueError("decoding mode must be sample or greedy")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an unsigned 32-bit integer")
        self.mode = mode
        self.temperature = validate_temperature(temperature)
        self.min_p = validate_min_p(min_p)
        self.seed = random.SystemRandom().getrandbits(32) if seed is None else seed
        self.random = random.Random(self.seed)

    def draw(self):
        # Exactly representable in FP32, always below 1 even after rounding.
        return self.random.getrandbits(24) / 2**24 if self.mode == "sample" else None

    def describe(self):
        result = dict(
            mode=self.mode,
            seed=self.seed if self.mode == "sample" else None,
            temperature=self.temperature if self.mode == "sample" else None,
        )
        if self.min_p:
            result["min_p"] = self.min_p if self.mode == "sample" else None
        return result
