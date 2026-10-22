"""Per-prediction raw RTL K/V storage and bounded-context generation policy.

Learned absolute positions cannot be shifted under existing cached activations.
At overflow we rebuild the newest three quarters of the window at positions
starting from zero. All intervening steps append one token at a fixed position.
"""

import hashlib
import json
import struct
from dataclasses import dataclass

from snackwell.character_model import ReferenceState, encode_prompt


def model_signature(model):
    # Recheck actual values on every append, including in-place edits. Packing
    # doubles preserves Python numeric values without formatting each weight
    # as decimal text. The checkpoint's file SHA is not a substitute for this.
    config = json.dumps(model["config"], sort_keys=True).encode()
    digest = hashlib.sha256(b"snackwell-kv-model-v2\0")
    digest.update(struct.pack("<I", len(config)))
    digest.update(config)
    if "tokenizer" in model:
        digest.update(json.dumps(model["tokenizer"], sort_keys=True).encode())
    for name, values in sorted(model["weights"].items()):
        key = name.encode()
        digest.update(struct.pack("<I", len(key)))
        digest.update(key)
        digest.update(struct.pack("<I", len(values)))
        digest.update(struct.pack(f"<{len(values)}d", *values))
    return digest.hexdigest()


@dataclass
class KVCache:
    # Only forward() commits these fields, after every numerical check passes.
    ids: tuple = ()
    layers: tuple = ()
    signature: str | None = None
    rtl_source_sha256: str | None = None
    reference: ReferenceState | None = None


class Generation:
    """One prompt owns one cache. Discard the object on cancellation or completion."""

    def __init__(self, model, prompt, *, cache=True):
        self.model = model
        self.enabled = cache
        self.context = model["config"]["context"]
        self.rebuild_tokens = max(1, 3 * self.context // 4)
        self.text = prompt
        self.ids = encode_prompt(model, prompt, pad=not cache)
        self.cache = KVCache() if cache else None
        self.steps = 0

    def next(self, output, executable, on_progress=None, *, draw=None, temperature=1.0):
        from snackwell.model_runner import forward

        ids, cache = self.ids, self.cache
        phase = "prefill" if self.steps == 0 else "decode" if cache else "recompute"
        if len(ids) > self.context:
            ids = ids[-self.rebuild_tokens :]
            cache = KVCache()
            phase = "rebuild"
        if on_progress:
            on_progress(dict(type="cache", phase=phase, context_tokens=len(ids)))
        result = forward(
            self.model,
            ids,
            output,
            executable,
            on_progress,
            draw=draw,
            last_only=True,
            temperature=temperature,
            kv_cache=cache,
            phase=phase,
        )
        self.text += result["next_character"]
        self.cache = cache
        self.ids = (
            ids + [result["selection"]["token_id"]]
            if self.enabled
            else (ids + [result["selection"]["token_id"]])[-self.context :]
        )
        self.steps += 1
        return result


def latency_report(reports, enabled, *, subword=False):
    def average(items):
        return sum(items) / len(items) if items else None

    decode = [r["wall_seconds"] for r in reports if r["cache"]["phase"] == "decode"]
    rebuild = [r["wall_seconds"] for r in reports if r["cache"]["phase"] == "rebuild"]
    total = sum(r["wall_seconds"] for r in reports)
    characters = sum(len(r.get("character", r.get("next_character", ""))) for r in reports)
    result = dict(
        generated_characters=characters,
        generated_tokens=len(reports),
        tokens_per_second=len(reports) / total if total else None,
        characters_per_second=characters / total if total else None,
        first_token_seconds=reports[0]["wall_seconds"],
        subsequent_token_seconds=average([r["wall_seconds"] for r in reports[1:]]),
        cached_decode=enabled,
        decode_seconds=average(decode),
        decode_steps=len(decode),
        rebuild_seconds=average(rebuild),
        rebuild_steps=len(rebuild),
    )
    if not subword:
        result.update(
            first_character_seconds=result["first_token_seconds"],
            subsequent_character_seconds=result["subsequent_token_seconds"],
        )
    return result
