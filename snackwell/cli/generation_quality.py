#!/usr/bin/env python3
"""Inspect fixed-prompt CPU-reference continuations, not RTL execution."""

import argparse
import json
import re
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.decoding import Decoder
from snackwell.model_preview import ModelPreview

PROMPTS = (
    "My lord, I have something to say",
    "The king is dead, and we must now",
    "the sky today is",
    "My name is ",
    "ROMEO:",
    "Good night,",
    "I love",
    "To be, or not to ",
)


def repetition(text):
    grams = [text[i : i + 4] for i in range(max(0, len(text) - 3))]
    fraction = 1 - len(set(grams)) / len(grams) if grams else 0
    triples = sum(
        any(
            text[i : i + period] * 3 == text[i : i + period * 3]
            for period in range(1, min(12, (len(text) - i) // 3) + 1)
        )
        for i in range(len(text))
    )
    return dict(repeated_fourgram_fraction=fraction, positions_starting_three_repeats=triples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--backend", choices=("scalar", "torch"), default="scalar")
    parser.add_argument("--data", type=Path, help="Training corpus for completed-word coverage")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--no-kv-cache", action="store_true")
    args = parser.parse_args()
    if not 4 <= args.tokens <= 256:
        parser.error("--tokens must be 4..256")
    model = load_checkpoint(args.checkpoint)
    preview = ModelPreview(model, args.backend)
    Decoder(temperature=args.temperature)
    known_words = None
    if args.data:
        text = args.data.read_text()
        known_words = set(re.findall(r"[A-Za-z]+", text[: int(len(text) * 0.9)].lower()))
    cases = []
    for mode in ("greedy", "sample"):
        for seed in args.seeds if mode == "sample" else [None]:
            for prompt in args.prompts:
                decoder = Decoder(mode, seed, args.temperature)
                generated = preview.generate(
                    prompt, args.tokens, decoder, cache=not args.no_kv_cache
                )
                continuation = generated[len(prompt) :]
                # Drop boundary words: the prompt or truncation can split them.
                words = re.findall(r"[A-Za-z]+", continuation.lower())[1:-1]
                cases.append(
                    dict(
                        mode=mode,
                        temperature=args.temperature if mode == "sample" else None,
                        seed=seed,
                        prompt=prompt,
                        continuation=continuation,
                        **repetition(continuation),
                        completed_words=len(words),
                        words_seen_in_training=sum(word in known_words for word in words)
                        if known_words is not None
                        else None,
                    )
                )
                print(f"{mode} seed={seed}: {prompt!r} -> {continuation!r}", flush=True)
    summary = {}
    for mode in ("greedy", "sample"):
        subset = [case for case in cases if case["mode"] == mode]
        summary[mode] = dict(
            cases=len(subset),
            mean_repeated_fourgram_fraction=sum(c["repeated_fourgram_fraction"] for c in subset)
            / len(subset),
            cases_with_three_repeats=sum(c["positions_starting_three_repeats"] > 0 for c in subset),
            completed_words=sum(c["completed_words"] for c in subset),
            words_seen_in_training=sum(c["words_seen_in_training"] for c in subset)
            if known_words is not None
            else None,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                kind="CPU reference generation diagnostic; not RTL execution",
                checkpoint_sha256=model["sha256"],
                backend=args.backend,
                context_policy="legacy_sliding" if args.no_kv_cache else "periodic_rebuild",
                word_coverage_note="Corpus word coverage is a spelling diagnostic, not a fluency score",
                tokens=args.tokens,
                temperature=args.temperature,
                summary=summary,
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
