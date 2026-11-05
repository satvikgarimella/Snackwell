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
    """Count exact repeats and expose the pattern, including benign cases such as III."""
    grams = [text[i : i + 4] for i in range(max(0, len(text) - 3))]
    fraction = 1 - len(set(grams)) / len(grams) if grams else 0
    triples, longest = 0, ""
    for i in range(len(text)):
        for period in range((len(text) - i) // 3, 0, -1):
            pattern = text[i : i + period]
            if pattern * 3 == text[i : i + period * 3]:
                triples += 1
                if len(pattern) > len(longest):
                    longest = pattern
                break
    return dict(
        repeated_fourgram_fraction=fraction,
        positions_starting_three_repeats=triples,
        longest_tripled_substring=longest,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--tokens", type=int, help="Generate this many tokens (default: 32)")
    length.add_argument(
        "--characters",
        type=int,
        help="Generate at least this many characters, finishing the last token; compare tokenizers fairly",
    )
    parser.add_argument("--backend", choices=("scalar", "torch"), default="scalar")
    parser.add_argument("--data", type=Path, help="Training corpus for completed-word coverage")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--no-token-healing", action="store_true")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    parser.add_argument("--no-kv-cache", action="store_true")
    args = parser.parse_args()
    if args.characters is not None and not 4 <= args.characters <= 256:
        parser.error("--characters must be 4..256")
    if args.tokens is None:
        args.tokens = 256 if args.characters is not None else 32
    if not 4 <= args.tokens <= 256:
        parser.error("--tokens must be 4..256")
    model = load_checkpoint(args.checkpoint)
    preview = ModelPreview(model, args.backend)
    Decoder(temperature=args.temperature, min_p=args.min_p)
    known_words = None
    if args.data:
        text = args.data.read_text()
        known_words = set(re.findall(r"[A-Za-z]+", text[: int(len(text) * 0.9)].lower()))
    cases = []
    for mode in ("greedy", "sample"):
        for seed in args.seeds if mode == "sample" else [None]:
            for prompt in args.prompts:
                decoder = Decoder(mode, seed, args.temperature, min_p=args.min_p)
                generated = preview.generate(
                    prompt,
                    args.tokens,
                    decoder,
                    cache=not args.no_kv_cache,
                    characters=args.characters,
                    heal=not args.no_token_healing,
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
                        generated_characters=len(continuation),
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
                tokens=args.tokens if args.characters is None else None,
                minimum_characters=args.characters,
                max_tokens=args.tokens,
                temperature=args.temperature,
                min_p=args.min_p,
                token_healing=not args.no_token_healing,
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
