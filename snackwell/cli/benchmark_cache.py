"""Pair cached RTL generation with full RTL recomputation of identical contexts."""

import argparse
import json
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.decoding import DEFAULT_TEMPERATURE, Decoder
from snackwell.kv_cache import Generation, latency_report
from snackwell.model_runner import forward
from snackwell.paths import ROOT
from snackwell.simulator import compile_model_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="My lord, I have something to say")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=92617660)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/character_transformer/checkpoint.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "build/kv-cache/paired")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 256:
        parser.error("tokens must be 1..256")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").unlink(missing_ok=True)
    model = load_checkpoint(args.checkpoint)
    executable = compile_model_runner(args.out)
    generation = Generation(model, args.prompt)
    decoder = Decoder(seed=args.seed, temperature=DEFAULT_TEMPERATURE)
    pairs, cached_reports = [], []
    for step in range(args.tokens):
        draw = decoder.draw()
        cached = generation.next(
            args.out / f"cached_{step:03d}", executable, draw=draw, temperature=decoder.temperature
        )
        full = forward(
            model,
            cached["input_ids"],
            args.out / f"full_{step:03d}",
            executable,
            draw=draw,
            temperature=decoder.temperature,
            last_only=True,
        )
        vocab = len(model["config"]["vocabulary"])
        if (
            cached["probabilities"][-vocab:] != full["probabilities"][-vocab:]
            or cached["sampling_probabilities"] != full["sampling_probabilities"]
            or cached["selection"] != full["selection"]
        ):
            raise ValueError(f"token {step}: cached output differs from uncached RTL")
        pair = dict(
            step=step,
            phase=cached["cache"]["phase"],
            context_tokens=len(cached["input_ids"]),
            character=cached["next_character"],
            bit_identical=True,
            cached={k: cached[k] for k in ("wall_seconds", "cycles", "launches")},
            full={k: full[k] for k in ("wall_seconds", "cycles", "launches")},
        )
        pairs.append(pair)
        cached_reports.append(cached)
        print(
            f"{step + 1}/{args.tokens} {pair['phase']}: "
            f"{cached['wall_seconds']:.2f}s vs {full['wall_seconds']:.2f}s; "
            f"{cached['cycles']:,} vs {full['cycles']:,} cycles; bit-identical",
            flush=True,
        )
    report = dict(
        verified=True,
        checkpoint_sha256=model["sha256"],
        rtl_source_sha256=executable.source_sha256,
        simulator=executable.backend,
        trace=executable.trace,
        compilation_included=False,
        prompt=args.prompt,
        generated=generation.text,
        decoding=decoder.describe(),
        comparison=f"full recomputation under identical {generation.rebuild_tokens}..{generation.context} context policy, not legacy sliding",
        latency=latency_report(cached_reports, True, subword="tokenizer" in model),
        pairs=pairs,
        totals={
            mode: {
                key: sum(p[mode][key] for p in pairs)
                for key in ("wall_seconds", "cycles", "launches")
            }
            for mode in ("cached", "full")
        },
    )
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(generation.text)
    print(json.dumps(report["latency"], indent=2))


if __name__ == "__main__":
    main()
