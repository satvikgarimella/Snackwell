#!/usr/bin/env python3
"""Run the compact location-answering transformer in terminal RTL simulation."""

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snackwell.character_model import load_checkpoint, model_description
from snackwell.paths import ROOT
from snackwell.sentence_model import complete
from snackwell.simulator import compile_model_runner

EXAMPLE = "The ball is in the kitchen. The book is in the garden. Where is the ball?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/sentence_completion/checkpoint.json"
    )
    parser.add_argument("--prompt", default=EXAMPLE)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "build/inference/sentence_completion")
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint)
    if (
        model.get("training", {}).get("task")
        != "two-fact location question; short generated answer"
    ):
        parser.error("--checkpoint must be a sentence-task checkpoint from train_sentence_model.py")
    print("Snackwell — compact sentence demo (two facts and a location question)")
    print("Objects: ball, book, cup, hat, key, toy, bag, box.")
    print("Places: kitchen, garden, bedroom, office, hallway, bathroom, attic, cellar.")
    print(f"Example: {EXAMPLE}")
    print(
        "Use two different objects in two different places. This model was trained for this format."
    )
    print("Every predicted token executes in RTL and is independently verified.")
    print(model_description(model), flush=True)
    executable = compile_model_runner(args.out / "simulator", compact=True)
    print(f"RTL: {executable.backend}; source {executable.source_sha256[:12]}", flush=True)
    while True:
        try:
            prompt = (input("\nPrompt> ") if args.interactive else args.prompt).strip()
            if prompt.strip() == "/quit":
                break
            if not prompt.strip():
                continue
            output = args.out / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
            print("Answer:", end="", flush=True)
            result = complete(
                model,
                prompt,
                output,
                executable,
                on_token=lambda text: print(text, end="", flush=True),
            )
            latency = result["latency"]
            print(
                f"\n{result['cycles']:,} cycles | {result['launches']} launches | RTL/reference match"
            )
            print(
                f"First token: {latency['first_token_seconds']:.2f}s; "
                f"total model time: {sum(t['wall_seconds'] for t in result['tokens']):.2f}s"
            )
            if not result["complete"]:
                print("Token limit reached before a complete sentence.")
            print(f"Run files: {output}")
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break
        except ValueError as error:
            print(f"\n{error}")
            if not args.interactive:
                raise SystemExit(1) from error
        if not args.interactive:
            break


if __name__ == "__main__":
    main()
