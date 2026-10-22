"""Preview model quality on the CPU. This does not execute or verify Snackwell RTL."""

import argparse

from snackwell.character_model import load_checkpoint, model_description
from snackwell.decoding import DEFAULT_TEMPERATURE, Decoder
from snackwell.model_preview import ModelPreview
from snackwell.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=str, default=str(ROOT / "models/character_transformer/checkpoint.json")
    )
    parser.add_argument("--prompt", default="My lord, I have something to say")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--decode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--no-kv-cache", action="store_true", help="Match the legacy RTL sliding-window policy"
    )
    args = parser.parse_args()
    if not 1 <= args.tokens <= 256:
        parser.error("--tokens must be 1..256")
    model = load_checkpoint(args.checkpoint)
    Decoder(args.decode, args.seed, args.temperature)
    preview = ModelPreview(model)
    print("CPU reference preview — no RTL execution or hardware verification.")
    print(f"Model: {model_description(model)}")
    print("Context policy: " + ("legacy sliding" if args.no_kv_cache else "periodic rebuild"))
    while True:
        try:
            prompt = input("CPU prompt> ") if args.interactive else args.prompt
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip() == "/quit":
            return
        decoder = Decoder(args.decode, args.seed, args.temperature)
        try:
            if args.interactive:
                updated = load_checkpoint(args.checkpoint)
                if updated["sha256"] != model["sha256"]:
                    model = updated
                    preview = ModelPreview(model)
                    print(f"Reloaded model: {model_description(model)}")
            print(f"Decoding: {decoder.describe()}")
            print(preview.generate(prompt, args.tokens, decoder, cache=not args.no_kv_cache))
        except ValueError as error:
            if not args.interactive:
                parser.exit(1, f"Preview failed: {error}\n")
            print(f"Preview failed: {error}")
        if not args.interactive:
            return


if __name__ == "__main__":
    main()
