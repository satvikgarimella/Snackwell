"""Run a trained text transformer and generation on the SM RTL."""

import argparse
import json
import subprocess
from pathlib import Path

from snackwell.character_model import load_checkpoint, model_description
from snackwell.decoding import DEFAULT_TEMPERATURE, Decoder
from snackwell.kv_cache import Generation, latency_report
from snackwell.model_runner import evaluate
from snackwell.paths import ROOT
from snackwell.simulator import compile_model_runner

DEFAULT_CHECKPOINT = ROOT / "models/character_transformer/checkpoint.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="To be, or not to ")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Use the original padded, sliding-window recomputation policy",
    )
    parser.add_argument(
        "--decode",
        choices=("sample", "greedy"),
        default="sample",
        help="Generation selection: sample probabilities (default) or take the maximum",
    )
    parser.add_argument("--seed", type=int, help="Repeat sampled generation with this 32-bit seed")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature, 0.1..2.0; lower values favor likely tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--simulator",
        choices=("auto", "icarus", "verilator"),
        default="auto",
        help="RTL simulator; auto uses Verilator when installed, otherwise Icarus",
    )
    parser.add_argument(
        "--trace", action="store_true", help="Save and summarize every simulated cycle"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter prompts and watch predictions in the terminal",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Show live terminal progress for --prompt"
    )
    parser.add_argument("--live", action="store_true", help="Open the live prediction browser view")
    parser.add_argument(
        "--port", type=int, default=8768, help="Local port for --live (default: 8768)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Print the live URL without opening it"
    )
    parser.add_argument(
        "--evaluate", type=Path, help="Held-out examples exported with the checkpoint"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "build/inference/character_transformer")
    args = parser.parse_args()
    if sum(bool(mode) for mode in (args.live, args.evaluate, args.interactive, args.stream)) > 1:
        parser.error("Choose one of --live, --evaluate, --interactive, or --stream")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.live and not args.interactive:
        (args.out / "result.json").unlink(missing_ok=True)
    try:
        if args.tokens < 1 or args.tokens > 256:
            raise ValueError("--tokens must be 1..256")
        model = load_checkpoint(args.checkpoint)
        if args.live and "tokenizer" in model:
            raise ValueError(
                "subword checkpoints use --interactive or --stream; the browser labels characters"
            )
        print(f"Model: {model_description(model)}", flush=True)
        print(f"Checkpoint file: {args.checkpoint.resolve()}", flush=True)
        decoder = Decoder(args.decode, args.seed, args.temperature)
        print("Preparing RTL simulator…", flush=True)
        executable = compile_model_runner(args.out, args.simulator, args.trace)
        print(
            f"RTL simulator: {executable.backend}; cycle trace: {'on' if args.trace else 'off'}",
            flush=True,
        )
        print(f"RTL build: {executable.source_sha256[:12]}", flush=True)
        if args.live:
            from snackwell.live_model import serve

            serve(
                model,
                args.out / "live",
                args.prompt,
                args.tokens,
                args.port,
                not args.no_browser,
                executable,
                args.decode,
                args.seed,
                checkpoint_path=args.checkpoint,
                temperature=args.temperature,
                cache=not args.no_kv_cache,
            )
            return
        if args.interactive:
            from snackwell.terminal_model import interactive

            interactive(
                model,
                args.out / "terminal",
                executable,
                args.tokens,
                args.decode,
                args.seed,
                checkpoint_path=args.checkpoint,
                temperature=args.temperature,
                cache=not args.no_kv_cache,
            )
            return
        if args.stream:
            from snackwell.terminal_model import predict

            predict(
                model,
                args.prompt,
                args.tokens,
                args.out,
                executable,
                decoder=decoder,
                cache=not args.no_kv_cache,
            )
            print(f"Results: {args.out.resolve()}")
            return
        if args.evaluate:
            report = evaluate(model, args.evaluate, args.out, executable)
            print(
                f"PASS {report['evaluated_tokens']} held-out tokens: loss={report['loss']:.4f}, "
                f"unigram={report['unigram_loss']:.4f}, perplexity={report['perplexity']:.3f}"
            )
        else:
            print(f"Decoding: {decoder.describe()}", flush=True)
            generated, reports = args.prompt, []
            generation = Generation(model, args.prompt, cache=not args.no_kv_cache)
            for step in range(args.tokens):
                result = generation.next(
                    args.out / f"token_{step:03d}",
                    executable,
                    draw=decoder.draw(),
                    temperature=decoder.temperature,
                )
                generated += result["next_character"]
                reports.append(
                    dict(
                        character=result["next_character"],
                        cycles=result["cycles"],
                        launches=result["launches"],
                        accuracy=result["accuracy"],
                        wall_seconds=result["wall_seconds"],
                        simulation=result["simulation"],
                        selection=result["selection"],
                        cache=result["cache"],
                    )
                )
                print(
                    f"PASS token {step + 1}/{args.tokens}: {result['next_character']!r}, "
                    f"{result['cycles']} cycles, {result['wall_seconds']:.2f}s "
                    f"({result['cache']['phase']})",
                    flush=True,
                )
            report = dict(
                verified=True,
                checkpoint_sha256=model["sha256"],
                prompt=args.prompt,
                generated=generated,
                decoding=decoder.describe(),
                tokens=reports,
                cycles=sum(r["cycles"] for r in reports),
                latency=latency_report(reports, not args.no_kv_cache, subword="tokenizer" in model),
                tokenization="subword" if "tokenizer" in model else "character",
            )
            print(generated)
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Results: {args.out.resolve()}")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        parser.exit(1, f"Inference failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Inference cancelled.\n")


if __name__ == "__main__":
    main()
