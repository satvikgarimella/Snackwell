"""Train the full character transformer with verified forward/backward/SGD on RTL."""

import argparse
import hashlib
import json
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.numeric import value
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.training import learning_rate_word, write_json
from snackwell.transformer_training import run_step
from snackwell.transformer_training_data import Corpus, checkpoint, initialize, resume_state
from snackwell.transformer_training_runtime import TransformerTrainingRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "build/data/tinyshakespeare.txt")
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--checkpoint", type=Path, help="start from portable model weights")
    origin.add_argument(
        "--resume", type=Path, help="continue RTL SGD with the same corpus and batches"
    )
    parser.add_argument("--steps", type=int, default=100, help="additional sequence updates")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--width", type=int, choices=(4, 8, 16, 32, 64, 128))
    parser.add_argument("--context", type=int, choices=(4, 8, 16, 32, 64, 128))
    parser.add_argument("--layers", type=int, choices=(1, 2, 4))
    parser.add_argument("--heads", type=int, choices=(1, 2, 4))
    parser.add_argument("--evaluation-windows", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--simulator", choices=("auto", "icarus", "verilator"), default="auto")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.log_every, args.evaluation_windows) < 1:
        parser.error("steps, log interval and evaluation windows must be positive")
    try:
        model = (
            load_checkpoint(args.resume or args.checkpoint)
            if args.resume or args.checkpoint
            else None
        )
        config = dict(context=16, width=16, layers=1, heads=2, epsilon=1e-5)
        if model:
            config = dict(model["config"])
        for key in ("context", "width", "layers", "heads"):
            requested = getattr(args, key)
            if requested is not None:
                if model and requested != config.get(key, 1):
                    raise ValueError(f"--{key} cannot change an imported model")
                config[key] = requested
        corpus = Corpus(
            args.data,
            config["context"],
            config.get("vocabulary"),
            model.get("tokenizer") if model else None,
        )
        state = resume_state(model, corpus) if args.resume else None
        seed = args.seed if args.seed is not None else state["seed"] if state else 1729
        rate = (
            args.learning_rate
            if args.learning_rate is not None
            else (state["learning_rate"] if state else 0.001 if model else 0.05)
        )
        rate = value(learning_rate_word(rate))
        if state and (seed != state["seed"] or rate != state["learning_rate"]):
            raise ValueError("resume keeps the saved seed and learning rate")
        config["vocabulary"] = corpus.vocabulary
        model = model or initialize(config, seed)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    completed = state["completed_steps"] if state else 0
    out = args.out or ROOT / "build/transformer-training" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    )
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--out must be a new directory; use --resume to continue a checkpoint")
    path = out / "checkpoint.json"
    initial = checkpoint(model, corpus, seed, rate, completed)
    write_json(out / "initial_checkpoint.json", initial)
    write_json(path, initial)
    params = sum(map(len, model["weights"].values()))
    print("Snackwell — full transformer training in RTL simulation", flush=True)
    print(
        f"Starting checkpoint: {(args.resume or args.checkpoint).resolve()}"
        if args.resume or args.checkpoint
        else "Starting a small model from random weights; this is the RTL training demonstration."
    )
    print(
        f"{config.get('layers', 1)} blocks, {config.get('heads', 1)} heads, width {config['width']}, "
        f"context {config['context']}; {params:,} parameters; SGD {rate:.6g}."
    )
    print("Every forward pass, loss, parameter gradient and update is independently checked.")
    print(f"Run files: {out.resolve()}", flush=True)
    began = time.monotonic()
    totals = dict(cycles=0, launches=0)
    history = []
    try:
        simulator = compile_model_runner(out, args.simulator, args.trace)
        write_json(
            out / "run.json",
            dict(
                data=str(args.data.resolve()),
                dataset_sha256=corpus.sha256,
                seed=seed,
                learning_rate=rate,
                initial_step=completed,
                steps=args.steps,
                evaluation_windows=args.evaluation_windows,
                simulator=simulator.backend,
                trace=simulator.trace,
                rtl_sha256=simulator.source_sha256,
                python_source_sha256={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((ROOT / "snackwell").rglob("*.py"))
                },
            ),
        )
        context = (
            SimulatorSession(simulator)
            if simulator.backend == "verilator"
            else nullcontext(simulator)
        )
        with context as execution:

            def runtime(name):
                return TransformerTrainingRuntime(out / name, execution)

            def count(result):
                for key in totals:
                    totals[key] += result[key]

            def evaluate(label, split):
                losses = []
                for index in range(args.evaluation_windows):
                    ids, targets = corpus.window(split, seed, index)
                    result, _ = run_step(
                        runtime(f"{label}/window_{index:03d}"), model, ids, targets
                    )
                    count(result)
                    losses.extend(result["tensors"]["observed"]["loss"])
                aggregate = runtime(label + "/mean")
                words = [w for loss in losses for w in (loss, 0, 0, 0)]
                total = aggregate.column_sum("sum", words, len(losses), 4)
                mean = aggregate.arithmetic("mean", "scale", total, scale=1 / len(losses))[0]
                count(
                    dict(
                        cycles=sum(s["cycles"] for s in aggregate.reports),
                        launches=len(aggregate.reports),
                    )
                )
                return value(mean)

            initial_train = evaluate("initial_train", "train")
            initial_validation = evaluate("initial_validation", "validation")
            print(
                f"Start: train loss {initial_train:.6f} | validation {initial_validation:.6f}",
                flush=True,
            )
            for offset in range(args.steps):
                ids, targets = corpus.window("train", seed, completed)
                result, candidate = run_step(
                    runtime(f"step_{completed + 1:06d}"), model, ids, targets, rate
                )
                write_json(path, checkpoint(candidate, corpus, seed, rate, completed + 1))
                model, completed = candidate, completed + 1
                count(result)
                item = {k: result[k] for k in ("loss", "cycles", "launches", "seconds")}
                item["step"] = completed
                history.append(item)
                with (out / "history.jsonl").open("a") as log:
                    log.write(json.dumps(item) + "\n")
                if offset == 0 or (offset + 1) % args.log_every == 0 or offset + 1 == args.steps:
                    print(
                        f"Step {completed:4d}: loss {result['loss']:.6f} | {result['cycles']:,} cycles | "
                        f"{result['seconds']:.2f}s | all gradients + updates verified",
                        flush=True,
                    )
            final_train = evaluate("final_train", "train")
            final_validation = evaluate("final_validation", "validation")
        result = dict(
            verified=True,
            scope="full transformer forward/backward/SGD",
            parameters=params,
            completed_steps=completed,
            steps_this_run=args.steps,
            initial_train_loss=initial_train,
            final_train_loss=final_train,
            initial_validation_loss=initial_validation,
            final_validation_loss=final_validation,
            validation_improved=final_validation < initial_validation,
            inference_checkpoint_promoted=False,
            evaluation_targets_per_split=args.evaluation_windows * config["context"],
            **totals,
            seconds=time.monotonic() - began,
            history=history,
        )
        write_json(out / "result.json", result)
        print(f"\nTrain loss:      {initial_train:.6f} -> {final_train:.6f}")
        print(f"Validation loss: {initial_validation:.6f} -> {final_validation:.6f}")
        if final_validation >= initial_validation:
            print(
                "Validation loss did not improve. The checkpoint is saved for resume; it is not promoted for inference."
            )
        else:
            print(
                f"Validation improved on {args.evaluation_windows * config['context']} sampled targets; generation quality needs a separate check."
            )
        print(
            f"Verified {args.steps} full training steps | {totals['cycles']:,} cycles | {result['seconds']:.1f}s"
        )
        print(f"Checkpoint: {path.resolve()}")
        print(
            f"Run inference: python3 {ROOT / 'scripts/run_model.py'} --checkpoint {path.resolve()} --interactive --tokens 4"
        )
    except (KeyboardInterrupt, Exception) as error:
        committed = load_checkpoint(path)["training"]["completed_steps"]
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(
            out / "failure.json", dict(status=status, completed_steps=committed, error=str(error))
        )
        print(f"\nTraining {status}. Last committed checkpoint: {path.resolve()}", file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            raise SystemExit(130) from error
        raise


if __name__ == "__main__":
    main()
