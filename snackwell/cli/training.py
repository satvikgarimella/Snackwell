"""Train a two-layer GELU network with forward, backward and SGD on Snackwell RTL."""

import argparse
import hashlib
import json
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from snackwell.numeric import value
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.training import (
    Shape,
    batch,
    checkpoint,
    dataset,
    evaluate,
    initialize,
    learning_rate_word,
    load_checkpoint,
    train_step,
    write_json,
)
from snackwell.training_runtime import TrainingRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="additional SGD updates")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", type=Path, help="continue a saved RTL-training checkpoint")
    parser.add_argument("--simulator", choices=("auto", "verilator", "icarus"), default="auto")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", type=Path, help="new run directory (must not already exist)")
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1:
        parser.error("--steps and --log-every must be positive")
    shape = Shape()
    try:
        previous = load_checkpoint(args.resume) if args.resume else None
        seed = args.seed if args.seed is not None else previous["seed"] if previous else 1729
        rate = (
            args.learning_rate
            if args.learning_rate is not None
            else (previous["learning_rate"] if previous else 0.2)
        )
        rate = value(learning_rate_word(rate))
        if previous and (
            seed != previous["seed"] or rate != value(learning_rate_word(previous["learning_rate"]))
        ):
            parser.error("resume keeps the checkpoint's seed and learning rate")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    output = args.out or ROOT / "build/training" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--out must be a new directory; use --resume to continue weights")
    completed = previous["completed_steps"] if previous else 0
    parameters = previous["parameter_words"] if previous else initialize(shape, seed)
    train, heldout = dataset(seed)
    write_json(output / "data.json", dict(train=train, heldout=heldout))
    write_json(
        output / "initial_checkpoint.json", checkpoint(parameters, shape, completed, seed, rate)
    )
    checkpoint_path = output / "checkpoint.json"
    write_json(checkpoint_path, checkpoint(parameters, shape, completed, seed, rate))
    print("Snackwell — training a GELU network in RTL simulation", flush=True)
    print("8 inputs -> 16 hidden units -> 4 outputs; 212 parameters; SGD, batch size 16.")
    print("Task: learn four nonlinear functions from 64 examples; 32 held-out examples.")
    print("Loss: mean squared error / 2. Every forward, gradient and update is checked.")
    print(f"Run files: {output.resolve()}", flush=True)
    began = time.monotonic()
    history, costs = [], []
    try:
        simulator = compile_model_runner(output, args.simulator, args.trace)
        write_json(
            output / "run.json",
            dict(
                steps=args.steps,
                initial_step=completed,
                seed=seed,
                learning_rate=rate,
                simulator=simulator.backend,
                trace=simulator.trace,
                rtl_sha256=simulator.source_sha256,
                python_source_sha256={
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((ROOT / "snackwell").rglob("*.py"))
                },
                resume=str(args.resume.resolve()) if args.resume else None,
            ),
        )
        context = (
            SimulatorSession(simulator)
            if simulator.backend == "verilator"
            else nullcontext(simulator)
        )
        with context as execution:

            def runtime(name):
                return TrainingRuntime(output / name, execution)

            initial_train = evaluate(runtime("initial_train"), parameters, *train, shape)
            initial_heldout = evaluate(runtime("initial_heldout"), parameters, *heldout, shape)
            costs.extend((initial_train, initial_heldout))
            print(
                f"Start: train loss {initial_train['loss']:.6f} | held-out {initial_heldout['loss']:.6f}",
                flush=True,
            )
            for offset in range(args.steps):
                x, target = batch(train, seed, completed)
                result = train_step(
                    runtime(f"step_{completed + 1:05d}"), parameters, x, target, shape, rate
                )
                candidate = result["tensors"]["updated"]
                # The checkpoint is the commit point. Failed/cancelled steps leave it intact.
                write_json(checkpoint_path, checkpoint(candidate, shape, completed + 1, seed, rate))
                parameters, completed = candidate, completed + 1
                costs.append(result)
                item = {k: result[k] for k in ("loss", "cycles", "launches", "seconds")}
                item["step"] = completed
                history.append(item)
                with (output / "history.jsonl").open("a") as log:
                    log.write(json.dumps(item) + "\n")
                if offset == 0 or (offset + 1) % args.log_every == 0 or offset + 1 == args.steps:
                    print(
                        f"Step {completed:4d}: batch loss {result['loss']:.6f} | "
                        f"{result['cycles']:,} cycles | gradients + updates verified",
                        flush=True,
                    )
            final_train = evaluate(runtime("final_train"), parameters, *train, shape)
            final_heldout = evaluate(runtime("final_heldout"), parameters, *heldout, shape)
            costs.extend((final_train, final_heldout))
        result = dict(
            verified=True,
            scope="two-layer GELU network; nonlinear regression",
            completed_steps=completed,
            steps_this_run=args.steps,
            history=history,
            initial_train_loss=initial_train["loss"],
            final_train_loss=final_train["loss"],
            initial_heldout_loss=initial_heldout["loss"],
            final_heldout_loss=final_heldout["loss"],
            cycles=sum(r["cycles"] for r in costs),
            launches=sum(r["launches"] for r in costs),
            seconds=time.monotonic() - began,
        )
        write_json(output / "result.json", result)
        print(f"\nTrain loss:    {initial_train['loss']:.6f} -> {final_train['loss']:.6f}")
        print(f"Held-out loss: {initial_heldout['loss']:.6f} -> {final_heldout['loss']:.6f}")
        print("Held-out predictions (first output of four):")
        for row in range(4):
            i = row * shape.outputs
            print(
                f"  target {value(heldout[1][i]): .4f} | before "
                f"{value(initial_heldout['tensors']['y'][i]): .4f} | after "
                f"{value(final_heldout['tensors']['y'][i]): .4f}"
            )
        print(
            f"Verified {args.steps} training steps | {result['cycles']:,} cycles | {result['seconds']:.1f}s"
        )
        print(f"Checkpoint: {checkpoint_path.resolve()}")
    except (KeyboardInterrupt, Exception) as error:
        status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        # A signal can arrive just after atomic replacement, before the local counter advances.
        committed = load_checkpoint(checkpoint_path)["completed_steps"]
        write_json(
            output / "failure.json",
            dict(status=status, completed_steps=committed, error=str(error)),
        )
        print(
            f"\nTraining {status}. Last committed checkpoint: {checkpoint_path.resolve()}",
            file=sys.stderr,
        )
        if isinstance(error, KeyboardInterrupt):
            raise SystemExit(130) from error
        raise


if __name__ == "__main__":
    main()
