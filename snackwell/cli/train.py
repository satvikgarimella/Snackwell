#!/usr/bin/env python3
"""Train/export reference weights on the host; this is NOT Snackwell training."""

import argparse
import hashlib
import json
import math
import random
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from snackwell.character_model import (
    FORMAT_SUBWORD,
    FORMAT_V2,
    load_checkpoint,
    reference_forward,
    validate_checkpoint,
)
from snackwell.paths import ROOT

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def training_objective(logits, targets, tail, prefix_weight):
    """Keep the full causal input; optionally emphasize its later predictions."""
    import torch.nn.functional as F

    errors = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape(
        targets.shape
    )
    return prefix_weight * errors.mean() + (1 - prefix_weight) * errors[:, -tail:].mean()


def score_targets(network, data, positions, context, device):
    """Score one next character per fixed position, with no split-boundary crossing."""
    import torch
    import torch.nn.functional as F

    if context < 1 or not positions or any(p < context or p >= len(data) for p in positions):
        raise ValueError("evaluation targets and their contexts must stay inside the split")
    network.eval()
    total, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            indexes = torch.tensor(positions[start : start + 64])
            ids = data[indexes[:, None] - torch.arange(context, 0, -1)].to(device)
            targets = data[indexes].to(device)
            logits = network(ids)[:, -1]
            total += float(F.cross_entropy(logits, targets, reduction="sum"))
            correct += int((logits.argmax(-1) == targets).sum())
    loss = total / len(positions)
    return dict(
        loss=loss,
        perplexity=math.exp(loss),
        accuracy=correct / len(positions),
        tokens=len(positions),
        target_positions=positions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "build/data/tinyshakespeare.txt")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--width", type=int, choices=(4, 8, 16, 32, 64, 128), default=16)
    parser.add_argument("--context", type=int, choices=(4, 8, 16, 32, 64, 128), default=16)
    parser.add_argument(
        "--bpe-vocabulary",
        type=int,
        choices=(128, 256, 512),
        help="Learn subword tokens on the training split; default: characters",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loss-tail", type=int, help="Emphasize the last N targets; default: all")
    parser.add_argument(
        "--prefix-loss-weight",
        type=float,
        default=0.2,
        help="Weight on all-position loss when --loss-tail is set",
    )
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--evaluation-targets", type=int, default=8192)
    parser.add_argument(
        "--report-test",
        action="store_true",
        help="Also report reserved test loss after validation selects weights",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--extend-context",
        action="store_true",
        help="Initialize a longer position table by repeating the resume checkpoint's positions",
    )
    parser.add_argument("--layers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--heads", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--ffn-multiplier", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--expand-ffn",
        action="store_true",
        help="Grow a resumed model's feed-forward layers with zero initial output",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--resume", type=Path, help="Initialize from exported weights (fresh optimizer)"
    )
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.0, help="Training-only dropout")
    parser.add_argument("--initialization", choices=("fan_in", "gpt"), default="fan_in")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--teacher", type=Path, help="CPU teacher checkpoint for reference distillation"
    )
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=1.5)
    parser.add_argument("--min-learning-rate", type=float, help="Final rate for cosine decay")
    parser.add_argument(
        "--out", type=Path, help="New experiment directory; never overwrite a model"
    )
    args = parser.parse_args()
    if args.width // args.heads < 4:
        parser.error("each attention head requires at least four features")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.loss_tail is None:
        args.loss_tail = args.context
    if not 1 <= args.loss_tail <= args.context:
        parser.error("--loss-tail must be between 1 and --context")
    if not 0 <= args.prefix_loss_weight <= 1:
        parser.error("--prefix-loss-weight must be in [0, 1]")
    if min(args.eval_every, args.evaluation_targets) < 1:
        parser.error("evaluation interval and target count must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight-decay must be finite and nonnegative")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("--warmup-steps must be nonnegative and smaller than --steps")
    if not 0 < args.distillation_weight <= 1:
        parser.error("--distillation-weight must be in (0, 1]")
    if not math.isfinite(args.distillation_temperature) or args.distillation_temperature <= 0:
        parser.error("--distillation-temperature must be finite and positive")
    if args.extend_context and not args.resume:
        parser.error("--extend-context requires --resume")
    if args.expand_ffn and not args.resume:
        parser.error("--expand-ffn requires --resume")
    if args.bpe_vocabulary and args.resume:
        parser.error("--resume uses the saved tokenizer; omit --bpe-vocabulary")
    if args.ffn_multiplier * args.width > 512:
        parser.error("feed-forward width cannot exceed 512")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.min_learning_rate is not None and not (
        math.isfinite(args.min_learning_rate) and 0 < args.min_learning_rate <= args.learning_rate
    ):
        parser.error("--min-learning-rate must be positive and no larger than --learning-rate")
    if args.download and not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            args.data.write_bytes(response.read())
    raw = args.data.read_bytes()
    text = raw.decode("utf-8")
    train_end, validation_end = int(len(text) * 0.9), int(len(text) * 0.95)
    resumed = load_checkpoint(args.resume) if args.resume else None
    tokenizer = None
    if args.bpe_vocabulary or (resumed and "tokenizer" in resumed):
        from snackwell.tokenizer import Tokenizer

        if resumed:
            if "tokenizer" not in resumed:
                parser.error("cannot change a character checkpoint to BPE when resuming")
            tokenizer = Tokenizer.restore(resumed["tokenizer"])
        else:
            tokenizer = Tokenizer.train(text[:train_end], args.bpe_vocabulary)
        vocabulary = tokenizer.vocabulary
        encode = tokenizer.encode
    else:
        vocabulary = sorted(set(text[:train_end]))
        if set(text) - set(vocabulary) or len(vocabulary) > 128:
            raise ValueError("unknown held-out characters or vocabulary exceeds 128 characters")
        mapping = {ch: i for i, ch in enumerate(vocabulary)}

        def encode(text):
            return [mapping[ch] for ch in text]

    splits = [
        encode(part)
        for part in (text[:train_end], text[train_end:validation_end], text[validation_end:])
    ]
    if min(map(len, splits)) <= args.context:
        parser.error("each split must contain more tokens than --context")

    import torch
    import torch.nn.functional as F

    from snackwell.torch_model import ReferenceTransformer

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requires an available Apple GPU")
    device = torch.device(args.device)
    n, d, v = args.context, args.width, len(vocabulary)
    if args.expand_ffn:
        from snackwell.model_expansion import expand_feedforward

        resumed = expand_feedforward(resumed, args.ffn_multiplier, args.seed)
    elif resumed and resumed["config"].get("ffn_multiplier", 2) != args.ffn_multiplier:
        parser.error("feed-forward changes require --expand-ffn")
    if (
        resumed
        and resumed.get("training", {}).get("dataset_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        parser.error("resume checkpoint must use the same corpus")
    if resumed and any(
        resumed["config"][key] != expected
        for key, expected in (("width", d), ("vocabulary", vocabulary))
    ):
        raise ValueError("resume checkpoint must match width and vocabulary")
    if (
        resumed
        and resumed["config"]["context"] != n
        and not (args.extend_context and n > resumed["config"]["context"])
    ):
        raise ValueError("context changes require --extend-context and a larger context")
    config = dict(
        context=n,
        width=d,
        epsilon=1e-5,
        vocabulary=vocabulary,
        layers=args.layers,
        heads=args.heads,
    )
    if args.ffn_multiplier != 2:
        config["ffn_multiplier"] = args.ffn_multiplier
    if resumed and any(resumed["config"].get(key, 1) != config[key] for key in ("layers", "heads")):
        raise ValueError("resume checkpoint must match layers and heads")
    weights = dict(resumed["weights"]) if resumed else None
    if weights is not None and n != resumed["config"]["context"]:
        positions = weights["position_embedding"]
        weights["position_embedding"] = [positions[i % len(positions)] for i in range(n * d)]
    network = ReferenceTransformer(
        config, weights, dropout=args.dropout, initialization=args.initialization
    ).to(device)
    teacher_model = load_checkpoint(args.teacher) if args.teacher else None
    teacher = None
    if teacher_model:
        if (
            teacher_model["config"]["vocabulary"] != vocabulary
            or teacher_model["config"]["context"] < n
            or teacher_model.get("tokenizer") != (tokenizer.metadata() if tokenizer else None)
        ):
            raise ValueError(
                "teacher must use the same vocabulary and support the training context"
            )
        if (
            teacher_model.get("training", {}).get("dataset_sha256")
            != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("teacher must use the same training corpus")
        teacher = (
            ReferenceTransformer(teacher_model["config"], teacher_model["weights"])
            .to(device)
            .eval()
        )
        teacher.requires_grad_(False)
    parameters = network.parameters_by_name
    forward = network
    train, validation, test = [torch.tensor(part, dtype=torch.long) for part in splits]
    args.out = args.out or ROOT / "build/reference-training" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    )
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--out must be a new directory; existing models are never overwritten")
    print(f"Host reference training only; output: {args.out.resolve()}", flush=True)
    (args.out / "run.json").write_text(
        json.dumps(
            dict(
                backend=f"PyTorch {args.device.upper()} reference training; not RTL",
                arguments={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                dataset_sha256=hashlib.sha256(raw).hexdigest(),
                input_checkpoint_sha256=resumed["sha256"] if resumed else None,
                torch_version=torch.__version__,
                source_sha256={
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__),
                        ROOT / "snackwell/torch_model.py",
                        ROOT / "snackwell/model_expansion.py",
                        ROOT / "snackwell/character_model.py",
                        ROOT / "snackwell/tokenizer.py",
                    )
                },
            ),
            indent=2,
        )
        + "\n"
    )

    def batch(data, count, generator):
        starts = torch.randint(len(data) - n, (count,), generator=generator)
        indexes = starts[:, None] + torch.arange(n)
        return data[indexes].to(device), data[indexes + 1].to(device)

    def score(data, seed):
        positions = sorted(
            random.Random(seed).sample(
                range(n, len(data)), min(args.evaluation_targets, len(data) - n)
            )
        )
        return score_targets(network, data, positions, n, device)

    generator = torch.Generator().manual_seed(args.seed + 1)
    initial = score(validation, args.seed + 2)
    print(f"Initial validation loss: {initial['loss']:.4f}", flush=True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in parameters.values() if p.ndim >= 2],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for p in parameters.values() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    history = []
    best_loss = initial["loss"]
    best = {k: p.detach().clone() for k, p in parameters.items()}
    best_step = 0
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        rate = args.learning_rate
        if step <= args.warmup_steps:
            rate *= step / args.warmup_steps
        elif args.min_learning_rate is not None:
            progress = (step - args.warmup_steps - 1) / max(1, args.steps - args.warmup_steps - 1)
            rate = (
                args.min_learning_rate
                + (args.learning_rate - args.min_learning_rate)
                * (1 + math.cos(math.pi * progress))
                / 2
            )
        for group in optimizer.param_groups:
            group["lr"] = rate
        network.train()
        x, y = batch(train, args.batch_size, generator)
        logits = forward(x)
        loss = training_objective(logits, y, args.loss_tail, args.prefix_loss_weight)
        if teacher is not None:
            temperature = args.distillation_temperature
            with torch.no_grad():
                soft_targets = F.softmax(teacher(x) / temperature, dim=-1)
            distillation_errors = (
                F.kl_div(
                    F.log_softmax(logits / temperature, dim=-1), soft_targets, reduction="none"
                ).sum(-1)
                * temperature**2
            )
            distillation = (
                args.prefix_loss_weight * distillation_errors.mean()
                + (1 - args.prefix_loss_weight) * distillation_errors[:, -args.loss_tail :].mean()
            )
            loss = (1 - args.distillation_weight) * loss + args.distillation_weight * distillation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            val = score(validation, args.seed + 2)
            history.append(
                dict(
                    step=step,
                    train_batch_loss=float(loss.detach()),
                    validation=val,
                    elapsed_seconds=time.monotonic() - started,
                )
            )
            print(
                f"step {step}: train={float(loss.detach()):.4f}, val={val['loss']:.4f}", flush=True
            )
            (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if val["loss"] < best_loss:
                best_loss, best_step = val["loss"], step
                best = {k: p.detach().clone() for k, p in parameters.items()}
    with torch.no_grad():
        for name in parameters:
            parameters[name].copy_(best[name])
    # Export and the independent math check always use portable CPU FP32.
    network.to("cpu")
    device = torch.device("cpu")
    test_metrics = score(test, args.seed + 3) if args.report_test else None
    counts = torch.bincount(train, minlength=v).double() + 1
    unigram = (counts / counts.sum()).tolist()
    baseline = (
        -sum(math.log(unigram[int(i)]) for i in test) / len(test) if args.report_test else None
    )
    model = validate_checkpoint(
        dict(
            format=FORMAT_SUBWORD if tokenizer else FORMAT_V2,
            config=config,
            **({"tokenizer": tokenizer.metadata()} if tokenizer else {}),
            weights={name: p.detach().reshape(-1).tolist() for name, p in parameters.items()},
            training=dict(
                backend=f"PyTorch {args.device.upper()} (reference training, not Snackwell)",
                torch_version=torch.__version__,
                seed=args.seed,
                steps=args.steps,
                selected_step=best_step,
                batch_size=args.batch_size,
                feedforward_expansion="new random input weights with zero output weights"
                if args.expand_ffn
                else None,
                loss_tail=args.loss_tail,
                prefix_loss_weight=args.prefix_loss_weight,
                evaluation="one target per fixed position with full context; validation selects weights",
                evaluation_targets=args.evaluation_targets,
                dropout=args.dropout,
                initialization=args.initialization if not resumed else "resume checkpoint",
                weight_decay=args.weight_decay,
                decay_parameters="matrices only; biases and normalization parameters excluded",
                warmup_steps=args.warmup_steps,
                teacher_checkpoint_sha256=teacher_model["sha256"] if teacher_model else None,
                distillation_weight=args.distillation_weight if teacher_model else None,
                distillation_temperature=args.distillation_temperature if teacher_model else None,
                position_initialization="repeat pretrained positions"
                if args.extend_context
                else None,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                resume_checkpoint_sha256=resumed["sha256"] if resumed else None,
                resumed_training=resumed.get("training") if resumed else None,
                optimizer_state="fresh AdamW",
                dataset_url=DATA_URL,
                dataset_sha256=hashlib.sha256(raw).hexdigest(),
                characters=len(text),
                split_boundaries=dict(train_end=train_end, validation_end=validation_end),
                initial_validation=initial,
                history=history,
                test=test_metrics,
                test_unigram_loss=baseline,
            ),
        )
    )
    checkpoint = args.out / "checkpoint.json"
    checkpoint.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rng = random.Random(args.seed + 4)
    examples = []
    export_data = test if args.report_test else validation
    export_offset = validation_end if args.report_test else train_end
    with torch.no_grad():
        for offset in sorted(rng.sample(range(len(export_data) - n), min(8, len(export_data) - n))):
            ids = export_data[offset : offset + n].tolist()
            targets = export_data[offset + 1 : offset + n + 1].tolist()
            probabilities = F.softmax(forward(torch.tensor([ids])), dim=-1)[0].reshape(-1).tolist()
            ref = reference_forward(model, ids)
            if max(abs(a - b) for a, b in zip(probabilities, ref["probabilities"])) > 5e-5:
                raise ValueError("independent reference does not match PyTorch export")
            examples.append(
                dict(
                    dataset_offset=export_offset
                    + (
                        sum(len(vocabulary[i]) for i in export_data[:offset].tolist())
                        if tokenizer
                        else offset
                    ),
                    split_token_offset=offset,
                    input_ids=ids,
                    target_ids=targets,
                    torch_probabilities=probabilities,
                )
            )
    (args.out / "evaluation.json").write_text(
        json.dumps(
            dict(
                checkpoint_sha256=checksum,
                dataset_sha256=hashlib.sha256(raw).hexdigest(),
                split="held-out final 5%" if args.report_test else "validation 90%-95%",
                unigram_probabilities=unigram,
                examples=examples,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Selected step {best_step}; full-context validation loss={best_loss:.4f}", flush=True)
    if best_step == 0:
        print(
            "Validation did not improve; exported the starting weights, not a trained replacement.",
            flush=True,
        )
    if test_metrics:
        print(f"Reserved test loss={test_metrics['loss']:.4f}; unigram={baseline:.4f}", flush=True)
    print(
        f"Exported {sum(p.numel() for p in parameters.values())} parameters to {checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
