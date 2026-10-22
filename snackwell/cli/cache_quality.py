#!/usr/bin/env python3
"""Compare context policies on held-out text using the optional CPU reference.

This measures language-model quality, not RTL correctness or hardware speed.
Both policies use the exact same checkpoint, target characters and corpus split.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8201)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/character_transformer/checkpoint.json"
    )
    parser.add_argument("--out", type=str, default="build/kv-cache/quality.json")
    args = parser.parse_args()
    if args.sequences < 1 or args.steps < 1:
        parser.error("sequences and steps must be positive")

    import torch
    import torch.nn.functional as F

    from snackwell.torch_model import ReferenceTransformer

    torch.set_num_threads(1)
    model = load_checkpoint(args.checkpoint)
    network = ReferenceTransformer(model["config"], model["weights"]).eval()
    raw = (ROOT / "build/data/tinyshakespeare.txt").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != model["training"]["dataset_sha256"]:
        raise ValueError("dataset differs from checkpoint training data")
    text = raw.decode()[model["training"]["split_boundaries"]["validation_end"] :]
    mapping = {c: i for i, c in enumerate(model["config"]["vocabulary"])}
    data = torch.tensor([mapping[c] for c in text])
    context = model["config"]["context"]
    retained = 3 * context // 4
    starts = torch.randint(
        context,
        len(data) - args.steps,
        (args.sequences,),
        generator=torch.Generator().manual_seed(args.seed),
    )
    metrics = {p: dict(loss=0.0, correct=0) for p in ("sliding", "cached_policy")}
    disagreement = 0
    counts = {}
    length = context
    with torch.no_grad():
        for step in range(args.steps):
            end = starts + step
            target = data[end]
            winners = []
            counts[length] = counts.get(length, 0) + args.sequences
            for policy, n in (("sliding", context), ("cached_policy", length)):
                inputs = data[end[:, None] + torch.arange(-n, 0)]
                logits = network(inputs)[:, -1]
                metrics[policy]["loss"] += float(F.cross_entropy(logits, target, reduction="sum"))
                winner = logits.argmax(-1)
                metrics[policy]["correct"] += int((winner == target).sum())
                winners.append(winner)
            disagreement += int((winners[0] != winners[1]).sum())
            length = retained if length == context else length + 1
    count = args.sequences * args.steps
    for entry in metrics.values():
        entry["loss"] /= count
        entry["perplexity"] = math.exp(entry["loss"])
        entry["accuracy"] = entry.pop("correct") / count
    result = dict(
        backend="PyTorch CPU reference; not RTL inference",
        checkpoint_sha256=model["sha256"],
        dataset_sha256=digest,
        split="held-out final 5%",
        seed=args.seed,
        tokens=count,
        sequences=args.sequences,
        steps=args.steps,
        context_lengths=counts,
        metrics=metrics,
        greedy_disagreement=disagreement / count,
        loss_delta=metrics["cached_policy"]["loss"] - metrics["sliding"]["loss"],
    )
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
