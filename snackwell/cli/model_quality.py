"""Compare CPU models at identical held-out next-character positions."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from snackwell.character_model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F

    from snackwell.torch_model import ReferenceTransformer

    torch.set_num_threads(1)
    models = [load_checkpoint(path) for path in args.checkpoints]
    if any("tokenizer" in model for model in models):
        parser.error(
            "subword models require scripts/compare_text_models.py; character target IDs are not comparable"
        )
    raw = args.data.read_bytes()
    text = raw.decode("utf-8")
    split_start = int(len(text) * (0.9 if args.split == "validation" else 0.95))
    split_end = int(len(text) * 0.95) if args.split == "validation" else len(text)
    max_context = max(model["config"]["context"] for model in models)
    positions = sorted(
        random.Random(args.seed).sample(range(split_start + max_context, split_end), args.targets)
    )
    results = []
    for path, model in zip(args.checkpoints, models):
        if model.get("training", {}).get("dataset_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("checkpoint training corpus differs from evaluation corpus")
        mapping = {ch: i for i, ch in enumerate(model["config"]["vocabulary"])}
        encoded = [mapping[ch] for ch in text]
        n = model["config"]["context"]
        network = ReferenceTransformer(model["config"], model["weights"]).eval()
        loss, correct = 0.0, 0
        target_losses = []
        target_correct = []
        with torch.no_grad():
            for start in range(0, len(positions), 64):
                batch = positions[start : start + 64]
                ids = torch.tensor([encoded[p - n : p] for p in batch])
                targets = torch.tensor([encoded[p] for p in batch])
                logits = network(ids)[:, -1]
                errors = F.cross_entropy(logits, targets, reduction="none")
                matches = logits.argmax(-1) == targets
                target_losses.extend(errors.tolist())
                target_correct.extend(matches.tolist())
                loss += float(errors.sum())
                correct += int(matches.sum())
        row = dict(
            checkpoint=str(path),
            checkpoint_sha256=model["sha256"],
            config=model["config"],
            parameters=sum(map(len, model["weights"].values())),
            loss=loss / len(positions),
            perplexity=math.exp(loss / len(positions)),
            correct=correct,
            targets=len(positions),
            accuracy=correct / len(positions),
            target_losses=target_losses,
            target_correct=target_correct,
        )
        results.append(row)
        print(f"{path}: loss={row['loss']:.4f}, accuracy={row['accuracy']:.2%}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                kind="CPU reference comparison, not RTL execution",
                dataset_sha256=hashlib.sha256(raw).hexdigest(),
                seed=args.seed,
                split=f"{args.split}; identical targets, each model uses its own context",
                target_positions=positions,
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
