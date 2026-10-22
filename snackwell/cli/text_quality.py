"""Compare canonical text likelihood on identical held-out passages using the CPU."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from snackwell.character_model import encode_prompt, load_checkpoint
from snackwell.model_preview import ModelPreview
from snackwell.tokenizer import Tokenizer


def text_spans(text, count, seed):
    starts = [i for i in range(512, len(text) - 1024) if text[i - 1] == "\n"]
    random.Random(seed).shuffle(starts)
    spans = []
    for start in starts:
        end = text.find("\n", start + 384) + 1
        if end <= start or any(start < b and end > a for a, b in spans):
            continue
        spans.append((start, end))
        if len(spans) == count:
            return sorted(spans)
    raise ValueError("not enough nonoverlapping newline-bounded passages in this split")


def score_text(model, prefix, target):
    """Teacher-force each target token; keep the same cache-window policy as RTL.

    The tokenization of a passage is one possible encoding of its text. This
    measures that canonical sequence's probability, not marginal probability
    over every tokenization and not character classification accuracy.
    """
    import torch
    import torch.nn.functional as F

    preview = ModelPreview(model)
    context = model["config"]["context"]
    ids = encode_prompt(model, prefix, pad=False)
    if len(ids) < context:
        raise ValueError("passage prefix is too short for this model's full context")
    if "tokenizer" in model:
        targets = Tokenizer.restore(model["tokenizer"]).encode(target)
    else:
        mapping = {c: i for i, c in enumerate(model["config"]["vocabulary"])}
        targets = [mapping[c] for c in target]
    if "".join(model["config"]["vocabulary"][i] for i in targets) != target:
        raise ValueError("target tokenization does not reconstruct the scored text")
    losses = []
    with torch.no_grad():
        for token in targets:
            if len(ids) > context:
                ids = ids[-max(1, 3 * context // 4) :]
            logits = preview.network(torch.tensor([ids]))[0, -1].double()
            losses.append(float(-F.log_softmax(logits, dim=-1)[token]))
            ids.append(token)
    return dict(nats=math.fsum(losses), tokens=len(targets), characters=len(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--spans", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2741)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.spans < 1:
        parser.error("--spans must be positive")
    raw = args.data.read_bytes()
    text = raw.decode()
    digest = hashlib.sha256(raw).hexdigest()
    start = int(len(text) * (0.9 if args.split == "validation" else 0.95))
    end = int(len(text) * 0.95) if args.split == "validation" else len(text)
    split = text[start:end]
    spans = text_spans(split, args.spans, args.seed)
    models = [load_checkpoint(path) for path in args.checkpoints]
    if any(m.get("training", {}).get("dataset_sha256") != digest for m in models):
        parser.error("checkpoint training corpus differs from evaluation corpus")
    scores = []
    for path, model in zip(args.checkpoints, models):
        results = [score_text(model, split[a - 512 : a], split[a:b]) for a, b in spans]
        nats, characters = (
            math.fsum(r["nats"] for r in results),
            sum(r["characters"] for r in results),
        )
        row = dict(
            checkpoint=str(path.resolve()),
            checkpoint_sha256=model["sha256"],
            spans=results,
            nats=nats,
            characters=characters,
            nats_per_character=nats / characters,
            bits_per_character=nats / characters / math.log(2),
        )
        scores.append(row)
        print(
            f"{model['sha256'][:12]}: {nats / characters:.6f} nats/character ({characters} held-out characters)",
            flush=True,
        )
    report = dict(
        kind="CPU canonical-tokenization NLL per character; not RTL or per-character accuracy",
        dataset_sha256=digest,
        split=args.split,
        seed=args.seed,
        context_policy="each model's context; periodic rebuild to three quarters",
        spans=[dict(start=start + a, end=start + b) for a, b in spans],
        results=scores,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
