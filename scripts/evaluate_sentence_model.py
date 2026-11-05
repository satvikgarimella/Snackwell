#!/usr/bin/env python3
"""Audit the bounded sentence model: autonomous host generation or verified RTL."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snackwell.character_model import load_checkpoint, reference_forward
from snackwell.sentence_task import LOCATIONS, OBJECTS, Example, dataset_hash, examples
from snackwell.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rtl", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer.restore(model["tokenizer"])
    if args.rtl:
        from snackwell.model_runner import forward
        from snackwell.sentence_model import complete
        from snackwell.simulator import compile_model_runner

        # All eight held-out query pairs, both positions, varied distractors.
        data = examples("test")
        selected = []
        for obj in OBJECTS:
            for first in (True, False):
                selected.append(
                    next(e for e in data if e.query == obj and (e.facts[0][0] == obj) == first)
                )
        # Change only the two locations. The answer must change with them.
        for rooms in (("kitchen", "garden"), ("garden", "kitchen")):
            selected.append(Example(tuple(zip(("ball", "book"), rooms)), "ball"))
        simulator = compile_model_runner(args.out / "simulator", compact=True)
        cases = []
        for index, example in enumerate(selected):
            result = complete(model, example.prompt, args.out / f"case_{index:02d}", simulator)
            case = dict(
                prompt=example.prompt,
                expected=example.answer,
                answer=result["answer"],
                exact=result["answer"] == example.answer,
                cycles=result["cycles"],
                launches=result["launches"],
                latency=result["latency"],
                verified=result["verified"],
            )
            cases.append(case)
            print(json.dumps(case), flush=True)
        # Compare cached appends to a fresh full-prefix computation, including
        # the selected token and the full final-row probability distribution.
        first = selected[0]
        report_path = args.out / "case_00/result.json"
        cached = json.loads(report_path.read_text())
        ids = tokenizer.encode(first.prompt) + [t["token_id"] for t in cached["tokens"][:-1]]
        fresh = forward(model, ids, args.out / "fresh", simulator, last_only=True)
        cached_last = json.loads(
            (args.out / f"case_00/token_{len(cached['tokens']) - 1:03d}/result.json").read_text()
        )
        if fresh["selection"]["token_id"] != cached["tokens"][-1]["token_id"]:
            raise ValueError("cached and fresh token selections differ")
        if fresh["probabilities"] != cached_last["probabilities"]:
            raise ValueError("cached and fresh final probabilities differ")
        report = dict(
            execution="Verified Snackwell RTL",
            checkpoint_sha256=model["sha256"],
            rtl_source_sha256=simulator.source_sha256,
            cases=cases,
            exact_answers=sum(c["exact"] for c in cases),
            total=len(cases),
            cached_fresh_match=True,
        )
    else:
        import torch

        from snackwell.torch_model import ReferenceTransformer

        torch.set_num_threads(1)
        network = ReferenceTransformer(model["config"], model["weights"]).eval()

        def generate(data):
            cases = []
            # Group by prompt length; no left padding or leaked answer tokens.
            groups = {}
            for example in data:
                ids = tokenizer.encode(example.prompt)
                groups.setdefault(len(ids), []).append((example, ids))
            with torch.inference_mode():
                for group in groups.values():
                    for start in range(0, len(group), 64):
                        batch = group[start : start + 64]
                        x = torch.tensor([ids for _, ids in batch])
                        generated = [[] for _ in batch]
                        ended = [False] * len(batch)
                        for _ in range(12):
                            if x.shape[1] > model["config"]["context"]:
                                break
                            choices = network(x)[:, -1].argmax(-1)
                            for row, token in enumerate(choices.tolist()):
                                if not ended[row]:
                                    generated[row].append(token)
                                    ended[row] = "." in tokenizer.vocabulary[token]
                            if all(ended):
                                break
                            x = torch.cat((x, choices[:, None]), dim=1)
                        for (example, _), ids in zip(batch, generated):
                            answer = tokenizer.decode(ids)
                            cases.append(
                                dict(
                                    prompt=example.prompt,
                                    expected=example.answer,
                                    answer=answer,
                                    exact=answer == example.answer,
                                    generated_ids=ids,
                                )
                            )
            return cases

        report = dict(
            execution="HOST autonomous greedy evaluation; no RTL inference",
            checkpoint_sha256=model["sha256"],
            splits={},
        )
        for split in ("validation", "test"):
            data = examples(split)
            cases = generate(data)
            report["splits"][split] = dict(
                exact_answers=sum(c["exact"] for c in cases),
                total=len(cases),
                dataset_sha256=dataset_hash(data),
                cases=cases,
            )
            print(split, report["splits"][split]["exact_answers"], "/", len(cases), flush=True)
        # Check every possible two-object/two-location prompt in this grammar,
        # including cases where BOTH pairs were held out of training.
        exhaustive = []
        for obj in OBJECTS:
            for room in LOCATIONS:
                for other in OBJECTS:
                    for elsewhere in LOCATIONS:
                        if obj != other and room != elsewhere:
                            facts = ((obj, room), (other, elsewhere))
                            exhaustive.extend((Example(facts, obj), Example(facts[::-1], obj)))
        cases = generate(exhaustive)
        report["exhaustive"] = dict(
            exact_answers=sum(c["exact"] for c in cases),
            total=len(cases),
            dataset_sha256=dataset_hash(exhaustive),
            cases=cases,
        )
        print("exhaustive", report["exhaustive"]["exact_answers"], "/", len(cases), flush=True)
        export_checks = []
        for example in examples("validation")[::144]:
            ids = tokenizer.encode(example.prompt)
            with torch.no_grad():
                actual = network(torch.tensor([ids]))[0, -1].tolist()
            expected = reference_forward(model, ids)["logits"][-len(tokenizer.vocabulary) :]
            error = max(abs(a - b) for a, b in zip(actual, expected))
            if error > 2e-4:
                raise ValueError(f"independent scalar/export mismatch: {error}")
            export_checks.append(dict(prompt=example.prompt, maximum_logit_error=error))
        report["export_checks"] = export_checks
    report["evaluation_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
