"""Paired model evaluation: hardware correctness and quantization quality are separate."""

import argparse
import json
import math
import random
from pathlib import Path

from snackwell.character_model import encode_prompt, load_checkpoint
from snackwell.kv_cache import Generation, KVCache
from snackwell.model_runner import forward
from snackwell.nvfp4_inference import POLICY
from snackwell.paths import ROOT
from snackwell.sentence_model import complete
from snackwell.sentence_task import examples
from snackwell.simulator import compile_model_runner
from snackwell.tokenizer import Tokenizer


def distribution_error(original, quantized):
    # RTL approximations can sum slightly above/below one. Normalize both for KL/TV.
    p = [x / math.fsum(original) for x in original]
    q = [x / math.fsum(quantized) for x in quantized]
    return dict(
        total_variation=math.fsum(abs(a - b) for a, b in zip(p, q, strict=True)) / 2,
        kl_fp32_to_nvfp4=math.fsum(
            a * math.log(a / max(b, 1e-300)) for a, b in zip(p, q, strict=True) if a > 0
        ),
    )


def compare_case(model, prompt, answer, output, executable, tokens):
    output.mkdir(parents=True, exist_ok=True)
    ids = (
        Tokenizer.restore(model["tokenizer"]).encode(prompt)
        if "tokenizer" in model
        else encode_prompt(model, prompt, pad=False)
    )
    targets = None
    if answer is not None:
        full = Tokenizer.restore(model["tokenizer"]).encode(prompt + answer)
        if full[: len(ids)] != ids:
            raise ValueError("evaluation tokenizer changed the prompt boundary")
        targets = full[len(ids) :]
        tokens = len(targets)
    if len(ids) + tokens - 1 > model["config"]["context"]:
        raise ValueError("comparison must fit within context without dropping prompt tokens")
    caches = {mode: KVCache() for mode in ("fp32", "nvfp4")}
    paired = []
    for step in range(tokens):
        reports, logits = {}, {}
        for mode in caches:
            folder = output / "paired" / mode / f"token_{step:03d}"
            reports[mode] = forward(
                model,
                ids,
                folder,
                executable,
                kv_cache=caches[mode],
                last_only=True,
                precision=mode,
            )
            logits[mode] = json.loads((folder / "logits.json").read_text())
        p, q = reports["fp32"], reports["nvfp4"]
        target = targets[step] if targets is not None else p["selection"]["token_id"]
        errors = [a - b for a, b in zip(logits["fp32"], logits["nvfp4"], strict=True)]
        paired.append(
            dict(
                input_ids=list(ids),
                target=target,
                prediction_agrees=p["predictions"][-1] == q["predictions"][-1],
                predictions={mode: r["predictions"][-1] for mode, r in reports.items()},
                negative_log_likelihood={
                    mode: -math.log(max(r["probabilities"][target], 1e-300))
                    for mode, r in reports.items()
                }
                if targets is not None
                else None,
                logit_max_absolute_error=max(map(abs, errors)),
                logit_relative_l2=math.sqrt(
                    math.fsum(e * e for e in errors)
                    / max(math.fsum(x * x for x in logits["fp32"]), 1e-300)
                ),
                **distribution_error(p["probabilities"], q["probabilities"]),
                cycles={mode: r["cycles"] for mode, r in reports.items()},
                wall_seconds={mode: r["wall_seconds"] for mode, r in reports.items()},
                reference_seconds={mode: r["reference_seconds"] for mode, r in reports.items()},
            )
        )
        ids.append(target)
    generated = {}
    for mode in caches:
        folder = output / "greedy" / mode
        if answer is not None:
            result = complete(model, prompt, folder, executable, precision=mode)
            generated[mode] = result["answer"]
        else:
            generation = Generation(model, prompt, heal=False, precision=mode)
            for step in range(tokens):
                generation.next(folder / f"token_{step:03d}", executable)
            generated[mode] = generation.text[len(prompt) :]
    return dict(
        prompt=prompt,
        expected_answer=answer,
        generated=generated,
        correct={mode: text == answer for mode, text in generated.items()} if answer else None,
        paired=paired,
        verified=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/sentence_completion/checkpoint.json"
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=8,
        help="deterministic sample of reserved sentence test questions",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="explicit prompt; repeat for several character/subword cases",
    )
    parser.add_argument("--tokens", type=int, default=4, help="new tokens per explicit prompt")
    parser.add_argument("--out", type=Path, default=ROOT / "build/inference/nvfp4-comparison")
    args = parser.parse_args()
    if not 1 <= args.cases <= 576 or not 1 <= args.tokens <= 32:
        parser.error("cases must be 1..576 and tokens 1..32")
    model = load_checkpoint(args.checkpoint)
    if args.prompt:
        cases = [(prompt, None) for prompt in args.prompt]
    else:
        if (
            model.get("training", {}).get("task")
            != "two-fact location question; short generated answer"
        ):
            parser.error("non-sentence checkpoints need --prompt")
        selected = random.Random(20261109).sample(examples("test"), args.cases)
        cases = [(example.prompt, example.answer) for example in selected]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").unlink(missing_ok=True)
    executable = compile_model_runner(args.out, compact=True)
    print(f"Comparing FP32 with {POLICY}; checkpoint {model['sha256'][:12]}", flush=True)
    results = []
    for i, (prompt, answer) in enumerate(cases):
        result = compare_case(
            model, prompt, answer, args.out / f"case_{i:03d}", executable, args.tokens
        )
        results.append(result)
        print(f"Case {i + 1}/{len(cases)}: hardware/reference checks passed", flush=True)
        for mode, text in result["generated"].items():
            print(f"  {mode:5s}: {text!r}", flush=True)
    steps = [step for case in results for step in case["paired"]]
    modes = ("fp32", "nvfp4")
    report = dict(
        verified=True,
        quantization_policy=POLICY,
        checkpoint_sha256=model["sha256"],
        rtl_source_sha256=executable.source_sha256,
        cases=results,
        scope="native NVFP4 linear layers; FP32 attention and nonlinear kernels; host shared-memory transport",
        selection="explicit prompts"
        if args.prompt
        else "reserved test split; random seed 20261109",
        paired_tokens=len(steps),
        top1_agreement=sum(s["prediction_agrees"] for s in steps) / len(steps),
        mean_total_variation=math.fsum(s["total_variation"] for s in steps) / len(steps),
        mean_kl_fp32_to_nvfp4=math.fsum(s["kl_fp32_to_nvfp4"] for s in steps) / len(steps),
        max_logit_absolute_error=max(s["logit_max_absolute_error"] for s in steps),
        mean_logit_relative_l2=math.fsum(s["logit_relative_l2"] for s in steps) / len(steps),
        answer_correct={mode: sum(c["correct"][mode] for c in results) for mode in modes}
        if not args.prompt
        else None,
        mean_answer_nll={
            mode: math.fsum(s["negative_log_likelihood"][mode] for s in steps) / len(steps)
            for mode in modes
        }
        if not args.prompt
        else None,
        paired_cycles={mode: sum(s["cycles"][mode] for s in steps) for mode in modes},
        paired_wall_seconds={
            mode: math.fsum(s["wall_seconds"][mode] for s in steps) for mode in modes
        },
        timing_scope="includes host quantization, launch handling, and verification; simulation cycles exclude host work",
    )
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Top-1 agreement on identical prefixes: {report['top1_agreement']:.1%}")
    print(f"Mean probability total variation: {report['mean_total_variation']:.5f}")
    if report["answer_correct"] is not None:
        print(f"Exact answers out of {len(cases)}: {report['answer_correct']}")
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
