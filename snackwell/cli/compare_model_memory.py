"""Check memory-transport equivalence and record actual external traffic."""

import argparse
import json
import random
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.kv_cache import Generation
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.sentence_task import examples
from snackwell.simulator import compile_model_runner


def compare(
    model,
    prompt,
    expected,
    output,
    simulator,
    precision,
    tokens,
    latency,
    *,
    resident=False,
    baseline=None,
):
    modes = (baseline or "global", "resident") if resident else ("shared", "global")
    generators = {
        mode: Generation(
            model,
            prompt,
            heal=False,
            precision=precision,
            memory="resident" if mode == "resident-unoptimized" else mode,
            memory_latency=latency,
            optimize_resident=mode != "resident-unoptimized",
        )
        for mode in modes
    }
    records = []
    try:
        for step in range(tokens):
            pair = {
                mode: generation.next(output / mode / f"token_{step:03d}", simulator)
                for mode, generation in generators.items()
            }
            left, right = (pair[mode] for mode in modes)
            if (
                left["input_ids"] != right["input_ids"]
                or [bits(x) for x in left["probabilities"]]
                != [bits(x) for x in right["probabilities"]]
                or left["selection"] != right["selection"]
            ):
                raise ValueError(f"memory transports differ at token {step + 1}")
            keys = (
                "reads",
                "writes",
                "responses",
                "request_stalls",
                "copy_reads",
                "copy_writes",
                "copy_arithmetic_overlap_cycles",
                "nvfp4_macs",
                "fp32_tensor_multiplies",
            )
            traffic = {key: sum(s["memory"][key] for s in right["stages"]) for key in keys}
            records.append(
                dict(
                    input_ids=left["input_ids"],
                    token_id=left["selection"]["token_id"],
                    text=left["next_character"],
                    phase=left["cache"]["phase"],
                    cycles={mode: r["cycles"] for mode, r in pair.items()},
                    wall_seconds={mode: r["wall_seconds"] for mode, r in pair.items()},
                    global_traffic=traffic,
                    identical=True,
                    **(
                        {
                            "resident": right["resident"],
                            **(
                                {
                                    "global_image_upload_bytes": left["launches"] * 65536,
                                    "global_image_readback_bytes": left["launches"] * 65536,
                                }
                                if modes[0] == "global"
                                else {"baseline_resident": left["resident"]}
                            ),
                        }
                        if resident
                        else {}
                    ),
                )
            )
            if expected is not None and "." in left["next_character"]:
                break
    finally:
        for generation in generators.values():
            generation.close()
    answer = generators[modes[1]].text[len(prompt) :]
    return dict(
        prompt=prompt,
        expected=expected,
        generated=answer,
        correct=answer == expected if expected is not None else None,
        verified=True,
        tokens=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/sentence_completion/checkpoint.json"
    )
    parser.add_argument("--precision", choices=("fp32", "nvfp4"), default="nvfp4")
    parser.add_argument("--memory-latency", type=int, default=2)
    parser.add_argument(
        "--cases", type=int, default=4, help="reserved sentence test questions; seed 20261109"
    )
    parser.add_argument(
        "--prompt", action="append", help="explicit prompt; repeat for multiple cases"
    )
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "build/inference/global-memory-comparison"
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="compare global image uploads with persistent external RAM",
    )
    parser.add_argument(
        "--baseline",
        choices=("global", "resident-copy", "resident-unoptimized"),
        help="baseline for --resident; default: global",
    )
    args = parser.parse_args()
    if args.baseline and not args.resident:
        parser.error("--baseline requires --resident")
    modes = (args.baseline or "global", "resident") if args.resident else ("shared", "global")
    if (
        not 1 <= args.cases <= 576
        or not 1 <= args.tokens <= 32
        or not 0 <= args.memory_latency <= 1000
    ):
        parser.error("cases must be 1..576, tokens 1..32, memory latency 0..1000")
    model = load_checkpoint(args.checkpoint)
    if args.prompt:
        cases = [(prompt, None) for prompt in args.prompt]
    else:
        if (
            model.get("training", {}).get("task")
            != "two-fact location question; short generated answer"
        ):
            parser.error("non-sentence checkpoints need --prompt")
        cases = [
            (e.prompt, e.answer)
            for e in random.Random(20261109).sample(examples("test"), args.cases)
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").unlink(missing_ok=True)
    simulator = compile_model_runner(args.out, compact=True)
    print(
        f"Comparing {'/'.join(modes)} memory: {args.precision}, checkpoint {model['sha256'][:12]}",
        flush=True,
    )
    results = []
    for i, (prompt, expected) in enumerate(cases):
        result = compare(
            model,
            prompt,
            expected,
            args.out / f"case_{i:03d}",
            simulator,
            args.precision,
            args.tokens,
            args.memory_latency,
            resident=args.resident,
            baseline=args.baseline,
        )
        results.append(result)
        print(
            f"PASS case {i + 1}/{len(cases)}: identical probabilities and tokens; {result['generated']!r}",
            flush=True,
        )
    tokens = [token for case in results for token in case["tokens"]]
    traffic = {
        key: sum(t["global_traffic"][key] for t in tokens) for key in tokens[0]["global_traffic"]
    }
    report = dict(
        verified=True,
        identical=True,
        memory_modes=modes,
        precision=args.precision,
        memory_latency=args.memory_latency,
        checkpoint_sha256=model["sha256"],
        rtl_source_sha256=simulator.source_sha256,
        compared_tokens=len(tokens),
        cases=results,
        answer_correct=sum(c["correct"] for c in results) if not args.prompt else None,
        cycles={mode: sum(t["cycles"][mode] for t in tokens) for mode in modes},
        wall_seconds={mode: sum(t["wall_seconds"][mode] for t in tokens) for mode in modes},
        global_traffic=traffic,
        scope="simulated external-memory kernel launches; host still quantizes, orchestrates and retains KV state",
    )
    if args.resident:
        sums = (
            "constant_upload_bytes",
            "host_upload_bytes",
            "host_readback_bytes",
            "protocol_request_bytes",
            "protocol_response_bytes",
            "rtl_launches",
            "copy_cycles",
            "device_copy_words",
            "reads",
            "writes",
            "responses",
        )
        report["resident"] = {key: sum(t["resident"][key] for t in tokens) for key in sums}
        report["resident"].update(
            capacity_bytes=tokens[0]["resident"]["capacity_bytes"],
            peak_allocation_bytes=max(t["resident"]["peak_allocation_bytes"] for t in tokens),
        )
        if modes[0] == "global":
            for key in ("global_image_upload_bytes", "global_image_readback_bytes"):
                report[key] = sum(t[key] for t in tokens)
        else:
            report["baseline_resident"] = {
                key: sum(t["baseline_resident"][key] for t in tokens) for key in sums
            }
            report["baseline_resident"].update(
                capacity_bytes=tokens[0]["baseline_resident"]["capacity_bytes"],
                peak_allocation_bytes=max(
                    t["baseline_resident"]["peak_allocation_bytes"] for t in tokens
                ),
            )
        report["scope"] = (
            "persistent external RAM; direct pointer loads/stores on RTL, host quantization and verified readbacks; no FPGA fit claim"
        )
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(tokens)} token comparisons; cycles: {report['cycles']}")
    print(
        f"External traffic: {traffic['reads']} reads, {traffic['writes']} writes, {traffic['responses']} responses"
    )
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
