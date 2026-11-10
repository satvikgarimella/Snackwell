"""Compare serial and broadcast global loads on identical resident model launches."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.kv_cache import Generation
from snackwell.numeric import bits
from snackwell.paths import ROOT
from snackwell.sentence_task import examples
from snackwell.simulator import ModelSimulator, compile_verilator_runner, rtl_source_digest


def comparison_runner(*, serial):
    """Direct-clock, resident-only runner; include compile options in KV identity."""
    source = rtl_source_digest()
    defines = ("SNACKWELL_SERIAL_GLOBAL_LOADS",) if serial else ()
    executable = compile_verilator_runner(session=True, defines=defines)
    if rtl_source_digest() != source:
        raise RuntimeError("RTL changed during comparison build")
    identity = hashlib.sha256((source + "\0" + "\0".join(defines)).encode()).hexdigest()
    return ModelSimulator(
        executable,
        "verilator",
        source_sha256=identity,
        session_executable=executable,
        compact=True,
    ), dict(
        source_sha256=source,
        build_sha256=identity,
        defines=defines,
        executable=str(executable),
        binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )


def profile(stages):
    result = {}
    for stage in stages:
        item = result.setdefault(stage["operation"], dict(cycles=0, launches=0))
        item["cycles"] += stage["cycles"]
        item["launches"] += 1
        for key in ("reads", "writes", "copy_reads", "nvfp4_macs", "fp32_tensor_multiplies"):
            item[key] = item.get(key, 0) + stage["memory"][key]
    return result


def compare(model, prompt, expected, runners, output, *, precision, tokens, latency):
    generators = {
        mode: Generation(
            model,
            prompt,
            heal=False,
            precision=precision,
            memory="resident",
            memory_latency=latency,
        )
        for mode in runners
    }
    records = []
    try:
        for step in range(tokens):
            # Alternate execution order; cycles are the reproducible comparison.
            order = list(runners) if step % 2 == 0 else list(reversed(runners))
            pair = {
                mode: generators[mode].next(output / mode / f"token_{step:03d}", runners[mode])
                for mode in order
            }
            left, right = pair["serial"], pair["broadcast"]
            probabilities = {
                mode: [bits(x) for x in r["probabilities"]] for mode, r in pair.items()
            }
            if (
                not all(r["verified"] for r in pair.values())
                or left["input_ids"] != right["input_ids"]
                or probabilities["serial"] != probabilities["broadcast"]
                or left["selection"] != right["selection"]
                or left["cache"]["phase"] != right["cache"]["phase"]
            ):
                raise ValueError(f"serial/broadcast outputs differ at token {step + 1}")
            records.append(
                dict(
                    identical=True,
                    input_ids=left["input_ids"],
                    probability_words=probabilities["serial"],
                    selection=left["selection"],
                    text=left["next_character"],
                    phase=left["cache"]["phase"],
                    runs={
                        mode: dict(
                            cycles=r["cycles"],
                            wall_seconds=r["wall_seconds"],
                            launches=r["launches"],
                            memory=r["resident"],
                            operations=profile(r["stages"]),
                        )
                        for mode, r in pair.items()
                    },
                )
            )
            if expected is not None and "." in left["next_character"]:
                break
    finally:
        for generation in generators.values():
            generation.close()
    answer = generators["broadcast"].text[len(prompt) :]
    return dict(
        prompt=prompt,
        expected=expected,
        generated=answer,
        correct=answer == expected if expected is not None else None,
        tokens=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/sentence_completion/checkpoint.json"
    )
    parser.add_argument("--precision", choices=("fp32", "nvfp4"), default="nvfp4")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--memory-latency", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "build/inference/broadcast-comparison")
    args = parser.parse_args()
    if (
        not 1 <= args.cases <= 576
        or not 1 <= args.tokens <= 32
        or not 0 <= args.memory_latency <= 1000
    ):
        parser.error("cases must be 1..576, tokens 1..32, memory latency 0..1000")
    model = load_checkpoint(args.checkpoint)
    if args.prompt:
        cases = [(p, None) for p in args.prompt]
    else:
        if (
            model.get("training", {}).get("task")
            != "two-fact location question; short generated answer"
        ):
            parser.error("non-sentence checkpoints require --prompt")
        cases = [
            (e.prompt, e.answer)
            for e in random.Random(20261109).sample(examples("test"), args.cases)
        ]
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "result.json"
    result_path.unlink(missing_ok=True)
    runners, builds = {}, {}
    for mode in ("serial", "broadcast"):
        print(f"Building {mode} load runner...", flush=True)
        runners[mode], builds[mode] = comparison_runner(serial=mode == "serial")
    if builds["serial"]["source_sha256"] != builds["broadcast"]["source_sha256"]:
        raise RuntimeError("RTL changed between comparison builds")
    results = []
    for i, (prompt, expected) in enumerate(cases):
        result = compare(
            model,
            prompt,
            expected,
            runners,
            args.out / f"case_{i:03d}",
            precision=args.precision,
            tokens=args.tokens,
            latency=args.memory_latency,
        )
        results.append(result)
        print(
            f"PASS case {i + 1}/{len(cases)}: identical probabilities and tokens; {result['generated']!r}",
            flush=True,
        )
    tokens = [t for case in results for t in case["tokens"]]
    totals = {}
    for mode in runners:
        runs = [t["runs"][mode] for t in tokens]
        totals[mode] = {
            key: sum(r[key] for r in runs) for key in ("cycles", "launches", "wall_seconds")
        }
        totals[mode]["memory"] = {
            key: sum(r["memory"][key] for r in runs)
            for key in (
                "reads",
                "writes",
                "responses",
                "host_upload_bytes",
                "host_readback_bytes",
                "constant_upload_bytes",
            )
        }
        phases = {}
        for phase in ("prefill", "decode", "rebuild"):
            selected = [t["runs"][mode] for t in tokens if t["phase"] == phase]
            if selected:
                phases[phase] = dict(
                    tokens=len(selected),
                    **{
                        key: sum(r[key] for r in selected) / len(selected)
                        for key in ("cycles", "wall_seconds")
                    },
                )
        totals[mode]["phase_averages"] = phases
    report = dict(
        verified=True,
        identical=True,
        precision=args.precision,
        memory="resident",
        memory_latency=args.memory_latency,
        checkpoint_sha256=model["sha256"],
        builds=builds,
        compared_tokens=len(tokens),
        answer_correct=sum(c["correct"] for c in results) if not args.prompt else None,
        totals=totals,
        cases=results,
        scope="RTL simulation with independent numerical and memory guard checks; host orchestrates and quantizes",
    )
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    for mode, total in totals.items():
        print(
            f"{mode}: {total['cycles']:,} cycles, {total['memory']['reads']:,} reads, {total['wall_seconds']:.2f}s"
        )
    print(f"PASS {len(tokens)} identical token comparisons. Results: {result_path}")


if __name__ == "__main__":
    main()
