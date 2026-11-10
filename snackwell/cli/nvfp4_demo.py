"""Verify packed NVFP4 matrix execution and measure error against FP32 inputs."""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from snackwell.assembler import assemble
from snackwell.character_model import block_prefix, encode_prompt, load_checkpoint
from snackwell.global_memory import compile_global_runner, run_global_program
from snackwell.kernels.nvfp4 import images, program
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, FMUL, reference
from snackwell.reference.nvfp4 import quantize


def checkpoint_projection(path, prompt):
    model = load_checkpoint(path)
    config, weights = model["config"], model["weights"]
    width = config["width"]
    if width not in (16, 32, 64):
        raise ValueError("projection demo supports checkpoint widths 16, 32, and 64")
    ids = encode_prompt(model, prompt, pad=False)
    if len(ids) < 4:
        raise ValueError("projection demo needs at least four prompt tokens")
    prefix = block_prefix(0, config.get("layers", 1))
    a = []
    for position in range(len(ids) - 4, len(ids)):
        x = [
            weights["token_embedding"][ids[position] * width + col]
            + weights["position_embedding"][position * width + col]
            for col in range(width)
        ]
        mean = math.fsum(x) / width
        variance = math.fsum((element - mean) ** 2 for element in x) / width
        inv = 1 / math.sqrt(variance + config["epsilon"])
        a.append(
            [
                value(
                    bits(
                        (x[col] - mean) * inv * weights[prefix + "gamma1"][col]
                        + weights[prefix + "beta1"][col]
                    )
                )
                for col in range(width)
            ]
        )
    wq = weights[prefix + "wq"]
    b = [[wq[k * width + column] for k in range(width)] for column in range(8)]
    # Use the full query-weight tensor's scale, including columns outside the demo.
    weight_scale = max(max(map(abs, wq)) / (448 * 6), value(1)) if any(wq) else 1.0
    return (
        a,
        b,
        weight_scale,
        {
            "checkpoint": str(Path(path).resolve()),
            "checkpoint_sha256": model["sha256"],
            "prompt": prompt,
            "operation": "first-block query projection, first eight output channels, before bias",
            "token_positions": list(range(len(ids) - 4, len(ids))),
            "input_preparation": "host embedding and layer normalization, rounded to FP32",
            "scope": "one projection slice; not a full-model accuracy evaluation",
        },
    )


def fp32_matmul(a, b):
    result = []
    for row in a:
        for column in b:
            accumulator = 0
            for x, y in zip(row, column, strict=True):
                accumulator = reference(FADD, accumulator, reference(FMUL, bits(x), bits(y)))
            result.append(accumulator)
    return result


def numerical_error(actual, expected):
    differences = [value(x) - value(y) for x, y in zip(actual, expected, strict=True)]
    squared_error = math.fsum(x * x for x in differences)
    squared_signal = math.fsum(value(x) ** 2 for x in expected)
    return {
        "max_absolute": max(map(abs, differences)),
        "root_mean_square": math.sqrt(squared_error / len(expected)),
        "relative_l2": math.sqrt(squared_error / squared_signal) if squared_signal else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latency", type=int, default=2, help="additional external-memory delay cycles"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="measure a query-projection slice from this checkpoint"
    )
    parser.add_argument(
        "--prompt", default="ROMEO:", help="prompt for the optional checkpoint projection"
    )
    args = parser.parse_args()
    if not 0 <= args.latency <= 1000:
        parser.error("latency must be between 0 and 1000")
    output = ROOT / "build/kernels/nvfp4" / ("projection" if args.checkpoint else "matrix")
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    if args.checkpoint:
        a, b, weight_scale, context = checkpoint_projection(args.checkpoint, args.prompt)
    else:
        a = [[float((r * 32 + k) % 19 - 9) for k in range(32)] for r in range(4)]
        b = [[float((c * 32 + k) % 13 - 6) for k in range(32)] for c in range(8)]
        weight_scale, context = None, {"operation": "deterministic matrix product"}
    qa, qb = quantize(a), quantize(b, tensor_scale=weight_scale)
    rows, columns, k_size = len(a), len(b), len(a[0])
    shared, memory, wanted_shared, wanted_global = images(qa, qb)
    full_precision = fp32_matmul(a, b)
    expected = wanted_global[6144 : 6144 + rows * columns]
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "a_rows": a,
                "b_columns": b,
                "quantized_a": asdict(qa),
                "quantized_b": asdict(qb),
                "context": context,
            },
            indent=2,
        )
        + "\n"
    )
    executable = compile_global_runner(output / "runner")
    cases = {}
    jobs = rows * columns * k_size // 256
    for name, overlap in (("serial", False), ("overlapped", True)):
        source = program(rows, columns, k_size, overlap=overlap)
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "program.asm").write_text(source)
        after, external, metrics = run_global_program(
            executable,
            assemble(source),
            shared,
            memory,
            folder,
            latency=args.latency,
            max_cycles=1000000,
        )
        if after != wanted_shared or external != wanted_global:
            raise RuntimeError(
                f"{name}: output, guard, status, or ticket mismatch; inspect {folder}"
            )
        if (metrics["reads"], metrics["writes"], metrics["copy_writes"]) != (
            jobs * 24,
            rows * columns,
            jobs * 24,
        ):
            raise RuntimeError(f"{name}: incorrect transfer count")
        with (folder / "trace.csv").open() as stream:
            trace = list(csv.DictReader(stream))
        metrics["array_passes"] = sum(int(row["matrix_job"]) for row in trace)
        metrics["nvfp4_macs"] = sum(int(row["nvfp4_mac"]).bit_count() for row in trace)
        metrics["fp32_tensor_multiplies"] = sum(
            int(row["fp32_tensor_multiply"]).bit_count() for row in trace
        )
        metrics["copy_transfers_during_arithmetic"] = sum(
            int(row["copy_read"]) + int(row["copy_write"])
            for row in trace
            if int(row["arithmetic_active"])
        )
        if (metrics["array_passes"], metrics["nvfp4_macs"], metrics["fp32_tensor_multiplies"]) != (
            jobs * 4,
            rows * columns * k_size,
            0,
        ):
            raise RuntimeError(f"{name}: incorrect arithmetic path or work count")
        cases[name] = metrics
    error = numerical_error(expected, full_precision)
    saved = cases["serial"]["cycles"] - cases["overlapped"]["cycles"]
    report = dict(
        passed=True,
        shape={"m": rows, "n": columns, "k": k_size},
        scale_groups=k_size // 16,
        output_tiles=rows * columns // 16,
        context=context,
        cases=cases,
        cycles_saved=saved,
        cycle_reduction_percent=100 * saved / cases["serial"]["cycles"],
        quantization_error=error,
        fp32_reference=full_precision,
        nvfp4_output=expected,
        staged_operand_bytes=jobs * 96,
        equivalent_fp32_operand_bytes=jobs * 512,
        guards_verified=True,
        tickets_verified=True,
        hardware_tested=False,
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS NVFP4: A[{rows},{k_size}] @ B[{k_size},{columns}], {jobs} tensor jobs")
    print(
        "All outputs match the independent quantized reference; every memory guard and ticket verified."
    )
    for row in range(rows):
        print(
            "  "
            + " ".join(f"{value(x):9.4f}" for x in expected[row * columns : (row + 1) * columns])
        )
    print(
        f"Serial: {cases['serial']['cycles']} cycles; double buffered: {cases['overlapped']['cycles']} cycles"
    )
    print(f"Trace: {rows * columns * k_size} narrow MACs; zero FP32 tensor multiplier launches.")
    if error["relative_l2"] is not None:
        print(
            f"Error versus original FP32 inputs: relative L2 {100 * error['relative_l2']:.2f}%; max absolute {error['max_absolute']:.6g}"
        )
    if args.checkpoint:
        print(
            f"Checkpoint {context['checkpoint_sha256'][:12]}: one query-projection slice, before bias."
        )
    print(f"Assembly, inputs, traces, memory images and result.json: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
