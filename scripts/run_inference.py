"""Run generated assembly kernels and multi-launch inference pipelines on RTL."""
import json
from pathlib import Path

from assemble import assemble
from inference_kernels import NAMES, build_case, value
from inference_reference import attention_reference, expected_stage, verify_approx, verify_stage
from run_kernel import (ROOT, compile_runner, run_program, write_hex, execute_kernel,
                        default_inputs, default_matrix_inputs)


def execute_case(case, output, executable):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    (output / "output.hex").unlink(missing_ok=True)
    for stage in case.stages:
        (output / stage.name / "result.json").unlink(missing_ok=True)
    # Preflight every stage's assembly before running any of the workload.
    programs = [assemble(stage.source) for stage in case.stages]
    memory = case.layout.memory.copy()
    write_hex(output / "initial_memory.hex", memory)
    (output / "layout.json").write_text(json.dumps({
        "regions": {name: {"byte_address": start*4, "words": count}
                    for name, (start, count) in case.layout.regions.items()},
        "constants": {str(address): f"{word:08x}" for word, address in case.layout.constants.items()},
        "rows": case.rows, "cols": case.cols,
    }, indent=2) + "\n")
    reports = []
    for stage, words in zip(case.stages, programs):
        directory = output / stage.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").unlink(missing_ok=True)
        (directory / "program.asm").write_text(stage.source)
        # Validate mathematical preconditions before launching this stage.
        expected_stage(stage, memory, case.layout)
        after, _, cycles = run_program(executable, words, memory, directory, warp_mask=1,
                                       max_cycles=200000)
        accuracy = verify_stage(stage, memory, after, case.layout)
        profile = json.loads((directory / "profile.json").read_text())
        report = {"stage": stage.name, "verified": True, "cycles": cycles,
                  "static_instructions": len(words), "accuracy": accuracy, "profile": profile}
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        # Pass the actual dumped memory to the next launch, with no host arithmetic.
        memory = after
    start, count = case.layout.regions["out"]
    output_bits = memory[start:start+count]
    report = {"kernel": case.name, "verified": True, "rows": case.rows, "cols": case.cols,
              "launches": len(reports), "cycles": sum(r["cycles"] for r in reports),
              "output_bits": [f"{word:08x}" for word in output_bits], "stages": reports}
    if case.name == "attention":
        report["end_to_end_accuracy"] = verify_approx(output_bits, attention_reference(case), "attention")
    write_hex(output / "output.hex", memory)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_suite(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    executable = compile_runner(output)
    reports = {}
    for name in ("vector_add", "vector_mul", "matmul_4x4", "matmul_4x4_simd"):
        a, b = default_matrix_inputs() if name.startswith("matmul") else default_inputs()
        reports[name] = execute_kernel(name, a, b, output / name, executable)
        print(f"PASS {name}: {reports[name]['cycles']} cycles", flush=True)
    for name in NAMES:
        case = build_case(name)
        reports[name] = execute_case(case, output / name, executable)
        print(f"PASS {name}: {reports[name]['cycles']} cycles ({reports[name]['launches']} launch(es))", flush=True)
    case = build_case("tiled_matmul", size=16)
    reports["tiled_matmul_16x16"] = execute_case(case, output / "tiled_matmul_16x16", executable)
    print(f"PASS tiled_matmul_16x16: {reports['tiled_matmul_16x16']['cycles']} cycles", flush=True)
    summary = {"verified": True, "workloads": {name: {"cycles": r["cycles"], "verified": r["verified"]}
                                              for name, r in reports.items()}}
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"All {len(reports)} workloads passed. Results: {output.resolve()}")
    return reports


def run_cli(args):
    output = args.out or ROOT / "build/kernels" / args.kernel
    (output / "result.json").unlink(missing_ok=True)
    if args.kernel == "all":
        if args.inputs:
            raise ValueError("--inputs applies to one named kernel, not the whole suite")
        return run_suite(output)
    inputs = json.loads(args.inputs.read_text()) if args.inputs else None
    case = build_case(args.kernel, rows=args.rows, cols=args.cols, size=args.size, inputs=inputs)
    executable = compile_runner(output)
    report = execute_case(case, output, executable)
    print(f"PASS {args.kernel}: {len(report['output_bits'])} outputs verified, {report['cycles']} simulated cycles, "
          f"{report['launches']} launch(es)")
    if "end_to_end_accuracy" in report:
        print(f"  Attention max absolute error: {report['end_to_end_accuracy']['max_absolute_error']:.3g}")
    preview = [value(int(word, 16)) for word in report["output_bits"][:16]]
    print("  First outputs: " + ", ".join(f"{x:.6g}" for x in preview))
    print(f"Assembly, images, traces, profiles, and results: {output.resolve()}")
    return report
