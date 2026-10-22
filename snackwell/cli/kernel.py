"""Assemble, execute, verify, and profile Snackwell kernels in RTL simulation."""

import argparse
import json
import subprocess
from pathlib import Path

from snackwell.inference_runner import execute_case, run_suite
from snackwell.kernels.basic import (
    KERNELS,
    comparison_markdown,
    default_inputs,
    default_matrix_inputs,
    execute_kernel,
    kernel_memory,
    same_fp,
)
from snackwell.kernels.inference import NAMES as INFERENCE_KERNELS
from snackwell.kernels.inference import build_case
from snackwell.numeric import bits as float_bits
from snackwell.numeric import value
from snackwell.numeric import value as bits_float
from snackwell.paths import ROOT
from snackwell.simulator import compile_runner


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
    print(
        f"PASS {args.kernel}: {len(report['output_bits'])} outputs verified, {report['cycles']} simulated cycles, "
        f"{report['launches']} launch(es)"
    )
    if "end_to_end_accuracy" in report:
        print(
            f"  End-to-end max absolute error: {report['end_to_end_accuracy']['max_absolute_error']:.3g}"
        )
    preview = [value(int(word, 16)) for word in report["output_bits"][:16]]
    print("  First outputs: " + ", ".join(f"{x:.6g}" for x in preview))
    print(f"Assembly, images, traces, profiles, and results: {output.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kernel",
        nargs="?",
        choices=[*KERNELS, "matmul_compare", *INFERENCE_KERNELS, "all"],
        default="vector_add",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="JSON with the named kernel's flat row-major inputs (see docs/INFERENCE_KERNELS.md)",
    )
    parser.add_argument(
        "--out", type=Path, help="Output directory (default: build/kernels/<kernel>)"
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Square tiled_matmul/linear dimension",
    )
    parser.add_argument(
        "--rows",
        type=int,
        choices=[4, 8, 16],
        default=4,
        help="Inference matrix rows / attention sequence length",
    )
    parser.add_argument(
        "--cols",
        type=int,
        choices=[4, 8, 16],
        default=4,
        help="Inference matrix columns / attention head width",
    )
    args = parser.parse_args()
    output = args.out or ROOT / "build/kernels" / args.kernel
    try:
        if args.kernel in INFERENCE_KERNELS or args.kernel == "all":
            run_cli(args)
            return
        (output / "result.json").unlink(missing_ok=True)
        if args.kernel == "matmul_compare":
            (output / "comparison.md").unlink(missing_ok=True)
            for name in ("matmul_4x4", "matmul_4x4_simd"):
                (output / name / "result.json").unlink(missing_ok=True)
        a, b = default_matrix_inputs() if args.kernel.startswith("matmul") else default_inputs()
        if args.inputs:
            inputs = json.loads(args.inputs.read_text())
            a, b = [float_bits(x) for x in inputs["a"]], [float_bits(x) for x in inputs["b"]]
        kernel_memory(a, b)  # Validate inputs before invoking the compiler.
        executable = compile_runner(output)
        if args.kernel == "matmul_compare":
            tensor = execute_kernel("matmul_4x4", a, b, output / "matmul_4x4", executable)
            simd = execute_kernel("matmul_4x4_simd", a, b, output / "matmul_4x4_simd", executable)
            if not all(
                same_fp(int(t, 16), int(s, 16)) for t, s in zip(tensor["c_bits"], simd["c_bits"])
            ):
                raise ValueError("tensor and SIMD matrix outputs disagree")
            report = {
                "kernel": "matmul_compare",
                "verified": True,
                "tensor": tensor,
                "simd": simd,
                "simd_cycles_over_tensor_cycles": simd["cycles"] / tensor["cycles"],
            }
            (output / "comparison.md").write_text(comparison_markdown(tensor, simd))
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            report = execute_kernel(args.kernel, a, b, output, executable)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        parser.exit(1, f"Kernel run failed: {error}\n")
    if args.kernel == "matmul_compare":
        print(
            "PASS matmul_compare: both kernels match the FP32 reference; inputs and memory guards unchanged"
        )
        print(f"  TMUL: {tensor['cycles']} cycles; SIMD FMUL/FADD: {simd['cycles']} cycles")
        print(f"  SIMD cycles / TMUL cycles: {report['simd_cycles_over_tensor_cycles']:.3f}x")
        result_bits = tensor["c_bits"]
    else:
        print(
            f"PASS {args.kernel}: all 16 outputs match the FP32 reference ({report['cycles']} simulated cycles)"
        )
        result_bits = report["c_bits"]
    if args.kernel.startswith("matmul"):
        print("  C =")
        for row in range(4):
            print(
                "    "
                + " ".join(
                    f"{bits_float(int(value, 16)):10g}"
                    for value in result_bits[row * 4 : row * 4 + 4]
                )
            )
    else:
        symbol = "+" if args.kernel == "vector_add" else "*"
        for i, value in enumerate(result_bits):
            print(
                f"  C[{i:2}] = {bits_float(a[i]):g} {symbol} {bits_float(b[i]):g} = {bits_float(int(value, 16)):g}"
            )
    print(f"Program, memory images, cycle traces, profiles and results: {output.resolve()}")


if __name__ == "__main__":
    main()
