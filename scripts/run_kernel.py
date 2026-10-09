#!/usr/bin/env python3
"""Assemble, execute, verify, and profile Snackwell kernels in RTL simulation."""
import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

from assemble import assemble, write_image
from fp32_reference import FADD, FMUL, reference
from kernel_profile import summarize_trace
from tensor_reference import matmul
from inference_kernels import NAMES as INFERENCE_KERNELS

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = 0xDEADBEEF


def float_bits(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("input elements must be numbers")
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_float(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def write_hex(path, words):
    path.write_text("".join(f"{word:08x}\n" for word in words))


def read_hex(path, count):
    tokens = re.sub(r"//[^\n]*", "", path.read_text()).split()
    if len(tokens) != count or any(not re.fullmatch(r"[0-9a-fA-F]{8}", word) for word in tokens):
        raise ValueError(f"{path}: expected {count} known 32-bit words")
    return [int(word, 16) for word in tokens]


def checked_command(args, cwd, log_path, timeout=30):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    log = result.stdout + result.stderr
    log_path.write_text(log)
    if result.returncode or re.search(r"\b(FAIL|FATAL|ERROR)\b|implicit definition of wire|dangling input port", log):
        raise RuntimeError(f"Command failed; see {log_path}\n{log[-2000:]}")
    return log


def compile_runner(output):
    output.mkdir(parents=True, exist_ok=True)
    executable = (output / "sim.vvp").resolve()
    sources = [ROOT / "snackwell_pkg.sv", ROOT / "integer_alu.sv"]
    sources += sorted(ROOT.glob("rtl/**/*.sv")) + sorted(ROOT.glob("rtl/**/*.v"))
    checked_command([shutil.which("iverilog") or "/opt/homebrew/bin/iverilog", "-g2012", "-Wall",
                     "-s", "tb_snackwell_kernel", "-o", str(executable),
                     *map(str, sources), str(ROOT / "tb/tb_snackwell_kernel.sv")],
                    ROOT, output / "compile.log")
    return executable


def run_program(executable, words, memory, output, active_masks=0xFFFF, entry=0, warp_mask=0xF,
                max_cycles=10000):
    """Execute an assembled program; no arithmetic is emulated here."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(memory) != 1024 or any(type(word) is not int or not 0 <= word < 2**32 for word in memory):
        raise ValueError("shared-memory image must have 1024 unsigned 32-bit words")
    if not 0 <= active_masks <= 0xFFFF or not 0 <= entry < 2**32 or not 1 <= warp_mask <= 15:
        raise ValueError("invalid launch masks or entry address")
    if type(max_cycles) is not int or not 1 <= max_cycles <= 1000000:
        raise ValueError("cycle limit must be 1 through 1000000")
    program_path, data_path = output / "program.hex", output / "input.hex"
    result_path, register_path = output / "output.hex", output / "registers.hex"
    # Remove outputs from earlier runs so a failed simulation cannot look successful.
    for name in ("output.hex", "registers.hex", "trace.csv", "profile.json"):
        (output / name).unlink(missing_ok=True)
    write_image(program_path, words)
    write_hex(data_path, memory)
    log = checked_command([shutil.which("vvp") or "/opt/homebrew/bin/vvp", str(executable.resolve()),
                           f"+PROGRAM={program_path}", f"+DATA={data_path}", f"+RESULT={result_path}",
                           f"+REGISTERS={register_path}", f"+ACTIVE_MASKS={active_masks:04x}", f"+ENTRY={entry}",
                           f"+WARP_MASK={warp_mask:x}", f"+TRACE={output / 'trace.csv'}", f"+MAX_CYCLES={max_cycles}"],
                          output, output / "simulation.log")
    match = re.search(r"PASS kernel completed in (\d+) cycles", log)
    if not match:
        raise RuntimeError("simulation did not report successful kernel completion")
    cycles = int(match[1])
    profile = summarize_trace(output / "trace.csv", cycles)
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return read_hex(result_path, 1024), read_hex(register_path, 256), cycles


def default_inputs():
    return [float_bits(i + 1) for i in range(16)], [float_bits(10 * (i + 1)) for i in range(16)]


def default_matrix_inputs():
    return [float_bits(i + 1) for i in range(16)], [float_bits(16 - i) for i in range(16)]


def kernel_memory(a, b):
    if len(a) != 16 or len(b) != 16:
        raise ValueError("kernels require exactly 16 values in each input array (matrices are flat row-major)")
    memory = [SENTINEL] * 1024
    memory[:16], memory[16:32] = a, b
    return memory


def vector_add_memory(a, b):
    return kernel_memory(a, b)


def same_fp(actual, expected):
    if (expected & 0x7FFFFFFF) > 0x7F800000:
        return (actual & 0x7FFFFFFF) > 0x7F800000
    return actual == expected


def verify_vector_add(before, after, active_masks=0xFFFF):
    verify_outputs(before, after, [reference(FADD, a, b) for a, b in zip(before[:16], before[16:32])],
                   active_masks)


def verify_vector_mul(before, after, active_masks=0xFFFF):
    verify_outputs(before, after, [reference(FMUL, a, b) for a, b in zip(before[:16], before[16:32])],
                   active_masks)


def verify_matmul(before, after, output_mask=0xFFFF):
    if len(before) != 1024:
        raise ValueError("expected complete shared-memory images")
    verify_outputs(before, after, matmul(before[:16], before[16:32]), output_mask)


def verify_outputs(before, after, expected_outputs, output_mask=0xFFFF):
    if len(before) != 1024 or len(after) != 1024:
        raise ValueError("expected complete shared-memory images")
    for word, actual in enumerate(after):
        if 32 <= word < 48 and (output_mask >> (word - 32)) & 1:
            expected = expected_outputs[word-32]
            correct = same_fp(actual, expected)
        else:
            expected = before[word]
            correct = actual == expected
        if not correct:
            raise ValueError(f"memory byte {word*4}: got {actual:08x}, expected {expected:08x}")


KERNELS = {
    "vector_add": (verify_vector_add, 0xF),
    "vector_mul": (verify_vector_mul, 0xF),
    "matmul_4x4": (verify_matmul, 0x1),
    "matmul_4x4_simd": (verify_matmul, 0xF),
}


def execute_kernel(name, a, b, output, executable):
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    before = kernel_memory(a, b)
    words = assemble((ROOT / "kernels" / f"{name}.asm").read_text())
    verify, warp_mask = KERNELS[name]
    after, _, cycles = run_program(executable, words, before, output, warp_mask=warp_mask)
    verify(before, after)
    report = {"kernel": name, "verified": True, "cycles": cycles,
              "instructions": len(words), "warp_mask": warp_mask,
              "a_bits": [f"{x:08x}" for x in a], "b_bits": [f"{x:08x}" for x in b],
              "c_bits": [f"{x:08x}" for x in after[32:48]],
              "profile": json.loads((output / "profile.json").read_text())}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def comparison_markdown(tensor, simd):
    rows = [
        ("Execution cycles (address setup through EXIT)", tensor["cycles"], simd["cycles"]),
        ("Warps launched", 1, 4),
        ("Instructions issued across all warps", tensor["profile"]["instructions_issued"],
         simd["profile"]["instructions_issued"]),
    ]
    for label, key in [("Shared-memory words read", "load_words"), ("Shared-memory words written", "store_words"),
                       ("Memory service cycles", "shared_service_cycles"),
                       ("Requests with bank conflicts", "conflicting_memory_requests")]:
        rows.append((label, tensor["profile"][key], simd["profile"][key]))
    text = ["# Snackwell 4x4 matrix multiplication", "", "Both programs passed the independent FP32 reference and memory guards.",
            "", "| Measurement | TMUL | SIMD FMUL/FADD |", "| --- | ---: | ---: |"]
    text += [f"| {label} | {left} | {right} |" for label, left, right in rows]
    text += ["", f"SIMD cycles / TMUL cycles: **{simd['cycles'] / tensor['cycles']:.3f}x**.", "",
             "Both compute 64 products and 64 additions, starting each sum at positive zero,",
             "rounding after each operation and accumulating in increasing inner-dimension order.",
             "The same row-major inputs and output locations are used. TMUL reuses operands inside",
             "the array; the SIMD program reloads operands for each output. This compares these two",
             "complete kernels, not isolated arithmetic throughput or an optimized SIMD lower bound.", "",
             "## Tensor instruction phases", "", "| Phase | Cycles |", "| --- | ---: |"]
    text += [f"| {phase.replace('_', ' ')} | {count} |" for phase, count in
             tensor["profile"]["tensor_phases"].items() if phase != "idle"]
    text += ["", "## SM issue cycles", "", "These categories partition total execution time. A dependency check only counts",
             "the cycle when the selected warp discovers a dependency. Later blocked cycles may",
             "appear under no eligible warp. These are scheduler observations, not causal stall attribution.",
             "", "| Category | TMUL | SIMD |", "| --- | ---: | ---: |"]
    categories = sorted(set(tensor["profile"]["issue_cycles"]) | set(simd["profile"]["issue_cycles"]))
    text += [f"| {category.replace('_', ' ')} | {tensor['profile']['issue_cycles'].get(category, 0)} | "
             f"{simd['profile']['issue_cycles'].get(category, 0)} |" for category in categories]
    text += ["", "Unit busy/service counters overlap issue activity and each other; do not sum them.",
             "Reset, host image loading, and warp launch are excluded. These are RTL simulation cycles,",
             "not measured FPGA time. FP operand values can change the cycle counts.", "",
             "Each kernel directory contains program/input/output images, result.json, profile.json,",
             "and trace.csv with one row per execution cycle.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", nargs="?", choices=[*KERNELS, "matmul_compare", *INFERENCE_KERNELS, "all"], default="vector_add")
    parser.add_argument("--inputs", type=Path, help="JSON with the named kernel's flat row-major inputs (see docs/INFERENCE_KERNELS.md)")
    parser.add_argument("--out", type=Path, help="Output directory (default: build/kernels/<kernel>)")
    parser.add_argument("--size", type=int, choices=[4, 8, 16], default=8, help="Square tiled_matmul/linear dimension")
    parser.add_argument("--rows", type=int, choices=[4, 8, 16], default=4, help="Inference matrix rows / attention sequence length")
    parser.add_argument("--cols", type=int, choices=[4, 8, 16], default=4, help="Inference matrix columns / attention head width")
    args = parser.parse_args()
    output = args.out or ROOT / "build/kernels" / args.kernel
    try:
        if args.kernel in INFERENCE_KERNELS or args.kernel == "all":
            from run_inference import run_cli
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
            if not all(same_fp(int(t, 16), int(s, 16)) for t, s in zip(tensor["c_bits"], simd["c_bits"])):
                raise ValueError("tensor and SIMD matrix outputs disagree")
            report = {"kernel": "matmul_compare", "verified": True, "tensor": tensor, "simd": simd,
                      "simd_cycles_over_tensor_cycles": simd["cycles"] / tensor["cycles"]}
            (output / "comparison.md").write_text(comparison_markdown(tensor, simd))
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            report = execute_kernel(args.kernel, a, b, output, executable)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Kernel run failed: {error}\n")
    if args.kernel == "matmul_compare":
        print("PASS matmul_compare: both kernels match the FP32 reference; inputs and memory guards unchanged")
        print(f"  TMUL: {tensor['cycles']} cycles; SIMD FMUL/FADD: {simd['cycles']} cycles")
        print(f"  SIMD cycles / TMUL cycles: {report['simd_cycles_over_tensor_cycles']:.3f}x")
        result_bits = tensor["c_bits"]
    else:
        print(f"PASS {args.kernel}: all 16 outputs match the FP32 reference ({report['cycles']} simulated cycles)")
        result_bits = report["c_bits"]
    if args.kernel.startswith("matmul"):
        print("  C =")
        for row in range(4):
            print("    " + " ".join(f"{bits_float(int(value, 16)):10g}" for value in result_bits[row*4:row*4+4]))
    else:
        symbol = "+" if args.kernel == "vector_add" else "*"
        for i, value in enumerate(result_bits):
            print(f"  C[{i:2}] = {bits_float(a[i]):g} {symbol} {bits_float(b[i]):g} = {bits_float(int(value, 16)):g}")
    print(f"Program, memory images, cycle traces, profiles and results: {output.resolve()}")


if __name__ == "__main__":
    main()
