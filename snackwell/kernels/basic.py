"""Inputs, verification, and reports for the hand-written vector and matrix kernels."""

import json

from snackwell.assembler import assemble
from snackwell.numeric import SENTINEL, SHARED_WORDS
from snackwell.numeric import bits as float_bits
from snackwell.paths import ROOT
from snackwell.reference.fp32 import FADD, FMUL, reference
from snackwell.reference.tensor import matmul
from snackwell.simulator import run_program


def default_inputs():
    return [float_bits(i + 1) for i in range(16)], [float_bits(10 * (i + 1)) for i in range(16)]


def default_matrix_inputs():
    return [float_bits(i + 1) for i in range(16)], [float_bits(16 - i) for i in range(16)]


def kernel_memory(a, b):
    if len(a) != 16 or len(b) != 16:
        raise ValueError(
            "kernels require exactly 16 values in each input array (matrices are flat row-major)"
        )
    memory = [SENTINEL] * SHARED_WORDS
    memory[:16], memory[16:32] = a, b
    return memory


def vector_add_memory(a, b):
    return kernel_memory(a, b)


def same_fp(actual, expected):
    if (expected & 0x7FFFFFFF) > 0x7F800000:
        return (actual & 0x7FFFFFFF) > 0x7F800000
    return actual == expected


def verify_vector_add(before, after, active_masks=0xFFFF):
    verify_outputs(
        before,
        after,
        [reference(FADD, a, b) for a, b in zip(before[:16], before[16:32])],
        active_masks,
    )


def verify_vector_mul(before, after, active_masks=0xFFFF):
    verify_outputs(
        before,
        after,
        [reference(FMUL, a, b) for a, b in zip(before[:16], before[16:32])],
        active_masks,
    )


def verify_matmul(before, after, output_mask=0xFFFF):
    if len(before) != SHARED_WORDS:
        raise ValueError("expected complete shared-memory images")
    verify_outputs(before, after, matmul(before[:16], before[16:32]), output_mask)


def verify_outputs(before, after, expected_outputs, output_mask=0xFFFF):
    if len(before) != SHARED_WORDS or len(after) != SHARED_WORDS:
        raise ValueError("expected complete shared-memory images")
    for word, actual in enumerate(after):
        if 32 <= word < 48 and (output_mask >> (word - 32)) & 1:
            expected = expected_outputs[word - 32]
            correct = same_fp(actual, expected)
        else:
            expected = before[word]
            correct = actual == expected
        if not correct:
            raise ValueError(f"memory byte {word * 4}: got {actual:08x}, expected {expected:08x}")


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
    report = {
        "kernel": name,
        "verified": True,
        "cycles": cycles,
        "instructions": len(words),
        "warp_mask": warp_mask,
        "a_bits": [f"{x:08x}" for x in a],
        "b_bits": [f"{x:08x}" for x in b],
        "c_bits": [f"{x:08x}" for x in after[32:48]],
        "profile": json.loads((output / "profile.json").read_text()),
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def comparison_markdown(tensor, simd):
    rows = [
        ("Execution cycles (address setup through EXIT)", tensor["cycles"], simd["cycles"]),
        ("Warps launched", 1, 4),
        (
            "Instructions issued across all warps",
            tensor["profile"]["instructions_issued"],
            simd["profile"]["instructions_issued"],
        ),
    ]
    for label, key in [
        ("Shared-memory words read", "load_words"),
        ("Shared-memory words written", "store_words"),
        ("Memory service cycles", "shared_service_cycles"),
        ("Requests with bank conflicts", "conflicting_memory_requests"),
    ]:
        rows.append((label, tensor["profile"][key], simd["profile"][key]))
    text = [
        "# Snackwell 4x4 matrix multiplication",
        "",
        "Both programs passed the independent FP32 reference and memory guards.",
        "",
        "| Measurement | TMUL | SIMD FMUL/FADD |",
        "| --- | ---: | ---: |",
    ]
    text += [f"| {label} | {left} | {right} |" for label, left, right in rows]
    text += [
        "",
        f"SIMD cycles / TMUL cycles: **{simd['cycles'] / tensor['cycles']:.3f}x**.",
        "",
        "Both compute 64 products and 64 additions, starting each sum at positive zero,",
        "rounding after each operation and accumulating in increasing inner-dimension order.",
        "The same row-major inputs and output locations are used. TMUL reuses operands inside",
        "the array; the SIMD program reloads operands for each output. This compares these two",
        "complete kernels, not isolated arithmetic throughput or an optimized SIMD lower bound.",
        "",
        "## Tensor instruction phases",
        "",
        "| Phase | Cycles |",
        "| --- | ---: |",
    ]
    text += [
        f"| {phase.replace('_', ' ')} | {count} |"
        for phase, count in tensor["profile"]["tensor_phases"].items()
        if phase != "idle"
    ]
    text += [
        "",
        "## SM issue cycles",
        "",
        "These categories partition total execution time. A dependency check only counts",
        "the cycle when the selected warp discovers a dependency. Later blocked cycles may",
        "appear under no eligible warp. These are scheduler observations, not causal stall attribution.",
        "",
        "| Category | TMUL | SIMD |",
        "| --- | ---: | ---: |",
    ]
    categories = sorted(
        set(tensor["profile"]["issue_cycles"]) | set(simd["profile"]["issue_cycles"])
    )
    text += [
        f"| {category.replace('_', ' ')} | {tensor['profile']['issue_cycles'].get(category, 0)} | "
        f"{simd['profile']['issue_cycles'].get(category, 0)} |"
        for category in categories
    ]
    text += [
        "",
        "Unit busy/service counters overlap issue activity and each other; do not sum them.",
        "Reset, host image loading, and warp launch are excluded. These are RTL simulation cycles,",
        "not measured FPGA time. FP operand values can change the cycle counts.",
        "",
        "Each kernel directory contains program/input/output images, result.json, profile.json,",
        "and trace.csv with one row per execution cycle.",
        "",
    ]
    return "\n".join(text)
