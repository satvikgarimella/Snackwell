"""Run and inspect the FP32 tensor-memory prototype through the full SM."""

import json

from snackwell.assembler import assemble
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.tensor import matmul
from snackwell.simulator import compile_runner, run_program


def main():
    output = ROOT / "build/kernels/tensor_memory"
    (output / "result.json").unlink(missing_ok=True)
    a = [bits(i + 1) for i in range(32)]
    b = [bits((i % 7) - 3) for i in range(32)]
    before = [0xDEADBEEF] * 1024
    before[:32], before[32:64] = a, b
    before[192:196] = [(32 << 12), (16 << 12) | 128, (32 << 12) | 16, (16 << 12) | 192]
    expected = [0] * 16
    for k in (0, 4):
        aa = [a[r * 8 + k + c] for r in range(4) for c in range(4)]
        expected = matmul(aa, b[k * 4 : (k + 4) * 4], expected)
    executable = compile_runner(output)
    words = assemble((ROOT / "kernels/tensor_memory.asm").read_text())
    actual, _, cycles = run_program(executable, words, before, output, warp_mask=1)
    wanted = before.copy()
    wanted[128:144], wanted[144:148] = expected, [0] * 4
    if actual != wanted:
        raise RuntimeError(f"Tensor-memory numerical/status/guard mismatch; inspect {output}")
    profile = json.loads((output / "profile.json").read_text())
    if profile["tensor_mac_pairs"] != 128 or profile["tensor_memory_operand_rows"] != 16:
        raise RuntimeError("Trace did not execute both matrix tiles")
    report = {
        "passed": True,
        "arithmetic": "FP32 sequential non-fused MAC",
        "shape": {"m": 4, "n": 4, "k": 8},
        "cycles": cycles,
        "output_bits": [f"{word:08x}" for word in actual[128:144]],
        "status": actual[144:148],
        "guards_verified": True,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS tensor memory: A[4,8] @ B[8,4], two matrix jobs through the full SM")
    print("Shared-memory operands -> existing 4x4 array -> persistent tensor tile")
    print("Both completion tickets waited and released; all memory guards verified.")
    print("C =")
    for row in range(4):
        print(
            "  " + " ".join(f"{value(word):8g}" for word in actual[128 + row * 4 : 132 + row * 4])
        )
    print(f"{cycles} simulated cycles; 128 MACs; tensor status = 0")
    print(f"Assembly, memory images, trace.csv and result.json: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
