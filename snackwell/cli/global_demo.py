"""Run global -> shared -> tensor memory -> global matrix multiplication."""

import csv
import json

from snackwell.assembler import assemble
from snackwell.global_memory import GLOBAL_WORDS, compile_global_runner, run_global_program
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.tensor import matmul


def main():
    output = ROOT / "build/kernels/global_tensor"
    (output / "result.json").unlink(missing_ok=True)
    a = [bits(i + 1) for i in range(32)]
    b = [bits((i % 7) - 3) for i in range(32)]
    memory = [0xDEADBEEF] * GLOBAL_WORDS
    memory[4096:4128], memory[5120:5152] = a, b
    shared = [0xDEADBEEF] * 1024
    shared[192:196] = [32 << 12, (16 << 12) | 128, (32 << 12) | 16, (16 << 12) | 192]
    expected = [0] * 16
    for k in (0, 4):
        tile = [a[r * 8 + k + c] for r in range(4) for c in range(4)]
        expected = matmul(tile, b[k * 4 : (k + 4) * 4], expected)
    executable = compile_global_runner(output)
    words = assemble((ROOT / "kernels/global_tensor.asm").read_text())
    after, external, report = run_global_program(executable, words, shared, memory, output)
    wanted_shared, wanted_global = shared.copy(), memory.copy()
    wanted_shared[:64] = a + b
    wanted_shared[144:152] = [0] * 8
    wanted_global[6144:6160] = expected
    if after != wanted_shared or external != wanted_global or report["status"] != 0:
        raise RuntimeError(f"Global tensor numerical/status/guard mismatch; inspect {output}")
    with (output / "trace.csv").open() as stream:
        jobs = sum(int(row["matrix_job"]) for row in csv.DictReader(stream))
    if (report["reads"], report["writes"], jobs) != (64, 16, 2):
        raise RuntimeError("Global tensor transfer/job count mismatch")
    report.update(
        passed=True,
        matrix_jobs=jobs,
        macs=128,
        guards_verified=True,
        shape={"m": 4, "n": 4, "k": 8},
        arithmetic="FP32 sequential non-fused MAC",
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS global memory: A[4,8] @ B[8,4] through the full SM")
    print("Global RAM -> shared memory -> 4x4 array/tensor memory -> global RAM")
    print("64 word reads, 16 word writes; both matrix jobs and all memory guards verified.")
    print("C =")
    for row in range(4):
        print("  " + " ".join(f"{value(word):8g}" for word in expected[row * 4 : row * 4 + 4]))
    print(f"{report['cycles']} simulated cycles; delayed/backpressured memory; status = 0")
    print(f"Memory images, trace.csv and result.json: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
