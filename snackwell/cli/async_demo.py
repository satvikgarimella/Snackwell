"""Compare serial versus overlapped tile copies using the same RTL and inputs."""

import argparse
import csv
import json

from snackwell.assembler import assemble
from snackwell.global_memory import compile_global_runner, run_global_program
from snackwell.kernels.async_tensor import images, program
from snackwell.numeric import value
from snackwell.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latency", type=int, default=2, help="additional external-memory delay cycles"
    )
    args = parser.parse_args()
    output = ROOT / "build/kernels/async_tensor"
    (output / "result.json").unlink(missing_ok=True)
    shared, memory, wanted_shared, wanted_global = images()
    executable = compile_global_runner(output / "runner")
    cases = {}
    for name, overlap in (("serial", False), ("overlapped", True)):
        source = program(overlap=overlap)
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
        )
        if after != wanted_shared or external != wanted_global:
            raise RuntimeError(
                f"{name}: numerical, status, ticket or memory guard mismatch; inspect {folder}"
            )
        if (metrics["reads"], metrics["writes"], metrics["copy_reads"], metrics["copy_writes"]) != (
            96,
            16,
            96,
            96,
        ):
            raise RuntimeError(f"{name}: incorrect memory transfer count")
        if metrics["copy_status"] or metrics["status"]:
            raise RuntimeError(f"{name}: memory operation failed")
        with (folder / "trace.csv").open() as stream:
            trace = list(csv.DictReader(stream))
        if sum(int(row["matrix_job"]) for row in trace) != 3:
            raise RuntimeError(f"{name}: incorrect matrix job count")
        metrics["copy_transfers_during_arithmetic"] = sum(
            int(row["copy_read"]) + int(row["copy_write"])
            for row in trace
            if int(row["arithmetic_active"])
        )
        cases[name] = metrics
    serial, overlapped = cases["serial"], cases["overlapped"]
    if serial["copy_arithmetic_overlap_cycles"] or not overlapped["copy_arithmetic_overlap_cycles"]:
        raise RuntimeError("Trace did not distinguish serial from concurrent copy/arithmetic")
    if (
        serial["copy_transfers_during_arithmetic"]
        or not overlapped["copy_transfers_during_arithmetic"]
    ):
        raise RuntimeError("No copy transfers took place during tensor arithmetic")
    if overlapped["cycles"] >= serial["cycles"]:
        raise RuntimeError("Overlapped demo did not reduce cycles at this memory latency")
    saved = serial["cycles"] - overlapped["cycles"]
    report = dict(
        passed=True,
        shape={"m": 4, "n": 4, "k": 12},
        cases=cases,
        cycles_saved=saved,
        cycle_reduction_percent=100 * saved / serial["cycles"],
        guards_verified=True,
        tickets_verified=True,
        hardware_tested=False,
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS asynchronous copies: A[4,12] @ B[12,4], three matrix jobs")
    print(
        "Both versions match the independent FP32 reference; all memory guards and tickets verified."
    )
    for row in range(4):
        print(
            "  "
            + " ".join(
                f"{value(word):8g}" for word in wanted_global[6144 + row * 4 : 6148 + row * 4]
            )
        )
    print(f"Serial: {serial['cycles']} cycles; double buffered: {overlapped['cycles']} cycles")
    print(
        f"Saved {saved} cycles ({report['cycle_reduction_percent']:.1f}%); "
        f"{overlapped['copy_arithmetic_overlap_cycles']} cycles with copying and PE arithmetic active together."
    )
    print(f"Assembly, memory images, traces and result.json: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
