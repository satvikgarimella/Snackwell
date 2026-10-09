#!/usr/bin/env python3
"""Compile and run every Snackwell testbench with Icarus Verilog."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from fp32_reference import write_vectors
from tensor_reference import write_tensor_vectors
from assemble import assemble
from run_kernel import default_inputs, vector_add_memory, run_program, verify_vector_add

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", help="Optional testbench stems")
    args = parser.parse_args()
    compiler = shutil.which("iverilog") or "/opt/homebrew/bin/iverilog"
    runtime = shutil.which("vvp") or "/opt/homebrew/bin/vvp"
    out = ROOT / "build" / "tests"
    out.mkdir(parents=True, exist_ok=True)
    vectors = out / "fp32_vectors.txt"
    write_vectors(vectors)
    tensor_vectors = out / "tensor_vectors.txt"
    write_tensor_vectors(tensor_vectors)
    sources = [ROOT / "snackwell_pkg.sv", ROOT / "integer_alu.sv"]
    sources += sorted(ROOT.glob("rtl/**/*.sv")) + sorted(ROOT.glob("rtl/**/*.v"))
    benches = sorted(ROOT.glob("tb/tb_*.sv"))
    if args.tests:
        unknown = set(args.tests) - {tb.stem for tb in benches}
        if unknown:
            parser.error(f"Unknown testbenches: {sorted(unknown)}")
        benches = [tb for tb in benches if tb.stem in args.tests]
    failures = []
    for tb in benches:
        work = out / tb.stem
        work.mkdir(exist_ok=True)
        log = ""
        try:
            compiled = subprocess.run(
                [compiler, "-g2012", "-Wall", "-s", tb.stem, "-o", str(work / "sim.vvp"),
                 *map(str, sources), str(tb)],
                capture_output=True, text=True, timeout=30,
            )
            log = compiled.stdout + compiled.stderr
            passed = compiled.returncode == 0 and not re.search(
                r"implicit definition of wire|dangling input port", log)
            if passed:
                if tb.stem == "tb_snackwell_kernel":
                    before = vector_add_memory(*default_inputs())
                    words = assemble((ROOT / "kernels/vector_add.asm").read_text())
                    after, _, cycles = run_program(work / "sim.vvp", words, before, work)
                    verify_vector_add(before, after)
                    log += (work / "simulation.log").read_text()
                    log += f"PASS: all 16 vector outputs and memory guards verified ({cycles} cycles)\n"
                else:
                    run = subprocess.run([runtime, "sim.vvp", f"+VECTORS={vectors}",
                                          f"+TENSOR_VECTORS={tensor_vectors}"], cwd=work,
                                         capture_output=True, text=True, timeout=30)
                    log += run.stdout + run.stderr
                    passed = run.returncode == 0 and not re.search(r"\b(FAIL|FATAL|ERROR)\b", run.stdout + run.stderr)
        except (subprocess.TimeoutExpired, OSError, ValueError, RuntimeError) as error:
            passed = False
            log += str(error)
        (work / "test.log").write_text(log)
        print(f"{'PASS' if passed else 'FAIL'} {tb.stem}", flush=True)
        if not passed:
            failures.append(tb.stem)
            print(log[-3000:])
    print(f"\n{len(benches) - len(failures)}/{len(benches)} testbenches passed. Logs: {out}")
    python_failed = False
    if not args.tests:
        try:
            checked = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "scripts/tests", "-v"],
                                     cwd=ROOT, capture_output=True, text=True, timeout=120)
            (out / "python.log").write_text(checked.stdout + checked.stderr)
            python_failed = checked.returncode != 0
            print(f"{'FAIL' if python_failed else 'PASS'} Python assembler and kernel tests", flush=True)
            if python_failed:
                print((checked.stdout + checked.stderr)[-5000:])
        except (subprocess.TimeoutExpired, OSError) as error:
            python_failed = True
            print(f"FAIL Python tests: {error}")
    return bool(failures) or python_failed


if __name__ == "__main__":
    sys.exit(main())
