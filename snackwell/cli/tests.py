"""Compile and run the RTL benches, then the Python regression suite."""

import argparse
import json
import re
import subprocess
import sys

from snackwell.assembler import assemble
from snackwell.kernels.basic import default_inputs, vector_add_memory, verify_vector_add
from snackwell.paths import ROOT
from snackwell.reference.fp32 import write_vectors
from snackwell.reference.tensor import write_tensor_vectors
from snackwell.simulator import checked_command, compile_testbench, run_program, tool_path


def run_testbench(bench, output, vectors, tensor_vectors):
    executable = compile_testbench(bench, output)
    if bench.stem == "tb_snackwell_kernel":
        before = vector_add_memory(*default_inputs())
        words = assemble((ROOT / "kernels/vector_add.asm").read_text())
        after, _, cycles = run_program(executable, words, before, output)
        verify_vector_add(before, after)
        return f"PASS: all 16 vector outputs and memory guards verified ({cycles} cycles)\n"
    checked_command(
        [
            tool_path("vvp"),
            str(executable),
            f"+VECTORS={vectors}",
            f"+TENSOR_VECTORS={tensor_vectors}",
        ],
        output,
        output / "simulation.log",
    )
    return ""


def run_python_tests(output):
    log_path = output / "python.log"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            # An optional first Verilator build can take up to 300 seconds.
            timeout=420,
        )
        log = result.stdout + result.stderr
        passed = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError) as error:
        log, passed = str(error), False
    log_path.write_text(log)
    count = re.search(r"Ran (\d+) tests?", log)
    print(f"{'PASS' if passed else 'FAIL'} Python assembler and kernel tests", flush=True)
    if not passed:
        print(log[-5000:])
    return {"passed": passed, "test_methods": int(count[1]) if count else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*", help="Optional testbench stems; omit to run all tests")
    args = parser.parse_args()
    benches = sorted((ROOT / "tb").glob("tb_*.sv"))
    if args.tests:
        unknown = set(args.tests) - {bench.stem for bench in benches}
        if unknown:
            parser.error(f"Unknown testbenches: {sorted(unknown)}")
        benches = [bench for bench in benches if bench.stem in args.tests]

    output = ROOT / "build/tests"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    vectors = output / "fp32_vectors.txt"
    tensor_vectors = output / "tensor_vectors.txt"
    write_vectors(vectors)
    write_tensor_vectors(tensor_vectors)
    failures = []
    for bench in benches:
        work = output / bench.stem
        work.mkdir(exist_ok=True)
        logs = [work / "compile.log", work / "simulation.log"]
        for path in logs:
            path.unlink(missing_ok=True)
        try:
            detail = run_testbench(bench, work, vectors, tensor_vectors)
            passed = True
        except (subprocess.TimeoutExpired, OSError, ValueError, RuntimeError) as error:
            detail, passed = str(error), False
        log = "".join(path.read_text() for path in logs if path.exists()) + detail
        (work / "test.log").write_text(log)
        print(f"{'PASS' if passed else 'FAIL'} {bench.stem}", flush=True)
        if not passed:
            failures.append(bench.stem)
            print(log[-3000:])

    count = len(benches) - len(failures)
    print(f"\n{count}/{len(benches)} testbenches passed. Logs: {output}")
    python = run_python_tests(output) if not args.tests else None
    passed = not failures and (python is None or python["passed"])
    report = {
        "passed": passed,
        "full_suite": not args.tests,
        "rtl": {"passed": count, "total": len(benches), "failures": failures},
        "python": python,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
