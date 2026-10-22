"""Check the FPGA host interface in simulation; this does not program a board."""

import argparse
import json

from snackwell.assembler import assemble
from snackwell.device import compile_device, lint_device, run_device_program
from snackwell.kernels.basic import (
    KERNELS,
    default_inputs,
    default_matrix_inputs,
    kernel_memory,
)
from snackwell.numeric import value as float_value
from snackwell.paths import ROOT
from snackwell.simulator import checked_command, tool_path


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    output = ROOT / "build/fpga"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "result.json"
    report_path.unlink(missing_ok=True)
    executable = compile_device(output)
    lint_device(output)
    print("PASS device elaboration/lint with simulation initializers disabled", flush=True)
    checked_command([tool_path("vvp"), str(executable)], ROOT, output / "control.log")
    print("PASS host loading, readback, backpressure, timeout, abort and relaunch", flush=True)
    workloads = []
    for name, (verify, warps) in KERNELS.items():
        inputs = default_matrix_inputs if name.startswith("matmul") else default_inputs
        memory = kernel_memory(*inputs())
        words = assemble((ROOT / "kernels" / f"{name}.asm").read_text())
        after, cycles = run_device_program(
            executable, words, memory, output / name, warp_mask=warps
        )
        verify(memory, after)
        print(f"PASS {name}: all 16 outputs and memory guards match ({cycles} execution cycles)")
        print("  C =", [float_value(word) for word in after[32:48]])
        workloads.append(dict(kernel=name, cycles=cycles, reference_match=True))
    report_path.write_text(
        json.dumps(
            dict(
                transport="simulated_register_bus",
                hardware_tested=False,
                synthesis_run=False,
                workloads=workloads,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"\nVerified the bus interface in RTL simulation. Results: {output}")
    print("Physical FPGA operation still requires a board transport, constraints and synthesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
