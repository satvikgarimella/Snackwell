"""Synthesize the full device; this does not place, route, or program an FPGA."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from snackwell.assembler import assemble
from snackwell.device import compile_device, run_device_program
from snackwell.kernels.basic import KERNELS, default_inputs, default_matrix_inputs, kernel_memory
from snackwell.paths import ROOT, rtl_sources
from snackwell.simulator import checked_command, tool_path


def yosys_path(override):
    candidates = (
        [override]
        if override
        else [
            os.environ.get("YOSYS"),
            "yosys",
            "yowasp-yosys",
            str(
                ROOT
                / "build/synthesis-venv"
                / ("Scripts/yowasp-yosys.exe" if os.name == "nt" else "bin/yowasp-yosys")
            ),
        ]
    )
    for candidate in candidates:
        if candidate and (executable := shutil.which(candidate)):
            return str(Path(executable).absolute())
    raise RuntimeError("Install Yosys with read_slang support; see docs/SYNTHESIS.md")


def synthesis_script(output, target):
    # These are Yosys strings, not shell arguments. All paths are relative to the
    # repository so native Yosys and YoWASP see the same input/output tree.
    def quoted(path):
        return json.dumps(str(path.relative_to(ROOT)))

    # Slang consumes its own file-list syntax; quotes passed directly through
    # the Yosys command parser otherwise become part of the filename.
    source_list = output / "sources.f"
    source_list.write_text("\n".join(quoted(path) for path in rtl_sources()) + "\n")
    lines = [
        f"read_slang -j 1 --top snackwell_device -D SYNTHESIS -f {source_list.relative_to(ROOT)}",
        "hierarchy -check -top snackwell_device",
    ]
    if target == "generic":
        # Independent lanes/PEs operate concurrently. Skip the expensive search
        # for mutually exclusive arithmetic resources across the flattened SM.
        lines.append("synth -top snackwell_device -noshare -latches error")
        # Separate partial vector aliases so simulators do not infer artificial
        # whole-vector feedback paths between otherwise independent bits.
        lines.append("splitnets -driver")
    else:
        lines.append("synth_intel_alm -top snackwell_device -family cyclonev -noiopad -noclkbuf")
    lines += [
        "check -assert",
        f"tee -o {quoted(output / 'stats.json')} stat -json",
        f"write_verilog -noattr {quoted(output / 'netlist.v')}",
        f"write_json {quoted(output / 'netlist.json')}",
    ]
    return "\n".join(lines) + "\n"


def verify_netlist(output, libraries=(), backend="verilator"):
    """Use only the exported gates and the package of ISA constants as the DUT."""
    sources = [
        str(ROOT / "rtl/snackwell_pkg.sv"),
        *map(str, libraries),
        str(output / "netlist.v"),
        str(ROOT / "tb/tb_snackwell_device.sv"),
    ]
    if backend == "verilator":
        executable = output / "verilator/netlist_sim"
        command = [
            tool_path("verilator"),
            "--binary",
            "--timing",
            "-Wno-fatal",
            "--top-module",
            "tb_snackwell_device",
            "-j",
            "4",
            "--Mdir",
            str(executable.parent),
            "-o",
            executable.name,
            *sources,
        ]
        run = [str(executable)]
    else:
        executable = output / "netlist.vvp"
        command = [
            tool_path("iverilog"),
            "-g2012",
            "-s",
            "tb_snackwell_device",
            "-o",
            str(executable),
            *sources,
        ]
        run = [tool_path("vvp"), str(executable)]
    compile_log = checked_command(command, ROOT, output / "compile-netlist.log", timeout=600)
    for warning in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT", "IMPLICIT", "MODMISSING"):
        if f"%Warning-{warning}:" in compile_log:
            raise RuntimeError(f"Netlist simulation compilation reported {warning}")
    checked_command(run, ROOT, output / "protocol.log", timeout=120)
    print("PASS synthesized netlist: host protocol, reset, abort and relaunch", flush=True)
    rtl = compile_device(output / "rtl")
    results = []
    for name, (verify, warps) in KERNELS.items():
        inputs = default_matrix_inputs if name.startswith("matmul") else default_inputs
        before = kernel_memory(*inputs())
        words = assemble((ROOT / "kernels" / f"{name}.asm").read_text())
        actual, cycles = run_device_program(
            executable, words, before, output / name, warp_mask=warps, timeout=300, backend=backend
        )
        expected, rtl_cycles = run_device_program(
            rtl, words, before, output / "rtl" / name, warp_mask=warps
        )
        verify(before, actual)
        if actual != expected or cycles != rtl_cycles:
            raise RuntimeError(f"{name}: synthesized netlist differs from RTL")
        results.append(
            dict(kernel=name, cycles=cycles, full_memory_match=True, rtl_cycle_match=True)
        )
        print(
            f"PASS synthesized netlist: {name} ({cycles} cycles), all 1,024 words match RTL",
            flush=True,
        )
    return dict(
        protocol_passed=True,
        simulator=backend,
        workloads=results,
        cell_models_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries
        },
    )


def cell_libraries(executable, override):
    """Find the simulation models belonging to this Yosys installation."""
    prefix = Path(executable).parent.parent
    candidates = (
        [Path(override)]
        if override
        else [
            prefix / "share/yosys/intel_alm/common",
            *prefix.glob("lib/python*/site-packages/yowasp_yosys/share/intel_alm/common"),
            prefix / "Lib/site-packages/yowasp_yosys/share/intel_alm/common",
        ]
    )
    names = ("alm_sim.v", "dff_sim.v", "dsp_sim.v", "mem_sim.v", "misc_sim.v")
    for directory in candidates:
        paths = [directory / name for name in names]
        if all(path.is_file() for path in paths):
            return paths
    raise RuntimeError(
        "Pass --cell-library-dir with this Yosys installation's intel_alm/common directory"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("generic", "cyclonev"), default="generic")
    parser.add_argument("--yosys", help="Yosys executable with a built-in read_slang frontend")
    parser.add_argument("--verify", action="store_true", help="Simulate the synthesized netlist")
    parser.add_argument("--cell-library-dir", help="Cyclone V primitive simulation model directory")
    parser.add_argument(
        "--simulator",
        choices=("verilator", "icarus"),
        default="verilator",
        help="Netlist simulator (default: verilator; icarus is much slower)",
    )
    args = parser.parse_args()
    output = ROOT / "build/synthesis" / args.target
    output.mkdir(parents=True, exist_ok=True)
    # Never leave a previous successful report or netlist behind after failure.
    for name in (
        "result.json",
        "synthesis-result.json",
        "stats.json",
        "netlist.v",
        "netlist.json",
        "netlist.vvp",
    ):
        (output / name).unlink(missing_ok=True)
    executable = yosys_path(args.yosys)
    libraries = (
        cell_libraries(executable, args.cell_library_dir)
        if args.verify and args.target == "cyclonev"
        else []
    )
    version = subprocess.check_output([executable, "-V"], cwd=ROOT, text=True).strip()
    script = output / "synthesize.ys"
    script.write_text(synthesis_script(output, args.target))
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in rtl_sources()
    }
    print(f"Synthesizing full snackwell_device ({args.target}) with {version}", flush=True)
    print(f"Log: {output / 'synthesis.log'}", flush=True)
    started = time.monotonic()
    # Yosys's own log remains complete across YoWASP's ABC subprocess calls.
    # Redirected stdout alone can stop at the first ABC invocation.
    with (output / "console.log").open("w") as log:
        process = subprocess.run(
            [
                executable,
                "-Q",
                "-T",
                "-l",
                str((output / "synthesis.log").relative_to(ROOT)),
                "-s",
                str(script.relative_to(ROOT)),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"Synthesis failed; see {output / 'synthesis.log'}")
    statistics = json.loads((output / "stats.json").read_text())
    netlist = json.loads((output / "netlist.json").read_text())
    # Generic synthesis must contain only gates/FFs, never an unresolved module
    # or an unimplemented memory/arithmetic operator silently counted as a cell.
    if args.target == "generic":
        for module in netlist["modules"].values():
            for cell in module["cells"].values():
                if not cell["type"].startswith("$_"):
                    raise RuntimeError(f"Unexpected generic cell: {cell['type']}")
    del netlist
    seconds = time.monotonic() - started
    print(f"PASS synthesis and structural checks ({seconds:.1f}s)", flush=True)
    report = dict(
        top="snackwell_device",
        target=args.target,
        tool=version,
        synthesis_run=True,
        place_and_route_run=False,
        timing_verified=False,
        bitstream_generated=False,
        hardware_tested=False,
        synthesis_seconds=seconds,
        rtl_sha256=sources,
        statistics=statistics,
        netlist_sha256=hashlib.sha256((output / "netlist.v").read_bytes()).hexdigest(),
        verification=None,
    )
    (output / "synthesis-result.json").write_text(json.dumps(report, indent=2) + "\n")
    report["verification"] = (
        verify_netlist(output, libraries, args.simulator) if args.verify else None
    )
    if sources != {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in rtl_sources()
    }:
        raise RuntimeError(
            "RTL changed during synthesis/verification; rerun for a consistent result"
        )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Netlist and report: {output}")
    print("No FPGA fit, timing closure, bitstream, or hardware execution is claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
