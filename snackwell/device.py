"""Exercise the FPGA host endpoint in simulation, without internal memory access.

This is verification of the register bus, not a driver for a physical board.
"""

import json

from snackwell.assembler import EXIT_WORD, write_image
from snackwell.paths import ROOT, rtl_sources
from snackwell.simulator import checked_command, compile_testbench, read_hex, tool_path


def compile_device(output):
    # Exclude simulation initialization exactly as in a synthesis build. Every
    # program/data word must enter through the public bus in this testbench.
    return compile_testbench(ROOT / "tb/tb_snackwell_device.sv", output, defines=("SYNTHESIS",))


def lint_device(output):
    output.mkdir(parents=True, exist_ok=True)
    checked_command(
        [
            tool_path("verilator"),
            "--lint-only",
            "--top-module",
            "snackwell_device",
            "-DSYNTHESIS",
            "-Wno-fatal",
            *map(str, rtl_sources()),
        ],
        ROOT,
        output / "lint.log",
    )
    # Existing width warnings are logged; new structural hazards must not pass
    # just because lint allows nonfatal warnings in the older SM modules.
    log = (output / "lint.log").read_text()
    for warning in ("LATCH", "MULTIDRIVEN", "UNOPTFLAT", "PINMISSING", "IMPLICIT"):
        if f"%Warning-{warning}:" in log:
            raise RuntimeError(f"Device lint reported {warning}; see {output / 'lint.log'}")


def run_device_program(
    executable,
    words,
    memory,
    output,
    *,
    warp_mask=15,
    active_masks=0xFFFF,
    entry=0,
    max_cycles=10000,
    timeout=30,
    backend="icarus",
):
    if backend not in ("icarus", "verilator"):
        raise ValueError("device backend must be icarus or verilator")
    for name, image, exact in (("program", words, False), ("memory", memory, True)):
        if not (len(image) == 1024 if exact else 1 <= len(image) <= 1024):
            raise ValueError(f"invalid {name} image length")
        if any(type(word) is not int or not 0 <= word < 2**32 for word in image):
            raise ValueError(f"{name} image must contain unsigned 32-bit words")
    for value, lower, upper, name in (
        (warp_mask, 1, 15, "warp mask"),
        (active_masks, 0, 65535, "active masks"),
        (entry, 0, 4092, "entry PC"),
        (max_cycles, 1, 1000000, "cycle limit"),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"invalid {name}")
    if entry % 4:
        raise ValueError("entry PC must be word aligned")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    for name in ("output.hex", "cycles.txt", "result.json"):
        (output / name).unlink(missing_ok=True)
    padded = list(words) + [EXIT_WORD] * (1024 - len(words))
    write_image(output / "program.hex", padded)
    write_image(output / "input.hex", memory)
    checked_command(
        [
            *([tool_path("vvp")] if backend == "icarus" else []),
            str(executable),
            f"+DEVICE_PROGRAM={output / 'program.hex'}",
            f"+DEVICE_DATA={output / 'input.hex'}",
            f"+DEVICE_OUTPUT={output / 'output.hex'}",
            f"+DEVICE_CYCLES={output / 'cycles.txt'}",
            f"+WARP_MASK={warp_mask}",
            f"+ACTIVE_MASKS={active_masks}",
            f"+ENTRY={entry}",
            f"+MAX_CYCLES={max_cycles}",
        ],
        ROOT,
        output / "simulation.log",
        timeout=timeout,
    )
    after = read_hex(output / "output.hex", 1024)
    cycles = int((output / "cycles.txt").read_text())
    if not 0 < cycles <= max_cycles:
        raise RuntimeError("device returned an invalid cycle count")
    (output / "result.json").write_text(
        json.dumps(
            dict(
                transport="simulated_register_bus",
                hardware_tested=False,
                cycles=cycles,
                warp_mask=warp_mask,
                active_masks=active_masks,
                entry=entry,
                launches=1,
            ),
            indent=2,
        )
        + "\n"
    )
    return after, cycles
