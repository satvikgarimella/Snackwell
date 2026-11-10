"""Execute through a delayed 64 KiB simulated external RAM, not physical SDRAM."""

import csv
import json
import os
from functools import lru_cache

from snackwell.assembler import write_image
from snackwell.paths import ROOT, rtl_sources
from snackwell.simulator import (
    ModelSimulator,
    SimulatorSession,
    checked_command,
    compile_testbench,
    read_hex,
    rtl_source_digest,
    tool_path,
    write_hex,
)

GLOBAL_WORDS = 16384


@lru_cache(maxsize=2)
def model_global_runner(backend, digest):
    directory = ROOT / "build/simulators/global-model" / f"{backend}-{digest[:20]}"
    directory.mkdir(parents=True, exist_ok=True)
    if backend == "icarus":
        return compile_global_runner(directory)
    executable = directory / "global_sim"
    executable.unlink(missing_ok=True)
    checked_command(
        [
            tool_path("verilator"),
            "--binary",
            "--timing",
            "--assert",
            "-DSYNTHESIS",
            "-Wno-fatal",
            "-CFLAGS",
            "-O3",
            "--top-module",
            "tb_snackwell_global_memory",
            "--Mdir",
            str(directory),
            "-o",
            executable.name,
            "-j",
            str(min(4, os.cpu_count() or 1)),
            *map(str, rtl_sources()),
            str(ROOT / "tb/tb_snackwell_global_memory.sv"),
        ],
        ROOT,
        directory / "compile.log",
        timeout=300,
    )
    return executable


def run_model_global(
    simulator, words, memory, output, *, active_masks, warp_mask, latency, max_cycles=200000
):
    """Persistent binary transport when available; independently traced SV fallback."""
    from snackwell.numeric import SENTINEL

    if type(latency) is not int or not 0 <= latency <= 1000:
        raise ValueError("global latency must be an integer from 0 to 1000")
    for name, number, low, high in (
        ("active masks", active_masks, 0, 65535),
        ("warp mask", warp_mask, 1, 15),
        ("cycle limit", max_cycles, 1, 1000000),
    ):
        if type(number) is not int or not low <= number <= high:
            raise ValueError(f"invalid {name}")
    for name, data, count in (("program", words, None), ("external memory", memory, GLOBAL_WORDS)):
        if not (1 <= len(data) <= 1024 if count is None else len(data) == count):
            raise ValueError(f"invalid {name} length")
        if any(type(w) is not int or not 0 <= w < 2**32 for w in data):
            raise ValueError(f"invalid {name} words")
    shared = [SENTINEL] * 1024
    if (
        isinstance(simulator, ModelSimulator)
        and simulator.backend == "verilator"
        and not simulator.trace
    ):
        with SimulatorSession(simulator) as session:
            return run_model_global(
                session,
                words,
                memory,
                output,
                active_masks=active_masks,
                warp_mask=warp_mask,
                latency=latency,
                max_cycles=max_cycles,
            )
    if isinstance(simulator, SimulatorSession) and not simulator.trace:
        after, registers, external, metrics = simulator.run_global_words(
            words, shared, memory, active_masks, warp_mask, latency, max_cycles
        )
    else:
        backend = getattr(simulator, "backend", "icarus")
        runner = model_global_runner(
            backend, getattr(simulator, "source_sha256", None) or rtl_source_digest()
        )
        after, external, metrics = run_global_program(
            runner,
            words,
            shared,
            memory,
            output,
            warp_mask=warp_mask,
            active_masks=active_masks,
            latency=latency,
            max_cycles=max_cycles,
            backend=backend,
            read_registers=True,
        )
        registers = metrics.pop("registers")
        with (output / "trace.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        metrics["nvfp4_macs"] = sum(int(r["nvfp4_mac"]).bit_count() for r in rows)
        metrics["fp32_tensor_multiplies"] = sum(
            int(r["fp32_tensor_multiply"]).bit_count() for r in rows
        )
    metrics.update(transport="simulated_external_memory", hardware_tested=False, latency=latency)
    return after, registers, external, metrics


def compile_global_runner(output):
    return compile_testbench(
        ROOT / "tb/tb_snackwell_global_memory.sv", output, defines=("SYNTHESIS",)
    )


def run_global_program(
    executable,
    words,
    shared,
    memory,
    output,
    *,
    warp_mask=1,
    active_masks=0xFFFF,
    latency=2,
    max_cycles=20000,
    expect_fault=False,
    backend="icarus",
    read_registers=False,
):
    if backend not in ("icarus", "verilator"):
        raise ValueError("global-memory backend must be icarus or verilator")
    for name, image, size in (
        ("program", words, None),
        ("shared", shared, 1024),
        ("global", memory, GLOBAL_WORDS),
    ):
        if not (1 <= len(image) <= 1024 if size is None else len(image) == size):
            raise ValueError(f"invalid {name} image length")
        if any(type(word) is not int or not 0 <= word < 2**32 for word in image):
            raise ValueError(f"{name} image must contain unsigned 32-bit words")
    for name, value, low, high in (
        ("warp mask", warp_mask, 1, 15),
        ("active masks", active_masks, 0, 65535),
        ("latency", latency, 0, 1000),
        ("cycle limit", max_cycles, 1, 1000000),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"invalid {name}")
    if type(expect_fault) is not bool:
        raise ValueError("expect_fault must be a boolean")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    for name in ("output.hex", "global-output.hex", "metrics.txt", "trace.csv", "result.json"):
        (output / name).unlink(missing_ok=True)
    write_image(output / "program.hex", words)
    write_hex(output / "input.hex", shared)
    write_hex(output / "global-input.hex", memory)
    checked_command(
        [
            *([tool_path("vvp")] if backend == "icarus" else []),
            str(executable),
            f"+PROGRAM={output / 'program.hex'}",
            f"+DATA={output / 'input.hex'}",
            f"+GLOBAL={output / 'global-input.hex'}",
            f"+OUTPUT={output / 'output.hex'}",
            f"+GLOBAL_OUTPUT={output / 'global-output.hex'}",
            f"+METRICS={output / 'metrics.txt'}",
            f"+TRACE={output / 'trace.csv'}",
            f"+WARP_MASK={warp_mask}",
            f"+ACTIVE_MASKS={active_masks}",
            f"+LATENCY={latency}",
            f"+MAX_CYCLES={max_cycles}",
            f"+EXPECT_FAULT={int(expect_fault)}",
            *([f"+REGISTERS={output / 'registers.hex'}"] if read_registers else []),
        ],
        ROOT,
        output / "simulation.log",
        timeout=60,
    )
    after = read_hex(output / "output.hex", 1024)
    external = read_hex(output / "global-output.hex", GLOBAL_WORDS)
    keys = (
        "cycles",
        "reads",
        "writes",
        "responses",
        "request_stalls",
        "independent_integer_issues",
        "dependency_stalls",
        "status",
        "copy_status",
        "copy_reads",
        "copy_writes",
        "copy_arithmetic_overlap_cycles",
    )
    values = [int(v) for v in (output / "metrics.txt").read_text().split()]
    if len(values) != len(keys) or not 0 < values[0] <= max_cycles:
        raise RuntimeError("Invalid global-memory metrics")
    metrics = dict(zip(keys, values, strict=True))
    if metrics["reads"] + metrics["writes"] != metrics["responses"]:
        raise RuntimeError("External requests did not drain")
    metrics.update(transport="simulated_external_memory", hardware_tested=False, latency=latency)
    if read_registers:
        metrics["registers"] = read_hex(output / "registers.hex", 256)
    (output / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return after, external, metrics
