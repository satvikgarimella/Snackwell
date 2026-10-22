"""Compile the SM testbench and execute programs with checked memory images."""

import hashlib
import json
import os
import re
import select
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from snackwell.assembler import write_image
from snackwell.numeric import SHARED_WORDS
from snackwell.paths import ROOT, rtl_sources
from snackwell.profile import summarize_trace


@dataclass(frozen=True)
class ModelSimulator:
    executable: Path
    backend: str
    trace: bool = False
    source_sha256: str | None = None
    session_executable: Path | None = None


class SimulatorSession:
    """Run separate, reset RTL launches in one Verilator process.

    Only filenames and launch controls cross this interface. The C++ host clocks
    the same RTL directly; traces use the original SystemVerilog testbench.
    """

    def __init__(self, simulator):
        if simulator.backend != "verilator":
            raise ValueError("persistent sessions require Verilator")
        self.executable = (
            simulator.session_executable if not simulator.trace else None
        ) or simulator.executable
        self.traced_executable = simulator.executable
        self.backend, self.trace = simulator.backend, simulator.trace
        self.source_sha256 = simulator.source_sha256
        self.process = None
        self.pending = b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        try:
            process.stdin.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        finally:
            process.stdout.close()
            self.pending = b""

    def run(self, output, active_masks, warp_mask, entry, max_cycles, timeout=30):
        directory = str(output)
        if "\n" in directory or "\r" in directory:
            raise ValueError("session paths cannot contain newlines")
        log_path = output / "simulation.log"
        lines = []
        try:
            if self.trace and self.executable != self.traced_executable:
                self.close()
                self.executable = self.traced_executable
            if self.process is None:
                self.process = subprocess.Popen(
                    [str(self.executable.resolve()), "+SESSION"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                )
            request = (
                f"{directory}\n{active_masks:04x} {warp_mask:x} {entry} {max_cycles} "
                f"{int(self.trace)}\n"
            )
            self.process.stdin.write(request.encode())
            deadline = time.monotonic() + timeout
            while True:
                if b"\n" in self.pending:
                    line, _, self.pending = self.pending.partition(b"\n")
                    text = line.decode(errors="replace") + "\n"
                    lines.append(text)
                    if re.search(r"\b(FAIL|FATAL|ERROR)\b|%Error", text):
                        raise RuntimeError(
                            f"RTL session failed; see {log_path}\n{''.join(lines)[-2000:]}"
                        )
                    if re.fullmatch(r"PASS kernel completed in \d+ cycles\n", text):
                        return "".join(lines)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise subprocess.TimeoutExpired(str(self.executable), timeout)
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    lines.append(self.pending.decode(errors="replace"))
                    raise RuntimeError(f"RTL session exited before completion; see {log_path}")
                self.pending += data
        except BaseException:
            self.close()
            raise
        finally:
            log_path.write_text("".join(lines))


def rtl_source_digest():
    """Identify the RTL and testbench loaded by a long-lived model runner."""
    digest = hashlib.sha256()
    for path in [
        *rtl_sources(),
        ROOT / "tb/tb_snackwell_kernel.sv",
        ROOT / "sim/snackwell_sim_top.sv",
        ROOT / "sim/session.cpp",
    ]:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def refresh_model_runner(simulator, output):
    """Reload changed RTL between prompts; never switch binaries mid-prediction."""
    if not isinstance(simulator, ModelSimulator):
        return simulator
    if simulator.source_sha256 == rtl_source_digest():
        return simulator
    return compile_model_runner(output, simulator.backend, simulator.trace)


def compile_model_runner(output: Path, backend="auto", trace=False) -> ModelSimulator:
    """Prefer compiled RTL for model inference; keep the Icarus regression path explicit."""
    source_sha256 = rtl_source_digest()
    session_executable = None
    if backend == "auto":
        backend = (
            "verilator" if shutil.which(os.environ.get("VERILATOR", "verilator")) else "icarus"
        )
    if backend == "icarus":
        executable = compile_runner(output)
    elif backend == "verilator":
        executable = compile_verilator_runner()
        session_executable = compile_verilator_runner(session=True)
    else:
        raise ValueError(f"Unknown simulator: {backend}")
    if source_sha256 != rtl_source_digest():
        raise RuntimeError("RTL changed during compilation; retry with a stable source tree")
    return ModelSimulator(executable, backend, trace, source_sha256, session_executable)


def compile_verilator_runner(*, session=False) -> Path:
    """Cache compiled RTL, with either the timed SV bench or the untimed C++ host."""
    import fcntl

    compiler = tool_path("verilator")
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    cxx = shutil.which(os.environ.get("CXX", "c++"))
    if cxx is None:
        raise RuntimeError("Verilator requires a C++ compiler on PATH")
    cxx_version = subprocess.check_output([cxx, "--version"], text=True).strip()
    host_sources = (
        [ROOT / "sim/snackwell_sim_top.sv", ROOT / "sim/session.cpp"]
        if session
        else [ROOT / "tb/tb_snackwell_kernel.sv"]
    )
    sources = [*rtl_sources(), *host_sources]
    flags = [
        *(["--cc", "--exe", "--build"] if session else ["--binary", "--timing"]),
        "--assert",
        "-CFLAGS",
        "-O3 -std=c++17",
        "--top-module",
        "snackwell_sim_top" if session else "tb_snackwell_kernel",
        "-Wno-fatal",
    ]
    identity = dict(
        verilator=compiler,
        version=version,
        cxx=cxx,
        cxx_version=cxx_version,
        flags=flags,
        environment={key: os.environ.get(key) for key in ("CC", "CXX", "CXXFLAGS", "LDFLAGS")},
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for source in sources:
        digest.update(str(source.relative_to(ROOT)).encode() + b"\0" + source.read_bytes() + b"\0")
    output = ROOT / "build/simulators/verilator" / digest.hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    executable, manifest = output / "snackwell_sim", output / "build.json"
    # CLI invocations and browser sessions may start at the same time.
    with (output / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if executable.is_file() and manifest.is_file():
            return executable
        checked_command(
            [
                compiler,
                *flags,
                "--Mdir",
                str(output),
                "-o",
                executable.name,
                "-j",
                str(min(4, os.cpu_count() or 1)),
                *map(str, sources),
            ],
            ROOT,
            output / "compile.log",
            timeout=300,
        )
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
    return executable


def write_hex(path, words):
    encoded = struct.pack(f">{len(words)}I", *words).hex("\n", 4)
    path.write_text(encoded + ("\n" if words else ""))


def read_hex(path, count):
    tokens = re.sub(r"//[^\n]*", "", path.read_text()).split()
    if len(tokens) != count or any(len(word) != 8 for word in tokens):
        raise ValueError(f"{path}: expected {count} known 32-bit words")
    try:
        # Decode the whole image in C, retaining strict word widths and rejection
        # of X/Z digits, signs, underscores, truncation and surplus words.
        return list(struct.unpack(f">{count}I", bytes.fromhex("".join(tokens))))
    except (ValueError, struct.error) as error:
        raise ValueError(f"{path}: expected {count} known 32-bit words") from error


def checked_command(args, cwd, log_path, timeout=30):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    log = result.stdout + result.stderr
    log_path.write_text(log)
    if result.returncode or re.search(
        r"\b(FAIL|FATAL|ERROR)\b|implicit definition of wire|dangling input port", log
    ):
        raise RuntimeError(f"Command failed; see {log_path}\n{log[-2000:]}")
    return log


def tool_path(name: str) -> str:
    """Find a simulator tool, allowing an uppercase environment-variable override."""
    command = os.environ.get(name.upper(), name)
    executable = shutil.which(command)
    if executable is None:
        package = "Verilator" if name == "verilator" else "Icarus Verilog"
        raise RuntimeError(f"{command!r} was not found. Install {package} and add it to PATH.")
    return executable


def compile_testbench(bench: Path, output: Path, *, defines=()) -> Path:
    """Compile one top-level testbench with the same source list and checks as kernels."""
    output.mkdir(parents=True, exist_ok=True)
    executable = (output / "sim.vvp").resolve()
    checked_command(
        [
            tool_path("iverilog"),
            "-g2012",
            "-Wall",
            *(f"-D{define}" for define in defines),
            "-s",
            bench.stem,
            "-o",
            str(executable),
            *map(str, rtl_sources()),
            str(bench.resolve()),
        ],
        ROOT,
        output / "compile.log",
    )
    return executable


def compile_runner(output: Path) -> Path:
    return compile_testbench(ROOT / "tb/tb_snackwell_kernel.sv", output)


def run_program(
    executable, words, memory, output, active_masks=0xFFFF, entry=0, warp_mask=0xF, max_cycles=10000
):
    """Execute an assembled program; no arithmetic is emulated here."""
    simulator = (
        executable
        if isinstance(executable, (ModelSimulator, SimulatorSession))
        else ModelSimulator(executable, "icarus", True)
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(memory) != SHARED_WORDS or any(
        type(word) is not int or not 0 <= word < 2**32 for word in memory
    ):
        raise ValueError("shared-memory image must have 1024 unsigned 32-bit words")
    if not 0 <= active_masks <= 0xFFFF or not 0 <= entry < 2**32 or not 1 <= warp_mask <= 15:
        raise ValueError("invalid launch masks or entry address")
    if type(max_cycles) is not int or not 1 <= max_cycles <= 1000000:
        raise ValueError("cycle limit must be 1 through 1000000")
    program_path, data_path = output / "program.hex", output / "input.hex"
    result_path, register_path = output / "output.hex", output / "registers.hex"
    # Remove outputs from earlier runs so a failed simulation cannot look successful.
    for name in ("output.hex", "registers.hex", "trace.csv", "profile.json"):
        (output / name).unlink(missing_ok=True)
    write_image(program_path, words)
    write_hex(data_path, memory)
    command = [str(simulator.executable.resolve())]
    if simulator.backend == "icarus":
        command.insert(0, tool_path("vvp"))
    command.extend(
        [
            f"+PROGRAM={program_path}",
            f"+DATA={data_path}",
            f"+RESULT={result_path}",
            f"+REGISTERS={register_path}",
            f"+ACTIVE_MASKS={active_masks:04x}",
            f"+ENTRY={entry}",
            f"+WARP_MASK={warp_mask:x}",
            f"+MAX_CYCLES={max_cycles}",
        ]
    )
    if simulator.trace:
        command.append(f"+TRACE={output / 'trace.csv'}")
    log = (
        simulator.run(output, active_masks, warp_mask, entry, max_cycles)
        if isinstance(simulator, SimulatorSession)
        else checked_command(command, output, output / "simulation.log")
    )
    match = re.search(r"PASS kernel completed in (\d+) cycles", log)
    if not match:
        raise RuntimeError("simulation did not report successful kernel completion")
    cycles = int(match[1])
    profile = (
        summarize_trace(output / "trace.csv", cycles) if simulator.trace else {"cycles": cycles}
    )
    profile.update(simulator=simulator.backend, trace_enabled=simulator.trace)
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return read_hex(result_path, SHARED_WORDS), read_hex(register_path, 256), cycles
