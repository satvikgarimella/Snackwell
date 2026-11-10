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

from snackwell.assembler import EXIT_WORD, write_image
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
    compact: bool = False


class SimulatorSession:
    """Run separate, reset RTL launches in one Verilator process.

    Programs, raw memory words and launch controls cross this interface. The C++
    host clocks the RTL directly; traces use the original SystemVerilog testbench.
    """

    def __init__(self, simulator):
        if simulator.backend != "verilator":
            raise ValueError("persistent sessions require Verilator")
        self.executable = (
            simulator.session_executable if not simulator.trace else None
        ) or simulator.executable
        self.traced_executable = simulator.executable
        self.direct_executable = simulator.session_executable
        self.backend, self.trace = simulator.backend, simulator.trace
        self.source_sha256 = simulator.source_sha256
        self.compact = simulator.compact and not simulator.trace
        self.process = None
        self.process_mode = None
        self.pending = b""
        self.resident = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.resident is not None:
            self.resident.invalidate()
            self.resident = None
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
            self.process_mode = None

    def _start(self, mode):
        if self.process is not None and self.process_mode != mode:
            self.close()
        if self.process is None:
            self.process = subprocess.Popen(
                [str(self.executable.resolve()), mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            self.process_mode = mode

    def run_words(self, words, memory, active_masks, warp_mask, entry, max_cycles, timeout=30):
        """Exchange fixed-size little-endian frames; execute all arithmetic in RTL."""
        if self.trace or self.direct_executable is None:
            raise ValueError("word transport requires the untraced direct-clock driver")
        try:
            self.executable = self.direct_executable
            self._start("+WORDS")
            request = struct.pack(
                "<2053I",
                0x534E5731,
                active_masks,
                warp_mask,
                entry,
                max_cycles,
                *words,
                *([EXIT_WORD] * (1024 - len(words))),
                *memory,
            )
            # Raw pipe writes may be partial, even for these fixed-size frames.
            view = memoryview(request)
            while view:
                written = self.process.stdin.write(view)
                if not written:
                    raise RuntimeError("RTL word transport closed while writing")
                view = view[written:]
            deadline = time.monotonic() + timeout
            response = bytearray()
            expected_bytes = 1282 * 4
            while len(response) < expected_bytes:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise subprocess.TimeoutExpired(str(self.executable), timeout)
                data = os.read(self.process.stdout.fileno(), expected_bytes - len(response))
                if not data:
                    raise RuntimeError(
                        "RTL word transport ended before completion: "
                        + response.decode(errors="replace")[-2000:]
                    )
                response.extend(data)
            result = struct.unpack("<1282I", response)
            if result[0] != 0x534E5731 or not 0 < result[1] <= max_cycles:
                raise RuntimeError("Invalid RTL word transport response")
            return list(result[2:1026]), list(result[1026:]), result[1]
        except BaseException:
            self.close()
            raise

    def run_resident_words(
        self,
        words,
        shared,
        uploads,
        readback,
        writable,
        *,
        active_masks=0xFFFF,
        warp_mask=1,
        latency=2,
        max_cycles=200000,
        timeout=30,
    ):
        """Update named external ranges, clock RTL, read named ranges and check all guards.

        External RAM survives launches. The driver checks every word outside
        writable ranges without transferring the whole RAM image to Python.
        """
        from snackwell.resident_memory import CAPACITY

        if self.trace or self.direct_executable is None:
            raise ValueError("resident memory requires untraced direct-clock Verilator")

        def valid_word(word):
            return type(word) is int and 0 <= word < 2**32

        if (
            not 1 <= len(words) <= 1024
            or len(shared) != 1024
            or not all(valid_word(w) for w in [*words, *shared])
            or type(max_cycles) is not int
            or not 1 <= max_cycles <= 1000000
            or type(latency) is not int
            or not 0 <= latency <= 1000
            or not 0 <= active_masks <= 65535
            or not 1 <= warp_mask <= 15
        ):
            raise ValueError("invalid resident launch")
        ranges = [*((start, len(data)) for start, data in uploads), *readback, *writable]
        if any(
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 1
            or start + count > CAPACITY
            for start, count in ranges
        ):
            raise ValueError("invalid resident range")
        if sum(len(data) for _, data in uploads) > CAPACITY or any(
            not valid_word(w) for _, data in uploads for w in data
        ):
            raise ValueError("invalid resident upload")
        request = [
            0x534E5231,
            active_masks,
            warp_mask,
            0,
            max_cycles,
            latency,
            *words,
            *([EXIT_WORD] * (1024 - len(words))),
            *shared,
            len(uploads),
        ]
        for start, data in uploads:
            request.extend((start, len(data), *data))
        for spans in (readback, writable):
            request.append(len(spans))
            request.extend(x for span in spans for x in span)
        count = 2 + 1024 + 256 + sum(n for _, n in readback) + 11
        try:
            self.executable = self.direct_executable
            self._start("+RESIDENT")
            view = memoryview(struct.pack(f"<{len(request)}I", *request))
            while view:
                written = self.process.stdin.write(view)
                if not written:
                    raise RuntimeError("RTL resident transport closed while writing")
                view = view[written:]
            deadline, response = time.monotonic() + timeout, bytearray()
            while len(response) < count * 4:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise subprocess.TimeoutExpired(str(self.executable), timeout)
                data = os.read(self.process.stdout.fileno(), count * 4 - len(response))
                if not data:
                    raise RuntimeError(
                        "RTL resident transport failed: "
                        + response.decode(errors="replace")[-2000:]
                    )
                response.extend(data)
            result = struct.unpack(f"<{count}I", response)
            if result[0] != 0x534E5231 or not 0 < result[1] <= max_cycles:
                raise RuntimeError("invalid resident response")
            keys = (
                "reads",
                "writes",
                "responses",
                "request_stalls",
                "status",
                "copy_status",
                "copy_reads",
                "copy_writes",
                "copy_arithmetic_overlap_cycles",
                "nvfp4_macs",
                "fp32_tensor_multiplies",
            )
            metrics = dict(
                zip(keys, result[-11:], strict=True),
                cycles=result[1],
                host_upload_bytes=4 * sum(len(data) for _, data in uploads),
                host_readback_bytes=4 * sum(n for _, n in readback),
                protocol_request_bytes=4 * len(request),
                protocol_response_bytes=4 * count,
            )
            if metrics["reads"] + metrics["writes"] != metrics["responses"]:
                raise RuntimeError("resident requests did not drain")
            if metrics["status"] or metrics["copy_status"]:
                raise ValueError("resident global/copy status reported an error")
            return list(result[2:1026]), list(result[1026:1282]), list(result[1282:-11]), metrics
        except BaseException:
            self.close()
            raise

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
            self._start("+SESSION")
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

    def run_global_words(
        self, words, shared, memory, active_masks, warp_mask, latency, max_cycles, timeout=30
    ):
        """Clock the external RAM handshake; no shared operand data is synthesized here."""
        if self.trace or self.direct_executable is None:
            raise ValueError("global word transport requires the untraced direct-clock driver")
        try:
            self.executable = self.direct_executable
            self._start("+GLOBAL")
            request = [
                0x534E4731,
                active_masks,
                warp_mask,
                0,
                max_cycles,
                latency,
                *words,
                *([EXIT_WORD] * (1024 - len(words))),
                *shared,
                *memory,
            ]
            view = memoryview(struct.pack(f"<{len(request)}I", *request))
            while view:
                written = self.process.stdin.write(view)
                if not written:
                    raise RuntimeError("RTL global transport closed while writing")
                view = view[written:]
            deadline = time.monotonic() + timeout
            response = bytearray()
            count = 2 + 1024 + 256 + 16384 + 11
            while len(response) < count * 4:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise subprocess.TimeoutExpired(str(self.executable), timeout)
                data = os.read(self.process.stdout.fileno(), count * 4 - len(response))
                if not data:
                    raise RuntimeError(
                        "RTL global transport ended before completion: "
                        + response.decode(errors="replace")[-2000:]
                    )
                response.extend(data)
            result = struct.unpack(f"<{count}I", response)
            if result[0] != 0x534E4731 or not 0 < result[1] <= max_cycles:
                raise RuntimeError("Invalid RTL global transport response")
            keys = (
                "reads",
                "writes",
                "responses",
                "request_stalls",
                "status",
                "copy_status",
                "copy_reads",
                "copy_writes",
                "copy_arithmetic_overlap_cycles",
                "nvfp4_macs",
                "fp32_tensor_multiplies",
            )
            metrics = dict(zip(keys, result[-11:], strict=True), cycles=result[1])
            if metrics["reads"] + metrics["writes"] != metrics["responses"]:
                raise RuntimeError("External requests did not drain")
            return list(result[2:1026]), list(result[1026:1282]), list(result[1282:-11]), metrics
        except BaseException:
            self.close()
            raise


def rtl_source_digest():
    """Identify the RTL and testbench loaded by a long-lived model runner."""
    digest = hashlib.sha256()
    for path in [
        *rtl_sources(),
        ROOT / "tb/tb_snackwell_kernel.sv",
        ROOT / "sim/snackwell_sim_top.sv",
        ROOT / "sim/session.cpp",
        ROOT / "tb/support/global_memory_model.sv",
        ROOT / "tb/tb_snackwell_global_memory.sv",
    ]:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def refresh_model_runner(simulator, output):
    """Reload changed RTL between prompts; never switch binaries mid-prediction."""
    if not isinstance(simulator, ModelSimulator):
        return simulator
    if simulator.source_sha256 == rtl_source_digest():
        return simulator
    return compile_model_runner(output, simulator.backend, simulator.trace, simulator.compact)


def compile_model_runner(
    output: Path, backend="auto", trace=False, compact=False
) -> ModelSimulator:
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
    return ModelSimulator(executable, backend, trace, source_sha256, session_executable, compact)


def compile_verilator_runner(*, session=False, defines=()) -> Path:
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
        *(f"-D{define}" for define in defines),
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
    for source in [*sources, *([ROOT / "tb/support/global_memory_model.sv"] if session else [])]:
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
    if result.returncode:
        log += f"\nexit_code={result.returncode}\n"
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
    if len(memory) != SHARED_WORDS or any(
        type(word) is not int or not 0 <= word < 2**32 for word in memory
    ):
        raise ValueError("shared-memory image must have 1024 unsigned 32-bit words")
    if not 0 <= active_masks <= 0xFFFF or not 0 <= entry < 2**32 or not 1 <= warp_mask <= 15:
        raise ValueError("invalid launch masks or entry address")
    if type(max_cycles) is not int or not 1 <= max_cycles <= 1000000:
        raise ValueError("cycle limit must be 1 through 1000000")
    if not 1 <= len(words) <= 1024 or any(
        type(word) is not int or not 0 <= word < 2**32 for word in words
    ):
        raise ValueError("program must contain 1 through 1024 unsigned 32-bit words")
    # A directory reused after a detailed run must not retain stale success files.
    if output.exists():
        for name in ("output.hex", "registers.hex", "trace.csv", "profile.json"):
            (output / name).unlink(missing_ok=True)
    if isinstance(simulator, SimulatorSession) and simulator.compact and not simulator.trace:
        return simulator.run_words(words, memory, active_masks, warp_mask, entry, max_cycles)
    output.mkdir(parents=True, exist_ok=True)
    program_path, data_path = output / "program.hex", output / "input.hex"
    result_path, register_path = output / "output.hex", output / "registers.hex"
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
