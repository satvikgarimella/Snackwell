"""Interactive character prediction with progress from the actual RTL launches."""

import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from snackwell.character_model import load_checkpoint, model_description
from snackwell.decoding import DEFAULT_TEMPERATURE, Decoder, validate_temperature
from snackwell.kv_cache import Generation, latency_report
from snackwell.simulator import refresh_model_runner


class TerminalPrediction:
    def __init__(self, prompt, tokens, stream=None, unit="character"):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.generated, self.tokens = prompt, tokens
        self.unit = unit
        self.step = self.cycles = self.launches = 0
        self.started = time.monotonic()
        self.last_draw = 0.0
        self.last_stage = None
        self.status = "Preparing the forward pass"
        self.draw(force=True)

    def draw(self, force=False):
        if not self.tty:
            if force:
                print(f"Text: {self.generated!r}\n{self.status}", file=self.stream, flush=True)
            return
        now = time.monotonic()
        if not force and now - self.last_draw < 0.1:
            return
        self.last_draw = now
        width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        text = f"Text: {self.generated!r}"
        if len(text) > width:
            text = "Text: ..." + text[-(width - 9) :]
        detail = f"[{self.step}/{self.tokens}] {self.status} | {now - self.started:.1f}s"
        # Both lines stay within the terminal width; leave the cursor on the first.
        self.stream.write(f"\r\033[2K{text}\n\033[2K{detail[:width]}\033[1A\r")
        self.stream.flush()

    def progress(self, event):
        if event["type"] == "kernel_start":
            self.status = f"{event['stage']} | launch {event['launch']} | {self.cycles:,} cycles"
            if not self.tty and event["stage"] != self.last_stage:
                print(f"  {self.unit} {self.step}: {event['stage']}", file=self.stream, flush=True)
            self.last_stage = event["stage"]
        elif event["type"] == "kernel_complete":
            self.cycles += event["cycles"]
            self.launches += 1
        elif event["type"] == "cache":
            self.status = f"{event['phase']} | {event['context_tokens']} context {self.unit}s"
            if not self.tty:
                print(f"  {self.unit} {self.step}: {self.status}", file=self.stream, flush=True)
        self.draw()

    def prediction(self, result):
        self.generated += result["next_character"]
        self.status = f"predicted {result['next_character']!r} | {self.cycles:,} cycles"
        self.draw(force=True)

    def finish(self, status):
        self.status = status
        self.draw(force=True)
        if self.tty:
            self.stream.write("\n\n")
            self.stream.flush()


def predict(model, prompt, tokens, output, executable, stream=None, decoder=None, cache=True):
    """Show and save a run; publish characters only after whole-model verification."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    generation = Generation(model, prompt, cache=cache)
    decoder = decoder or Decoder()
    print(
        f"Model: {model_description(model)}\nDecoding: {decoder.describe()}",
        file=stream or sys.stdout,
        flush=True,
    )
    if getattr(executable, "source_sha256", None):
        print(
            f"RTL build: {executable.source_sha256[:12]} ({executable.backend})",
            file=stream or sys.stdout,
            flush=True,
        )
    unit = "token" if "tokenizer" in model else "character"
    display = TerminalPrediction(prompt, tokens, stream, unit)
    reports = []
    try:
        for step in range(tokens):
            display.step = step + 1
            result = generation.next(
                output / f"token_{step:03d}",
                executable,
                display.progress,
                draw=decoder.draw(),
                temperature=decoder.temperature,
            )
            display.prediction(result)
            reports.append(
                dict(
                    character=result["next_character"],
                    cycles=result["cycles"],
                    launches=result["launches"],
                    accuracy=result["accuracy"],
                    wall_seconds=result["wall_seconds"],
                    simulation=result["simulation"],
                    rtl_source_sha256=result.get("rtl_source_sha256"),
                    selection=result["selection"],
                    cache=result["cache"],
                )
            )
        report = dict(
            verified=True,
            checkpoint_sha256=model.get("sha256"),
            prompt=prompt,
            generated=display.generated,
            decoding=decoder.describe(),
            tokens=reports,
            cycles=sum(r["cycles"] for r in reports),
            latency=latency_report(reports, cache, subword="tokenizer" in model),
            tokenization="subword" if "tokenizer" in model else "character",
        )
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        display.finish("Stopped before completion; only verified predictions are shown")
        raise
    display.finish(
        f"Done | {display.cycles:,} cycles | {display.launches} launches | RTL/reference match"
    )
    latency = report["latency"]
    timing = f"First {unit}: {latency['first_token_seconds']:.2f}s"
    if latency["subsequent_token_seconds"] is not None:
        timing += f"; subsequent: {latency['subsequent_token_seconds']:.2f}s/{unit}"
    print(timing + " (verified RTL simulation)", file=stream or sys.stdout)
    if cache:
        details = []
        if latency["decode_steps"]:
            details.append(f"cached decode: {latency['decode_seconds']:.2f}s/{unit}")
        if latency["rebuild_steps"]:
            details.append(
                f"context rebuild: {latency['rebuild_seconds']:.2f}s "
                f"({latency['rebuild_steps']} rebuilds)"
            )
        if details:
            print("; ".join(details), file=stream or sys.stdout)
    if unit == "token":
        characters = latency["generated_characters"]
        print(
            f"Generated {characters} character{'s' if characters != 1 else ''} "
            f"in {tokens} token{'s' if tokens != 1 else ''}; "
            f"{latency['tokens_per_second']:.2f} tokens/s, "
            f"{latency['characters_per_second']:.2f} characters/s (including prefill/rebuilds)",
            file=stream or sys.stdout,
        )
    return report


def interactive(
    model,
    output,
    executable,
    tokens=4,
    mode="sample",
    seed=None,
    checkpoint_path=None,
    temperature=DEFAULT_TEMPERATURE,
    cache=True,
):
    unit = "token" if "tokenizer" in model else "character"
    print(
        f"\nSnackwell — live RTL {unit} prediction\n"
        f"Context: {model['config']['context']} {unit}s. New {unit}s per prompt: {tokens}.\n"
        "Enter text and press Enter. /tokens N changes the length; /quit exits.\n"
        "/temperature T adjusts sampling; /seed N repeats a random sequence.\n"
        f"Ctrl+C stops a prediction. Each {unit} runs through the full SM RTL.\n"
        f"Decoding: {mode}. Use --decode greedy for the most likely {unit} each time.\n"
        + (
            "KV cache enabled; context is rebuilt periodically when full.\n"
            if cache
            else f"KV cache disabled; the sliding context is recomputed for every {unit}.\n"
        )
    )
    while True:
        try:
            prompt = input("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip() == "/quit":
            return
        if prompt.startswith("/tokens "):
            try:
                count = int(prompt.split(maxsplit=1)[1])
                if not 1 <= count <= 256:
                    raise ValueError
                tokens = count
                print(f"Will predict {tokens} new {unit}s per prompt.\n")
            except ValueError:
                print("Use /tokens N, with N from 1 to 256.\n")
            continue
        if prompt.startswith("/temperature "):
            try:
                temperature = validate_temperature(float(prompt.split(maxsplit=1)[1]))
                print(f"Sampling temperature: {temperature}.\n")
            except ValueError as error:
                print(f"{error}\n")
            continue
        if prompt.startswith("/seed "):
            try:
                new_seed = int(prompt.split(maxsplit=1)[1])
                Decoder(mode, new_seed, temperature)
                seed = new_seed
                print(f"Sampling seed: {seed}.\n")
            except ValueError:
                print("Use /seed N, with N from 0 to 4294967295.\n")
            continue
        if not prompt:
            continue
        run_output = output / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
        try:
            updated_simulator = refresh_model_runner(executable, output / "simulator")
            if updated_simulator is not executable:
                print(f"Reloaded RTL simulator: {updated_simulator.source_sha256[:12]}")
                executable = updated_simulator
            # Freeze one checkpoint for the whole prediction, but do not keep
            # old weights indefinitely in a long-lived interactive session.
            if checkpoint_path is not None:
                updated = load_checkpoint(checkpoint_path)
                if updated["sha256"] != model.get("sha256"):
                    print(f"Reloaded checkpoint: {model_description(updated)}")
                model = updated
                unit = "token" if "tokenizer" in model else "character"
            predict(
                model,
                prompt,
                tokens,
                run_output,
                executable,
                decoder=Decoder(mode, seed, temperature),
                cache=cache,
            )
            print(f"Run files: {run_output}\n")
        except KeyboardInterrupt:
            print("Prediction cancelled. Enter another prompt, or /quit.\n")
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"Prediction failed: {error}\n")
