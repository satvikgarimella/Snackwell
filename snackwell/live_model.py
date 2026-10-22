"""Local browser demo streaming actual RTL launch progress and character predictions."""

import json
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from snackwell.character_model import encode_prompt, load_checkpoint
from snackwell.decoding import DEFAULT_TEMPERATURE, Decoder
from snackwell.kv_cache import Generation, latency_report
from snackwell.simulator import compile_model_runner, refresh_model_runner


class PredictionStopped(Exception):
    """The user stopped a run between simulation launches."""


class LiveModelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        model,
        output,
        prompt="To be, or not to ",
        tokens=4,
        port=8768,
        executable=None,
        mode="sample",
        seed=None,
        checkpoint_path=None,
        temperature=DEFAULT_TEMPERATURE,
        cache=True,
    ):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.executable = executable or compile_model_runner(self.output)
        self.mode, self.seed = mode, seed
        self.temperature = Decoder(mode, seed, temperature).temperature
        self.prompt, self.tokens = prompt, tokens
        self.cache = cache
        self.run_lock = threading.Lock()
        self.cancel = threading.Event()
        self.runs = {}
        super().__init__(("127.0.0.1", port), LiveModelHandler)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def defaults(self):
        return dict(
            prompt=self.prompt,
            tokens=self.tokens,
            decoding=self.mode,
            temperature=self.temperature,
            kv_cache=self.cache,
            checkpoint_sha256=self.model.get("sha256"),
            **self.model["config"],
        )


class LiveModelHandler(BaseHTTPRequestHandler):
    server: LiveModelServer

    def log_message(self, *args):
        pass

    def reply(self, status, payload, content_type="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def local_request(self):
        # Bind and accept only this loopback origin, including for cancellation.
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != host or self.headers.get("Origin") not in (
            None,
            self.server.url,
        ):
            self.reply(403, dict(error="Open the demo using its printed local URL."))
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        if self.path == "/":
            self.reply(
                200,
                Path(__file__).with_name("live_model.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        elif self.path == "/config":
            self.reply(200, self.server.defaults)
        elif self.path.startswith("/events?"):
            params = parse_qs(urlsplit(self.path).query)
            try:
                run = self.server.runs[params["run"][0]]
                after = int(params.get("after", ["0"])[0])
                self.reply(200, run.snapshot(after))
            except (KeyError, ValueError) as error:
                self.reply(400, dict(error=f"Invalid progress request: {error}"))
        else:
            self.reply(404, dict(error="Not found"))

    def do_POST(self):
        if not self.local_request():
            return
        if self.path == "/stop":
            self.server.cancel.set()
            self.reply(200, dict(stopping=True))
            return
        if self.path != "/generate":
            self.reply(404, dict(error="Not found"))
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON request")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32768:
                raise ValueError("Request is empty or too large")
            request = json.loads(self.rfile.read(size))
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object")
            prompt, tokens = request.get("prompt"), request.get("tokens")
            if not isinstance(prompt, str) or len(prompt) > 4096:
                raise ValueError("Prompt must contain 1..4096 characters")
            if type(tokens) is not int or not 1 <= tokens <= 256:
                raise ValueError("Choose 1..256 new characters")
        except (ValueError, UnicodeError) as error:
            self.reply(400, dict(error=str(error)))
            return
        if not self.server.run_lock.acquire(blocking=False):
            self.reply(
                409,
                dict(error="A prediction is already running. Stop it or wait for it to finish."),
            )
            return
        try:
            # The lock keeps this model fixed for the entire run. A changed or
            # invalid file must never silently fall back to an old checkpoint.
            if self.server.checkpoint_path is not None:
                model = load_checkpoint(self.server.checkpoint_path)
            else:
                model = self.server.model
            encode_prompt(model, prompt, pad=not self.server.cache)
            self.server.model = model
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            self.server.run_lock.release()
            self.reply(400, dict(error=f"Could not load prompt/checkpoint: {error}"))
            return
        self.server.cancel.clear()
        run = LivePrediction(self.server, prompt, tokens)
        # Keep recent completed runs available for a final browser poll.
        for run_id in list(self.server.runs)[:-7]:
            del self.server.runs[run_id]
        self.server.runs[run.id] = run
        threading.Thread(target=run.work, daemon=True).start()
        self.reply(202, dict(run=run.id))


class LivePrediction:
    def __init__(self, server, prompt, tokens):
        self.server = server
        self.prompt, self.tokens = prompt, tokens
        self.id = uuid.uuid4().hex
        self.events = []
        self.event_lock = threading.Lock()
        self.finished = False

    def work(self):
        try:
            self.generate(self.prompt, self.tokens)
        finally:
            self.server.run_lock.release()
            with self.event_lock:
                self.finished = True

    def event(self, event):
        with self.event_lock:
            self.events.append(event)

    def snapshot(self, after):
        with self.event_lock:
            if not 0 <= after <= len(self.events):
                raise ValueError("Progress cursor is outside this run")
            return dict(events=self.events[after:], cursor=len(self.events), finished=self.finished)

    def generate(self, prompt, tokens):
        output = self.server.output / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
        generated, reports, cycles, launches = prompt, [], 0, 0
        started = time.monotonic()
        decoder = Decoder(self.server.mode, self.server.seed, self.server.temperature)

        def progress(event):
            if self.server.cancel.is_set():
                raise PredictionStopped
            self.event(event)

        try:
            output.mkdir()
            self.server.executable = refresh_model_runner(
                self.server.executable, self.server.output / "simulator"
            )
            self.event(
                dict(
                    type="run_start",
                    output=str(output),
                    decoding=decoder.describe(),
                    model=self.server.defaults,
                )
            )
            model = self.server.model
            vocabulary = model["config"]["vocabulary"]
            generation = Generation(model, prompt, cache=self.server.cache)
            for step in range(tokens):
                ids = generation.ids
                if len(ids) > generation.context:
                    ids = ids[-generation.rebuild_tokens :]
                progress(
                    dict(type="token_start", step=step + 1, context=[vocabulary[i] for i in ids])
                )
                result = generation.next(
                    output / f"token_{step:03d}",
                    self.server.executable,
                    progress,
                    draw=decoder.draw(),
                    temperature=decoder.temperature,
                )
                if self.server.cancel.is_set():
                    raise PredictionStopped
                generated += result["next_character"]
                cycles += result["cycles"]
                launches += result["launches"]
                probabilities = (
                    result["sampling_probabilities"] or result["probabilities"][-len(vocabulary) :]
                )
                top = sorted(range(len(vocabulary)), key=lambda i: (-probabilities[i], i))[:5]
                report = dict(
                    step=step + 1,
                    character=result["next_character"],
                    generated=generated,
                    cycles=result["cycles"],
                    launches=result["launches"],
                    total_cycles=cycles,
                    total_launches=launches,
                    elapsed=time.monotonic() - started,
                    candidates=[
                        dict(character=vocabulary[i], probability=probabilities[i]) for i in top
                    ],
                    verified=result["verified"],
                    selection=result["selection"],
                    cache=result["cache"],
                    wall_seconds=result["wall_seconds"],
                )
                reports.append(report)
                self.event(dict(type="prediction", **report))
            summary = dict(
                verified=True,
                checkpoint_sha256=model["sha256"],
                prompt=prompt,
                generated=generated,
                decoding=decoder.describe(),
                cycles=cycles,
                tokens=reports,
                latency=latency_report(reports, self.server.cache),
            )
            (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.event(dict(type="done", generated=generated, output=str(output)))
        except PredictionStopped:
            self.event(dict(type="stopped", generated=generated, output=str(output)))
        except Exception as error:
            # Stream failures explicitly; a partial run never gets a success report.
            self.event(dict(type="error", message=str(error)))


def serve(
    model,
    output,
    prompt,
    tokens,
    port,
    open_browser,
    executable=None,
    mode="sample",
    seed=None,
    checkpoint_path=None,
    temperature=DEFAULT_TEMPERATURE,
    cache=True,
):
    if not 0 <= port <= 65535:
        raise ValueError("--port must be 0..65535")
    with LiveModelServer(
        model,
        output,
        prompt,
        tokens,
        port,
        executable,
        mode,
        seed,
        checkpoint_path,
        temperature,
        cache,
    ) as server:
        print(f"Snackwell live predictions: {server.url}", flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop the server.", flush=True)
        if open_browser:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.cancel.set()
            print("\nLive demo stopped.")
