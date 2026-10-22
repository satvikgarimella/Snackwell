import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from snackwell.character_model import load_checkpoint
from snackwell.live_model import LiveModelServer
from snackwell.simulator import compile_model_runner
from tests.test_model import fixture


class LiveModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        model = fixture()
        model["sha256"] = "test-fixture"
        executable = compile_model_runner(Path(cls.temporary.name), backend="icarus")
        cls.server = LiveModelServer(
            model,
            cls.temporary.name,
            "abcd",
            1,
            port=0,
            executable=executable,
            mode="sample",
            seed=1729,
            temperature=0.7,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temporary.cleanup()

    def request(self, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=30)
        connection.request(
            "POST" if payload is not None else "GET",
            path,
            json.dumps(payload) if payload is not None else None,
            headers or {"Content-Type": "application/json"},
        )
        return connection, connection.getresponse()

    def test_streams_actual_rtl_prediction_and_saves_result(self):
        connection, response = self.request("/generate", dict(prompt="abcd", tokens=1))
        self.assertEqual(response.status, 202)
        run = json.loads(response.read())["run"]
        connection.close()
        events = self.wait_for_run(run)
        self.assertEqual(events[-1]["type"], "done")
        prediction = next(event for event in events if event["type"] == "prediction")
        kernels = [event for event in events if event["type"] == "kernel_complete"]
        self.assertEqual(prediction["cycles"], sum(event["cycles"] for event in kernels))
        self.assertEqual(prediction["launches"], len(kernels))
        self.assertGreater(len(kernels), 10)
        output = Path(events[-1]["output"])
        saved = json.loads((output / "result.json").read_text())
        token = json.loads((output / "token_000/result.json").read_text())
        self.assertEqual(saved["generated"], "abcd" + token["next_character"])
        self.assertEqual(saved["decoding"], dict(mode="sample", seed=1729, temperature=0.7))
        probabilities = token["sampling_probabilities"]
        vocabulary = self.server.model["config"]["vocabulary"]
        top = sorted(range(len(vocabulary)), key=lambda i: (-probabilities[i], i))[:5]
        self.assertEqual(
            prediction["candidates"],
            [dict(character=vocabulary[i], probability=probabilities[i]) for i in top],
        )
        started = next(event for event in events if event["type"] == "run_start")
        self.assertEqual(started["model"]["temperature"], 0.7)
        self.assertAlmostEqual(sum(c["probability"] for c in prediction["candidates"]), 1, places=5)
        kinds = [event["type"] for event in events]
        self.assertLess(kinds.index("attention"), kinds.index("prediction"))

    def test_cancel_and_busy_request_do_not_report_success(self):
        connection, response = self.request("/generate", dict(prompt="abcd", tokens=256))
        self.assertEqual(response.status, 202)
        run = json.loads(response.read())["run"]
        connection.close()
        other, busy = self.request("/generate", dict(prompt="abcd", tokens=1))
        self.assertEqual(busy.status, 409)
        busy.read()
        other.close()
        stopper, stopped = self.request("/stop", {})
        self.assertEqual(stopped.status, 200)
        stopped.read()
        stopper.close()
        events = self.wait_for_run(run)
        self.assertEqual(events[-1]["type"], "stopped")
        self.assertFalse((Path(events[0]["output"]) / "result.json").exists())
        # The next request can use the model again after cancellation.
        with self.server.run_lock:
            pass

    def wait_for_run(self, run):
        events, cursor = [], 0
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            connection, response = self.request(f"/events?run={run}&after={cursor}")
            self.assertEqual(response.status, 200)
            progress = json.loads(response.read())
            connection.close()
            events.extend(progress["events"])
            cursor = progress["cursor"]
            self.assertEqual(cursor, len(events))
            if progress["finished"]:
                # Progress can be resumed without losing or duplicating events.
                connection, response = self.request(f"/events?run={run}&after={cursor}")
                self.assertEqual(json.loads(response.read())["events"], [])
                connection.close()
                return events
            time.sleep(0.05)
        self.fail("Live inference timed out")

    def test_rejects_invalid_input_and_foreign_origin(self):
        for payload in (dict(prompt="unknown", tokens=1), dict(prompt="abcd", tokens=0), []):
            connection, response = self.request("/generate", payload)
            self.assertEqual(response.status, 400)
            self.assertIn("error", json.loads(response.read()))
            connection.close()
        connection, response = self.request(
            "/stop", {}, {"Content-Type": "application/json", "Origin": "https://example.com"}
        )
        self.assertEqual(response.status, 403)
        response.read()
        connection.close()

    def test_reload_between_runs_and_reject_invalid_checkpoint(self):
        original = self.server.model
        path = Path(self.temporary.name) / "reload.json"
        try:
            self.server.checkpoint_path = path
            for width in (4, 8):
                model = fixture(width=width)
                # Validation must use the newly loaded vocabulary too.
                model["config"]["vocabulary"][-1] = "e"
                path.write_text(json.dumps(model))
                expected = load_checkpoint(path)
                connection, response = self.request("/generate", dict(prompt="abce", tokens=1))
                self.assertEqual(response.status, 202)
                run = json.loads(response.read())["run"]
                connection.close()
                events = self.wait_for_run(run)
                self.assertEqual(events[-1]["type"], "done")
                started = events[0]
                self.assertEqual(started["model"]["checkpoint_sha256"], expected["sha256"])
                self.assertEqual(started["model"]["width"], width)
                saved = json.loads((Path(events[-1]["output"]) / "result.json").read_text())
                self.assertEqual(saved["checkpoint_sha256"], expected["sha256"])
            path.write_text("{}")
            connection, response = self.request("/generate", dict(prompt="abce", tokens=1))
            self.assertEqual(response.status, 400)
            self.assertIn("checkpoint", json.loads(response.read())["error"])
            connection.close()
            self.assertFalse(self.server.run_lock.locked())
        finally:
            with self.server.run_lock:
                self.server.checkpoint_path = None
                self.server.model = original


if __name__ == "__main__":
    unittest.main()
