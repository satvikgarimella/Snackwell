import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from snackwell.character_model import load_checkpoint
from snackwell.paths import ROOT
from snackwell.simulator import compile_model_runner
from snackwell.terminal_model import interactive, predict
from tests.test_model import fixture


class TerminalModelTests(unittest.TestCase):
    def test_interactive_prompt_runs_rtl_and_recovers_from_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoint.json"
            checkpoint.write_text(json.dumps(fixture()))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_model.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(output),
                    "--interactive",
                    "--simulator",
                    "icarus",
                    "--seed",
                    "1729",
                ],
                input="unknown\n/tokens 0\n/temperature 0\n/seed -1\n/tokens 1\n/temperature 0.6\n/seed 1730\nab \nab \n/quit\n",
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("outside the vocabulary", result.stdout)
            self.assertIn("Use /tokens N", result.stdout)
            self.assertIn("temperature must be", result.stdout)
            self.assertIn("Use /seed N", result.stdout)
            self.assertIn("character 1: embedding", result.stdout)
            reports = list(output.glob("terminal/*/result.json"))
            self.assertEqual(len(reports), 2)
            report = json.loads(reports[0].read_text())
            self.assertTrue(report["verified"])
            self.assertGreater(report["latency"]["first_character_seconds"], 0)
            self.assertIsNone(report["latency"]["subsequent_character_seconds"])
            self.assertTrue(report["latency"]["cached_decode"])
            self.assertEqual(report["latency"]["decode_steps"], 0)
            self.assertEqual(report["prompt"], "ab ")
            self.assertEqual(len(report["tokens"]), 1)
            self.assertEqual(report["decoding"], dict(mode="sample", seed=1730, temperature=0.6))
            self.assertEqual(report["tokens"][0]["selection"]["temperature"], 0.6)
            repeated = json.loads(reports[1].read_text())
            self.assertEqual(report["generated"], repeated["generated"])
            self.assertEqual(report["tokens"][0]["selection"], repeated["tokens"][0]["selection"])
            self.assertIn(f"Text: {report['generated']!r}", result.stdout)
            self.assertNotIn("\033", result.stdout)
            self.assertIn("First character:", result.stdout)
            self.assertIn("RTL build:", result.stdout)
            self.assertIn("RTL/reference match", result.stdout)

    def test_cancellation_clears_stale_success_and_never_appends_a_character(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "result.json").write_text('{"verified":true}')
            stream = io.StringIO()
            with (
                patch("snackwell.kv_cache.Generation.next", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                predict(fixture(), "abcd", 1, output, None, stream)
            self.assertFalse((output / "result.json").exists())
            self.assertIn("Stopped before completion", stream.getvalue())
            self.assertNotIn("predicted ", stream.getvalue())

    def test_interactive_reloads_checkpoint_before_next_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "checkpoint.json"
            path.write_text(json.dumps(fixture()))
            original = load_checkpoint(path)
            replacement = fixture(width=8)
            executable = compile_model_runner(output, backend="icarus")
            calls = 0

            def next_prompt(_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    path.write_text(json.dumps(replacement))
                return "abcd" if calls <= 2 else "/quit"

            stream = io.StringIO()
            with patch("builtins.input", side_effect=next_prompt), patch("sys.stdout", stream):
                interactive(
                    original, output, executable, tokens=1, mode="greedy", checkpoint_path=path
                )
            reports = [json.loads(p.read_text()) for p in output.glob("*/result.json")]
            self.assertEqual(len(reports), 2)
            self.assertEqual(
                {r["checkpoint_sha256"] for r in reports},
                {original["sha256"], load_checkpoint(path)["sha256"]},
            )
            self.assertIn("Reloaded checkpoint", stream.getvalue())
            self.assertIn("width 8", stream.getvalue())

    def test_interactive_replaces_stale_rtl_before_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            simulator = compile_model_runner(output, backend="icarus")
            stale = replace(simulator, executable=output / "obsolete.vvp", source_sha256="old")
            stream = io.StringIO()
            with (
                patch("builtins.input", side_effect=["abcd", "/quit"]),
                patch("sys.stdout", stream),
            ):
                interactive(fixture(), output, stale, tokens=1, mode="greedy")
            report = json.loads(next(output.glob("*/result.json")).read_text())
            self.assertTrue(report["verified"])
            self.assertEqual(report["tokens"][0]["rtl_source_sha256"], simulator.source_sha256)
            self.assertIn("Reloaded RTL simulator", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
