"""Tokenizer identity, portable checkpoints and complete subword execution on RTL."""

import copy
import importlib.util
import json
import math
import random
import string
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snackwell.character_model import (
    FORMAT_SUBWORD,
    encode_prompt,
    load_checkpoint,
    parameter_shapes,
    validate_checkpoint,
)
from snackwell.decoding import Decoder
from snackwell.kv_cache import Generation, model_signature
from snackwell.model_preview import ModelPreview
from snackwell.model_runner import forward
from snackwell.paths import ROOT
from snackwell.reference.inference import verify_approx
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.tokenizer import Tokenizer
from snackwell.transformer_training import run_step
from snackwell.transformer_training_data import Corpus, checkpoint
from snackwell.transformer_training_runtime import TransformerTrainingRuntime


def fixture():
    vocabulary = list(" " + string.ascii_lowercase)
    merges = [(a, b) for a in range(1, 27) for b in range(1, 27)][: 512 - len(vocabulary)]
    vocabulary += [vocabulary[a] + vocabulary[b] for a, b in merges]
    tokenizer = Tokenizer(vocabulary, merges, 27)
    config = dict(context=4, width=8, layers=2, heads=2, epsilon=1e-5, vocabulary=vocabulary)
    rng = random.Random(2753)
    weights = {
        name: [
            (1 if "gamma" in name else 0) + rng.uniform(-0.1, 0.1) for _ in range(math.prod(shape))
        ]
        for name, shape in parameter_shapes(4, 8, 512, 2).items()
    }
    return validate_checkpoint(
        dict(format=FORMAT_SUBWORD, config=config, weights=weights, tokenizer=tokenizer.metadata())
    )


class TokenizerTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional CPU likelihood oracle")
    def test_shared_text_score_uses_target_characters_and_latency_uses_tokens(self):
        from snackwell.cli.text_quality import score_text
        from snackwell.kv_cache import latency_report

        model = fixture()
        model["weights"] = {k: [0.0] * len(v) for k, v in model["weights"].items()}
        result = score_text(model, "ab " * 10, "ab cd")
        self.assertEqual(result["characters"], 5)
        self.assertEqual(result["tokens"], 3)
        self.assertAlmostEqual(result["nats"], 3 * math.log(512), places=12)
        reports = [
            dict(character="abc", wall_seconds=2.0, cache=dict(phase="prefill")),
            dict(character="d", wall_seconds=1.0, cache=dict(phase="decode")),
        ]
        timing = latency_report(reports, True, subword=True)
        self.assertEqual(timing["characters_per_second"], 4 / 3)
        self.assertEqual(timing["tokens_per_second"], 2 / 3)
        self.assertNotIn("subsequent_character_seconds", timing)

    def test_roundtrip_boundaries_and_deterministic_training(self):
        training = "To be, or not to be!\n" * 80
        codec = Tokenizer.train(training, 128)
        self.assertEqual(codec.metadata(), Tokenizer.train(training, 128).metadata())
        restored = Tokenizer.restore(json.loads(json.dumps(codec.metadata())))
        for text in (training, "  To be\n\n", "To be! To be!", "", "not to be"):
            self.assertEqual(restored.decode(restored.encode(text)), text)
            self.assertEqual(restored.encode(text), codec.encode(text))
        with self.assertRaisesRegex(ValueError, "outside"):
            restored.encode("Z")
        self.assertLess(len(codec.encode(training)), len(training))

    def test_invalid_metadata_and_shapes_fail_before_execution(self):
        model = fixture()
        mutations = [
            lambda m: m["tokenizer"]["merges"][0].__setitem__(0, 27),
            lambda m: m["tokenizer"].__setitem__("pattern", ".*"),
            lambda m: m["tokenizer"]["vocabulary"].__setitem__(27, "broken"),
            lambda m: m["config"].__setitem__("context", 3),
            lambda m: m["config"].__setitem__("width", 7),
            lambda m: m.pop("tokenizer"),
            lambda m: m["tokenizer"].__setitem__("alphabet_size", True),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(model)
            mutate(candidate)
            with self.assertRaises(ValueError):
                validate_checkpoint(candidate)
        changed = copy.deepcopy(model)
        changed["tokenizer"]["merges"][0][0] = 2
        self.assertNotEqual(model_signature(model), model_signature(changed))

    def test_generated_ids_are_not_retokenized_in_either_cache_mode(self):
        model = fixture()
        token = model["config"]["vocabulary"].index("a")
        self.assertNotEqual(encode_prompt(model, "aa", pad=False), [token, token])
        for cache in (True, False):
            generation = Generation(model, "a", cache=cache)
            before = list(generation.ids)
            with patch(
                "snackwell.model_runner.forward",
                return_value=dict(next_character="a", selection=dict(token_id=token)),
            ):
                generation.next(Path("unused"), None)
            self.assertEqual(
                generation.ids, (before + [token]) if cache else (before + [token])[-4:]
            )


class SubwordRTLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/subword_model"
        cls.output.mkdir(parents=True, exist_ok=True)
        cls.simulator = compile_model_runner(cls.output)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.output)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.execution = self.simulator
        if self.simulator.backend == "verilator":
            self.execution = SimulatorSession(self.simulator)
            self.addCleanup(self.execution.close)

    def test_cached_full_and_preview_agree_through_rebuild(self):
        model = fixture()
        generation = Generation(model, "aa")
        decoder = Decoder("sample", 1729, 0.7)
        phases = []
        for index in range(6):
            draw = decoder.draw()
            cached = generation.next(
                self.path / f"cached{index}", self.execution, draw=draw, temperature=0.7
            )
            full = forward(
                model,
                cached["input_ids"],
                self.path / f"full{index}",
                self.execution,
                draw=draw,
                temperature=0.7,
                last_only=True,
            )
            self.assertEqual(cached["sampling_probabilities"], full["sampling_probabilities"])
            self.assertEqual(cached["selection"], full["selection"])
            phases.append(cached["cache"]["phase"])
        self.assertIn("rebuild", phases)
        expected = ModelPreview(model, "scalar").generate("aa", 6, Decoder("sample", 1729, 0.7))
        self.assertEqual(generation.text, expected)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional independent autograd oracle")
    def test_full_training_all_gradients_and_checkpoint_reload(self):
        import torch
        import torch.nn.functional as F

        from snackwell.torch_model import ReferenceTransformer

        torch.set_num_threads(1)
        model = fixture()
        ids, targets = [511, 510, 1, 511], [510, 1, 511, 509]
        runtime = TransformerTrainingRuntime(self.path / "train", self.execution)
        actual, updated = run_step(runtime, model, ids, targets, 0.01)
        network = ReferenceTransformer(model["config"], model["weights"]).double()
        loss = F.cross_entropy(network(torch.tensor([ids])).reshape(4, 512), torch.tensor(targets))
        loss.backward()
        for name, p in network.parameters_by_name.items():
            verify_approx(
                actual["tensors"]["gradients"][name],
                p.grad.flatten().tolist(),
                "transformer_training",
            )
        corpus_path = self.path / "corpus.txt"
        corpus_path.write_text("ab cd ef gh ij kl mn op qr st uv wx yz " * 100)
        corpus = Corpus(corpus_path, 4, model["config"]["vocabulary"], model["tokenizer"])
        saved = self.path / "checkpoint.json"
        saved.write_text(json.dumps(checkpoint(updated, corpus, 1729, 0.01, 1)))
        restored = load_checkpoint(saved)
        self.assertEqual(restored["tokenizer"], model["tokenizer"])
        self.assertEqual(restored["weights"], updated["weights"])
        result = forward(restored, ids, self.path / "updated", self.execution)
        self.assertTrue(result["verified"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional host training dependency")
    def test_host_export_and_rtl_training_cli_resume_use_same_tokenizer(self):
        corpus = self.path / "corpus.txt"
        corpus.write_text("abcd abcd bcda cdab\n" * 100)
        out = self.path / "host"
        common = [
            "--data",
            str(corpus),
            "--width",
            "4",
            "--context",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
        ]
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_reference_model.py"),
                *common,
                "--bpe-vocabulary",
                "128",
                "--steps",
                "2",
                "--batch-size",
                "2",
                "--evaluation-targets",
                "16",
                "--eval-every",
                "1",
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        model = load_checkpoint(out / "checkpoint.json")
        self.assertIn("tokenizer", model)
        for label, origin in (("rtl", "--checkpoint"), ("resumed", "--resume")):
            destination = self.path / label
            source = out if label == "rtl" else self.path / "rtl"
            run = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/train_transformer.py"),
                    "--data",
                    str(corpus),
                    origin,
                    str(source / "checkpoint.json"),
                    "--steps",
                    "1",
                    "--evaluation-windows",
                    "1",
                    "--out",
                    str(destination),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            saved = load_checkpoint(destination / "checkpoint.json")
            self.assertEqual(saved["tokenizer"], model["tokenizer"])
        self.assertEqual(saved["training"]["completed_steps"], 2)
