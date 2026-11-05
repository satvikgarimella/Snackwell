"""Prompt preservation and independently checked SM sampling filters."""

import copy
import math
import os
import shutil
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from snackwell.character_model import FORMAT_SUBWORD, parameter_shapes, validate_checkpoint
from snackwell.decoding import Decoder, prepare_prompt
from snackwell.kv_cache import Generation
from snackwell.model_preview import ModelPreview
from snackwell.model_runner import forward
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.tokenizer import Tokenizer


def fixture():
    tokenizer = Tokenizer.train("my name is alice\nmy name is bob\n" * 80, 128)
    vocabulary = tokenizer.vocabulary
    config = dict(context=16, width=8, layers=2, heads=2, epsilon=1e-5, vocabulary=vocabulary)
    weights = {
        name: [float("gamma" in name)] * math.prod(shape)
        for name, shape in parameter_shapes(16, 8, len(vocabulary), 2).items()
    }
    weights["lm_bias"] = [-10.0] * len(vocabulary)
    for token, score in (("m", 6), (" alice", 4), (" bob", 3), (" ", -4)):
        weights["lm_bias"][vocabulary.index(token)] = score
    return validate_checkpoint(
        dict(format=FORMAT_SUBWORD, config=config, weights=weights, tokenizer=tokenizer.metadata())
    )


class SamplingFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/sampling_filters"
        cls.executable = compile_model_runner(cls.output, compact=True)

    def test_prefix_preparation_preserves_spaces_and_partial_words(self):
        model = fixture()
        vocab = model["config"]["vocabulary"]
        for prompt in ("my name is ", "my name is  ", "my name is ali"):
            ids, prefix = prepare_prompt(model, prompt)
            self.assertTrue(prefix)
            self.assertEqual("".join(vocab[i] for i in ids) + prefix, prompt)
            padded, same_prefix = prepare_prompt(model, prompt, cache=False)
            self.assertEqual(len(padded), 16)
            self.assertEqual(prefix, same_prefix)
            self.assertTrue(("".join(vocab[i] for i in padded) + prefix).endswith(prompt))
        for prompt in (" ", "m", "my name is alice\n"):
            self.assertEqual(prepare_prompt(model, prompt)[1], "")
        self.assertEqual(prepare_prompt(model, "my name is ", heal=False)[1], "")
        long_prompt = "my name is " * 20
        ids, prefix = prepare_prompt(model, long_prompt)
        self.assertEqual(len(ids), 16)
        self.assertTrue(long_prompt.endswith("".join(vocab[i] for i in ids) + prefix))

    def test_filter_boundaries_and_packed_mask_on_both_simulators(self):
        inputs = [bits(0.5), bits(0.125), bits(0.125) - 1, 0]
        outputs = []
        backends = ["icarus"]
        if shutil.which(os.environ.get("VERILATOR", "verilator")):
            backends.append("verilator")
        for backend in backends:
            executable = compile_model_runner(self.output / backend, backend, compact=True)
            with (
                SimulatorSession(executable)
                if backend == "verilator"
                else nullcontext(executable) as runner
            ):
                runtime = ModelRuntime(self.output / backend / "kernels", runner)
                filtered = runtime.rows("boundary", "min_p", inputs, 1, 4, min_p=0.25)
                self.assertEqual(filtered, [bits(0.5), bits(0.125), 0, 0])
                tied = runtime.rows("ties", "min_p", [bits(0.5)] * 4, 1, 4, min_p=1)
                self.assertEqual(tied, [bits(0.5)] * 4)
                rows = [bits((i % 4 + 1) / 16) for i in range(4 * 128)]
                got = runtime.rows("rows", "min_p", rows, 4, 128, min_p=0.5)
                self.assertEqual(got, [w if value(w) >= 0.125 else 0 for w in rows])
                values = [bits(i - 256) for i in range(512)]
                allowed = [i in (0, 31, 32, 127, 255, 511) for i in range(512)]
                masked = runtime.restrict_tokens("packed", values, allowed)
                self.assertEqual(
                    masked, [w if allowed[i] else 0xFF800000 for i, w in enumerate(values)]
                )
                large = runtime.rows("large", "min_p", [bits(0.25)] * 512, 1, 512, min_p=1)
                self.assertEqual(large, [bits(0.25)] * 512)
                outputs.append((filtered, got, masked, large))
                self.assertTrue(
                    all(
                        r["allocated_words"] <= 1024 and r["instructions"] <= 1024
                        for r in runtime.reports
                    )
                )
        if len(outputs) == 2:
            self.assertEqual(outputs[0], outputs[1])

    def test_healing_runs_on_rtl_and_cache_appends_the_full_token(self):
        model, prompt = fixture(), "my name is "
        for cache in (True, False):
            for mode in ("greedy", "sample"):
                generation = Generation(model, prompt, cache=cache)
                decoder = Decoder(mode, 1729, 0.7, min_p=1)
                reports = []
                for step in range(2):
                    result = generation.next(
                        self.output / f"model_{cache}_{mode}_{step}",
                        self.executable,
                        draw=decoder.draw(),
                        temperature=0.7,
                        min_p=1,
                    )
                    reports.append(result)
                self.assertEqual(reports[0]["next_character"], "alice")
                self.assertEqual(reports[0]["selection"]["token_text"], " alice")
                self.assertEqual(reports[0]["selection"]["healed_prefix"], " ")
                self.assertEqual(reports[1]["selection"]["healed_prefix"], "")
                self.assertEqual(generation.text, prompt + "alicem")
                self.assertEqual(
                    generation.text,
                    ModelPreview(model, "scalar").generate(
                        prompt, 2, Decoder(mode, 1729, 0.7, min_p=1), cache=cache
                    ),
                )
                raw = forward(
                    model,
                    reports[0]["input_ids"],
                    self.output / f"raw_{cache}_{mode}",
                    self.executable,
                    last_only=True,
                )
                self.assertEqual(reports[0]["probabilities"], raw["probabilities"])
                self.assertEqual(reports[0]["predictions"], raw["predictions"])
                self.assertTrue(any(s["operation"] == "token_mask" for s in reports[0]["stages"]))
                self.assertEqual(
                    any(s["operation"] == "min_p" for s in reports[0]["stages"]), mode == "sample"
                )
                if cache:
                    fresh = forward(
                        model,
                        reports[1]["input_ids"],
                        self.output / f"fresh_{mode}",
                        self.executable,
                        last_only=True,
                    )
                    self.assertEqual(reports[1]["probabilities"], fresh["probabilities"])

    def test_unchanged_token_and_interruption_never_delete_prompt_text(self):
        model = fixture()
        model["weights"]["lm_bias"][model["config"]["vocabulary"].index(" ")] = 10
        generation = Generation(model, "my name is ")
        initial = copy.deepcopy(generation.ids)
        with patch("snackwell.model_runner.forward", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                generation.next(self.output / "cancelled", self.executable)
        self.assertEqual(generation.ids, initial)
        self.assertEqual(generation.token_prefix, " ")
        result = generation.next(self.output / "unchanged", self.executable)
        self.assertEqual(result["next_character"], "")
        self.assertEqual(generation.text, "my name is ")
        self.assertEqual(generation.token_prefix, "")

        # Healing a full window must retain the token IDs through the first rebuild.
        model = fixture()
        prompt = "my name is " * 20
        generation = Generation(model, prompt)
        generation.next(self.output / "full_healed", self.executable)
        rebuilt = generation.next(self.output / "healed_rebuild", self.executable)
        fresh = forward(
            model,
            rebuilt["input_ids"],
            self.output / "healed_fresh",
            self.executable,
            last_only=True,
        )
        self.assertEqual(rebuilt["cache"]["phase"], "rebuild")
        self.assertEqual(rebuilt["probabilities"], fresh["probabilities"])
        self.assertEqual(generation.text, prompt + "alicem")

    def test_invalid_options_fail_before_launch(self):
        for option in (-1, 1.1, math.inf, math.nan, True, "0.1"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                Decoder(min_p=option)
        runtime = ModelRuntime(self.output / "invalid", self.executable)
        for allowed in ([False] * 4, [1] * 4, [True]):
            with self.assertRaises(ValueError):
                runtime.restrict_tokens("bad", [0] * 4, allowed)
        self.assertEqual(runtime.reports, [])
