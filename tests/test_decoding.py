import json
import math
import unittest

from snackwell.assembler import assemble
from snackwell.character_model import encode_prompt
from snackwell.decoding import Decoder
from snackwell.kernels.inference import Assembly, Layout, exp_negative
from snackwell.model_preview import ModelPreview
from snackwell.model_runner import forward
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.inference import stable_softmax
from snackwell.simulator import compile_runner, run_program
from tests.test_model import fixture


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/decoding_python"
        cls.executable = compile_runner(cls.output)

    def test_sampling_boundaries_and_zero_padding(self):
        runtime = ModelRuntime(self.output / "boundaries", self.executable)
        row = [0, 0.25, 0, 0.5, 0.25] + [0] * 123
        draws = [0, 0.25, 0.75 - 2**-24, 0.75, 1 - 2**-24, 0.5, 0.25 - 2**-24, 0.9]
        chosen = runtime.rows(
            "sample",
            "categorical",
            [bits(w) for w in row] * 8,
            8,
            128,
            draws=[bits(d) for d in draws],
        )
        self.assertEqual(chosen, [1, 3, 3, 4, 4, 3, 1, 4])

    def test_sampling_respects_weights_instead_of_always_taking_maximum(self):
        runtime = ModelRuntime(self.output / "distribution", self.executable)
        chosen = runtime.rows(
            "sample",
            "categorical",
            [bits(w) for w in (1, 2, 1, 0)] * 64,
            64,
            4,
            draws=[bits((i + 0.5) / 64) for i in range(64)],
        )
        self.assertEqual([chosen.count(i) for i in range(4)], [16, 32, 16, 0])

    def test_sampling_single_support_tiny_weights_and_rounding(self):
        runtime = ModelRuntime(self.output / "rounding", self.executable)
        rows = [
            [0, 0, 0, bits(1)],
            [1, 1, 0, 0],
            [bits(1e-20), 0, 0, 0],
            [bits(0.1), bits(0.2), bits(0.3), 0],
        ]
        chosen = runtime.rows(
            "sample",
            "categorical",
            sum(rows, []),
            4,
            4,
            draws=[bits(1 - 2**-24), bits(0.5), 0, bits(1 - 2**-24)],
        )
        self.assertEqual(chosen, [3, 1, 0, 2])

    def test_invalid_sampling_inputs_are_rejected(self):
        runtime = ModelRuntime(self.output / "invalid", self.executable)
        for weights, draw in [
            ([0] * 4, 0),
            ([-1, 1, 1, 1], 0.5),
            ([float("nan"), 1, 1, 1], 0.5),
            ([1] * 4, 1),
            ([1] * 4, -0.1),
            ([1] * 4, float("nan")),
        ]:
            with self.subTest(weights=weights, draw=draw), self.assertRaises(ValueError):
                runtime.rows(
                    "sample",
                    "categorical",
                    [bits(w) for w in weights] * 4,
                    4,
                    4,
                    draws=[bits(draw)] * 4,
                )

    def test_sampled_model_still_checks_the_original_distribution(self):
        model = fixture()
        ids = encode_prompt(model, "abcd")
        greedy = forward(model, ids, self.output / "greedy", self.executable)
        sampled = forward(model, ids, self.output / "sampled", self.executable, draw=0.75)
        for key in ("probabilities", "predictions", "accuracy", "greedy_next_character"):
            self.assertEqual(greedy[key], sampled[key])
        self.assertTrue(sampled["verified"])
        self.assertEqual(sampled["launches"], greedy["launches"] + 1)
        self.assertEqual(sampled["selection"]["mode"], "sample")
        self.assertEqual(
            sampled["next_character"],
            model["config"]["vocabulary"][sampled["selection"]["token_id"]],
        )
        stage = json.loads(
            next((self.output / "sampled").glob("*_sample_decode/result.json")).read_text()
        )
        self.assertTrue(stage["verified"])
        self.assertEqual(stage["operation"], "categorical")

    def test_seed_reproduces_fp32_draws_below_one(self):
        first, second, other = Decoder(seed=1729), Decoder(seed=1729), Decoder(seed=1730)
        draws = [first.draw() for _ in range(64)]
        self.assertEqual(draws, [second.draw() for _ in range(64)])
        self.assertNotEqual(draws, [other.draw() for _ in range(64)])
        self.assertTrue(all(0 <= d < 1 and value(bits(d)) == d for d in draws))
        self.assertIsNone(Decoder("greedy").draw())
        for seed in (-1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                Decoder(seed=seed)

    def test_temperature_probabilities_keep_masks_and_stability(self):
        runtime = ModelRuntime(self.output / "temperature", self.executable)
        row = [1000, 1001, 999, -math.inf]
        for temperature in (0.1, 0.7, 2.0):
            got = runtime.rows(
                "softmax",
                "softmax",
                [bits(x) for x in row] * 4,
                4,
                4,
                temperature=temperature,
            )
            expected = stable_softmax([x / temperature for x in row])
            for i, word in enumerate(got):
                self.assertAlmostEqual(value(word), expected[i % 4], places=5)
            self.assertEqual(got[3::4], [0] * 4)

    def test_temperature_changes_sampling_without_changing_raw_model(self):
        model = fixture()
        ids = encode_prompt(model, "abcd")
        raw = forward(model, ids, self.output / "temperature_raw", self.executable, last_only=True)
        decoder = Decoder(seed=1729, temperature=0.7)
        sampled = forward(
            model,
            ids,
            self.output / "temperature_sample",
            self.executable,
            draw=decoder.draw(),
            temperature=decoder.temperature,
            last_only=True,
        )
        self.assertEqual(raw["probabilities"], sampled["probabilities"])
        self.assertEqual(raw["predictions"], sampled["predictions"])
        self.assertNotEqual(raw["probabilities"][-5:], sampled["sampling_probabilities"])
        self.assertIn("sampling_probabilities", sampled["accuracy"])
        preview = ModelPreview(model, backend="scalar")
        self.assertEqual(
            preview.generate("abcd", 1, Decoder(seed=1729, temperature=0.7)),
            "abcd" + sampled["next_character"],
        )
        softmax = next(s for s in sampled["stages"] if s["stage"] == "sampling_softmax")
        self.assertTrue(softmax["verified"])
        self.assertEqual(sampled["launches"], raw["launches"] + 2)

    def test_invalid_temperatures_fail_before_inference(self):
        for temperature in (0, -1, 0.01, 3, math.nan, math.inf, True):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                Decoder(temperature=temperature)

    def test_exponential_rounding_budget_near_zero_and_in_tail(self):
        layout = Layout()
        layout.allocate("x", 4)
        layout.allocate("out", 4)
        source = Assembly(layout, "exponential accuracy regression")
        source.emit("XOR R15, R15, R15\nTID R10")
        source.scale_index(10, 4)
        source.emit("LD_SHARED R0, [R10]")
        exp_negative(source, 0, 1, 2, 3, 4, 5)
        source.emit(f"ST_SHARED [R10 + {layout.address('out')}], R1\nEXIT")
        words = assemble(source.text())
        # The old eight-squaring routine missed this error budget near -0.12.
        cases = [
            -0.12,
            -0.04,
            -0.02,
            0,
            -0.001,
            -0.25,
            -0.5,
            -1,
            -2,
            -4,
            -8,
            -12,
            -16,
            -32,
            -64,
            -80,
        ]
        for index in range(0, len(cases), 4):
            inputs = cases[index : index + 4]
            memory = layout.memory.copy()
            memory[:4] = [bits(x) for x in inputs]
            after, _, _ = run_program(
                self.executable, words, memory, self.output / f"exp_{index}", warp_mask=1
            )
            for word, x in zip(after[4:8], inputs):
                self.assertLessEqual(abs(value(word) - math.exp(value(bits(x)))), 2e-6, x)
