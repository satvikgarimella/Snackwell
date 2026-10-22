import copy
import math
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from snackwell.character_model import (
    FORMAT,
    FORMAT_V2,
    encode_prompt,
    parameter_shapes,
    validate_checkpoint,
)
from snackwell.kernels.inference import Layout, tiled_matmul
from snackwell.model_runner import forward
from snackwell.model_runtime import ModelRuntime
from snackwell.numeric import bits, value
from snackwell.paths import ROOT
from snackwell.reference.inference import tensor_product
from snackwell.simulator import compile_runner


def fixture(context=4, width=4, layers=1, heads=1, ffn_multiplier=2):
    rng = random.Random(0xC4A2)
    shapes = parameter_shapes(context, width, 5, layers, ffn_multiplier)
    weights = {
        name: [
            (1 if "gamma" in name else 0) + rng.uniform(-0.2, 0.2) for _ in range(math.prod(shape))
        ]
        for name, shape in shapes.items()
    }
    config = dict(context=context, width=width, epsilon=1e-5, vocabulary=list(" abcd"))
    modern = layers > 1 or heads > 1 or max(context, width) > 16 or ffn_multiplier != 2
    if modern:
        config.update(layers=layers, heads=heads)
    if ffn_multiplier != 2:
        config["ffn_multiplier"] = ffn_multiplier
    return validate_checkpoint(
        dict(
            format=FORMAT_V2 if modern else FORMAT,
            config=config,
            weights=weights,
        )
    )


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / "build/tests/model_python"
        cls.executable = compile_runner(cls.output)

    def test_full_model_predictions_and_causality(self):
        model = fixture()
        events = []
        first = forward(model, [1, 2, 3, 4], self.output / "first", self.executable, events.append)
        changed = forward(model, [1, 4, 4, 4], self.output / "changed", self.executable)
        self.assertTrue(first["verified"])
        self.assertEqual(
            set(first["accuracy"]), {"embedding", "block", "final_norm", "logits", "probabilities"}
        )
        self.assertEqual(first["probabilities"][:5], changed["probabilities"][:5])
        self.assertNotEqual(first["probabilities"][5:], changed["probabilities"][5:])
        self.assertEqual(first["cycles"], sum(s["cycles"] for s in first["stages"]))
        starts = [event for event in events if event["type"] == "kernel_start"]
        completed = [event for event in events if event["type"] == "kernel_complete"]
        self.assertEqual(
            [event["stage"] for event in starts], [s["stage"] for s in first["stages"]]
        )
        self.assertEqual(sum(event["cycles"] for event in completed), first["cycles"])
        self.assertEqual(
            [event["launch"] for event in completed], list(range(1, first["launches"] + 1))
        )
        attention = next(event["weights"] for event in events if event["type"] == "attention")
        self.assertEqual(len(attention), 4)
        self.assertAlmostEqual(sum(attention), 1, places=5)

    def test_large_vocabulary_softmax_mask_and_argmax(self):
        runtime = ModelRuntime(self.output / "vocabulary", self.executable)
        x = [bits((col - 20) / 8 if col < 65 else 100) for _ in range(4) for col in range(128)]
        masked = runtime.rows("mask", "vocabulary_mask", x, 4, 128, vocabulary=65)
        probabilities = runtime.rows("softmax", "softmax", masked, 4, 128)
        winners = runtime.rows("argmax", "argmax", probabilities, 4, 128)
        self.assertEqual(winners, [64] * 4)
        for row in range(4):
            self.assertEqual(probabilities[row * 128 + 65 : (row + 1) * 128], [0] * 63)
        # Equal maxima choose the lowest index.
        self.assertEqual(runtime.rows("ties", "argmax", [bits(1)] * 16, 4, 4), [0] * 4)

    def test_streamed_matmul_and_lookup(self):
        rng = random.Random(972)
        runtime = ModelRuntime(self.output / "streaming", self.executable)
        a = [bits(rng.uniform(-2, 2)) for _ in range(8 * 32)]
        b = [bits(rng.uniform(-2, 2)) for _ in range(32 * 32)]
        got = runtime.matmul("large", a, b, 8, 32, 32)
        self.assertEqual(got, tensor_product(a, b, 8, 32, 32))
        # A logical table larger than the physical 1024-word memory.
        table = [bits(i / 32) for i in range(65 * 16)]
        position = [bits(-i / 64) for i in range(4 * 16)]
        got = runtime.lookup([0, 64, 32, 1], table, position, 16, 65)
        self.assertEqual(
            [value(w) for w in got],
            [
                token / 2 + c / 32 - (r * 16 + c) / 64
                for r, token in enumerate([0, 64, 32, 1])
                for c in range(16)
            ],
        )
        self.assertTrue(
            all(s["allocated_words"] <= 1024 and s["instructions"] <= 1024 for s in runtime.reports)
        )

    def test_transport_corruption_fails_whole_model_check(self):
        model = fixture()
        original = ModelRuntime.lookup

        def corrupted(runtime, *args, **kwargs):
            result = original(runtime, *args, **kwargs)
            result[0] = bits(value(result[0]) + 1)
            return result

        output = self.output / "corrupt"
        output.mkdir(exist_ok=True)
        (output / "result.json").write_text('{"verified":true}')
        with (
            patch.object(ModelRuntime, "lookup", corrupted),
            self.assertRaisesRegex(ValueError, "transformer output"),
        ):
            forward(model, [0, 1, 2, 3], output, self.executable)
        self.assertFalse((output / "result.json").exists())

    def test_k_chunk_accumulation_preserves_fp32_order(self):
        runtime = ModelRuntime(self.output / "chunk_k", self.executable)
        rng = random.Random(92)
        # Mixed magnitudes and cancellation expose a host sum or a changed
        # reduction tree, even when both are close in real-number arithmetic.
        for k in (64, 128):
            a = [bits(rng.choice((-1, 1)) * 2 ** rng.randrange(-12, 12)) for _ in range(4 * k)]
            b = [bits(rng.uniform(-1, 1)) for _ in range(k * 8)]
            expected = tensor_product(a, b, 4, 8, k)
            got = runtime.matmul("normal", a, b, 4, 8, k)
            transposed = [b[r * 8 + c] for c in range(8) for r in range(k)]
            got_transposed = runtime.matmul("transposed", a, transposed, 4, 8, k, True)
            self.assertEqual(got, expected)
            self.assertEqual(got_transposed, expected)

    def test_matmul_warp_partition_preserves_outputs_and_hides_waits(self):
        rng = random.Random(823)
        m, n, k = 8, 16, 32
        a = [bits(rng.uniform(-2, 2)) for _ in range(m * k)]
        b = [bits(rng.uniform(-2, 2)) for _ in range(k * n)]
        initial = [bits(rng.uniform(-1, 1)) for _ in range(m * n)]
        for transpose in (False, True):
            right = [b[r * n + c] for c in range(n) for r in range(k)] if transpose else b
            results, cycles = [], []
            for warps in (1, 2, 3, 4):
                layout = Layout()
                layout.allocate("a", len(a), a)
                layout.allocate("b", len(right), right)
                layout.allocate("out", len(initial), initial)
                runtime = ModelRuntime(self.output / f"warps_{transpose}_{warps}", self.executable)
                results.append(
                    runtime.launch(
                        "matmul",
                        layout,
                        tiled_matmul(
                            layout, m, n, k, transpose_b=transpose, accumulate=True, warps=warps
                        ),
                        "matmul",
                        ("a", "b", "out"),
                        "out",
                        warp_mask=(1 << warps) - 1,
                        m=m,
                        n=n,
                        k=k,
                        transpose_b=transpose,
                        accumulate=True,
                    )
                )
                cycles.append(runtime.reports[-1]["cycles"])
            self.assertTrue(all(result == results[0] for result in results))
            self.assertLess(cycles[-1], cycles[0])

    def test_matmul_bias_applied_once_after_all_k_chunks(self):
        from snackwell.reference.fp32 import FADD, reference

        runtime = ModelRuntime(self.output / "matmul_bias", self.executable)
        rng = random.Random(289)
        for m, n, k in ((4, 4, 4), (8, 16, 64), (4, 8, 128)):
            a = [bits(rng.uniform(-1, 1)) for _ in range(m * k)]
            b = [bits(rng.uniform(-1, 1)) for _ in range(k * n)]
            bias = [bits(rng.uniform(-4, 4)) for _ in range(n)]
            product = tensor_product(a, b, m, n, k)
            expected = [reference(FADD, x, bias[i % n]) for i, x in enumerate(product)]
            got = runtime.matmul("biased", a, b, m, n, k, bias=bias)
            self.assertEqual(got, expected)

    def test_attention_mask_keeps_absolute_query_position(self):
        runtime = ModelRuntime(self.output / "mask_offset", self.executable)
        scores = [bits(2)] * (64 * 64)
        got = runtime.attention_scale(scores, 64, 4)
        tail = runtime.attention_scale(scores[-4 * 64 :], 4, 4, columns=64, row_offset=60)
        self.assertEqual(got[-4 * 64 :], tail)
        for row in range(64):
            self.assertEqual(
                got[row * 64 : (row + 1) * 64], [bits(1)] * (row + 1) + [0xFF800000] * (63 - row)
            )

    def test_multihead_layers_and_generation_match_full_forward(self):
        model = fixture(context=8, width=8, layers=4, heads=2)
        ids = [1, 2, 3, 4, 1, 3, 2, 4]
        full = forward(model, ids, self.output / "multi_full", self.executable, draw=0.7)
        tail = forward(
            model, ids, self.output / "multi_tail", self.executable, draw=0.7, last_only=True
        )
        self.assertEqual(tail["probabilities"], full["probabilities"][-5:])
        self.assertEqual(tail["predictions"], full["predictions"][-1:])
        self.assertEqual(tail["output_row_offset"], len(ids) - 1)
        self.assertEqual(tail["output_rows"], 1)
        self.assertEqual(tail["selection"], full["selection"])
        self.assertLess(tail["cycles"], full["cycles"])
        changed = forward(model, ids[:-1] + [0], self.output / "multi_causal", self.executable)
        self.assertEqual(full["probabilities"][:-5], changed["probabilities"][:-5])

    def test_long_context_embedding_preserves_absolute_positions(self):
        runtime = ModelRuntime(self.output / "long_embedding", self.executable)
        ids = [i * 13 % 128 for i in range(64)]
        width = 8
        table = [bits(i / 32) for i in range(128 * width)]
        position = [bits(-i / 64) for i in range(64 * width)]
        got = runtime.lookup(ids, table, position, width, 128)
        self.assertEqual(
            [value(w) for w in got],
            [
                (token * width + col) / 32 - (row * width + col) / 64
                for row, token in enumerate(ids)
                for col in range(width)
            ],
        )
        self.assertEqual(len(runtime.reports), 4)
        self.assertTrue(all(s["allocated_words"] <= 1024 for s in runtime.reports))

    def test_multihead_checkpoint_rejects_invalid_configuration(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        for key, bad in (
            ("layers", 0),
            ("layers", True),
            ("heads", 3),
            ("heads", 4),
            ("context", 256),
            ("width", 256),
        ):
            with self.subTest(key=key, value=bad):
                altered = copy.deepcopy(model)
                altered["config"][key] = bad
                with self.assertRaises(ValueError):
                    validate_checkpoint(altered)

    def test_checkpoint_validation_and_cli_failure(self):
        model = fixture()
        self.assertEqual(encode_prompt(model, "ab"), [0, 0, 1, 2])
        for prompt in ("", "unknown"):
            with self.assertRaises(ValueError):
                encode_prompt(model, prompt)
        for modify in (
            lambda m: m["weights"].pop("wq"),
            lambda m: m["weights"]["lm_bias"].append(0),
            lambda m: m["weights"]["wq"].__setitem__(0, float("nan")),
            lambda m: m["config"].__setitem__("epsilon", 0),
            lambda m: m["config"].__setitem__("vocabulary", ["a", "a"]),
        ):
            bad = copy.deepcopy(model)
            modify(bad)
            with self.assertRaises(ValueError):
                validate_checkpoint(bad)
        output = self.output / "cli"
        output.mkdir(exist_ok=True)
        (output / "result.json").write_text('{"verified":true}')
        checkpoint = output / "invalid.json"
        checkpoint.write_text("{}")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_model.py"),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
