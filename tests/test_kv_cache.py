import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from snackwell.character_model import encode_prompt, reference_forward
from snackwell.kv_cache import Generation, KVCache, model_signature
from snackwell.model_preview import ModelPreview
from snackwell.model_runner import forward
from snackwell.numeric import bits
from snackwell.simulator import compile_model_runner
from tests.test_model import fixture


class KVCacheTests(unittest.TestCase):
    def test_signature_tracks_in_place_edits_and_ignores_dictionary_order(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        original = model_signature(model)
        reordered = dict(
            model,
            config=dict(reversed(list(model["config"].items()))),
            weights=dict(reversed(list(model["weights"].items()))),
        )
        self.assertEqual(original, model_signature(reordered))
        changed = copy.deepcopy(model)
        changed["sha256"] = "unchanged-file-checksum"
        before = changed["weights"]["token_embedding"][0]
        changed["weights"]["token_embedding"][0] += 2**-40
        self.assertEqual(bits(before), bits(changed["weights"]["token_embedding"][0]))
        self.assertNotEqual(original, model_signature(changed))
        changed = copy.deepcopy(model)
        changed["config"]["vocabulary"].reverse()
        self.assertNotEqual(original, model_signature(changed))

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name)
        cls.executable = compile_model_runner(cls.output, backend="auto")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_each_prefix_matches_uncached_rtl_across_padding_boundaries(self):
        model = fixture(context=16, width=8, layers=2, heads=2)
        ids = [1, 2, 4, 3, 0, 2, 3, 1, 4]
        cache = KVCache()
        for length in range(1, len(ids) + 1):
            with self.subTest(length=length):
                cached = forward(
                    model,
                    ids[:length],
                    self.output / f"cached_{length}",
                    self.executable,
                    kv_cache=cache,
                    last_only=True,
                    draw=0.37,
                    temperature=0.7,
                )
                rebuilt = KVCache()
                full = forward(
                    model,
                    ids[:length],
                    self.output / f"full_{length}",
                    self.executable,
                    kv_cache=rebuilt,
                    last_only=False,
                    draw=0.37,
                    temperature=0.7,
                )
                self.assertEqual(cached["probabilities"][-5:], full["probabilities"][-5:])
                self.assertEqual(cached["sampling_probabilities"], full["sampling_probabilities"])
                self.assertEqual(cached["selection"], full["selection"])
                self.assertEqual(cache.layers, rebuilt.layers)
                self.assertEqual(cache.ids, tuple(ids[:length]))
                self.assertEqual(cached["cache"]["reused_tokens"], length - 1)
                self.assertEqual(cached["cache"]["computed_tokens"], 1)
                self.assertEqual(cached["cache"]["bytes"], 2 * 2 * length * 8 * 4)
                if length > 4:
                    self.assertLess(cached["cycles"], full["cycles"])
                    self.assertLess(cached["launches"], full["launches"])

    def test_overflow_rebuilds_positions_and_all_layers_before_reusing_cache(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        generation = Generation(model, "abcdabcd")
        phases = []
        for step in range(7):
            result = generation.next(self.output / f"overflow_{step}", self.executable, draw=0.6)
            phases.append(result["cache"]["phase"])
            full = forward(
                model,
                result["input_ids"],
                self.output / f"overflow_full_{step}",
                self.executable,
                draw=0.6,
                last_only=True,
            )
            self.assertEqual(result["probabilities"][-5:], full["probabilities"][-5:])
            self.assertEqual(result["selection"], full["selection"])
            if result["cache"]["phase"] == "rebuild":
                self.assertEqual(result["cache"]["reused_tokens"], 0)
                self.assertEqual(
                    result["input_ids"], encode_prompt(model, generation.text[:-1])[-6:]
                )
                self.assertEqual(result["cache"]["computed_tokens"], 6)
        self.assertEqual(
            phases, ["prefill", "rebuild", "decode", "decode", "rebuild", "decode", "decode"]
        )
        preview = ModelPreview(model, backend="scalar")
        decoder = SimpleNamespace(mode="sample", temperature=1.0, draw=lambda: 0.6)
        self.assertEqual(preview.generate("abcdabcd", 7, decoder), generation.text)

    def test_context_64_and_long_prompt_truncation(self):
        model = fixture(context=64)
        prompt = "abcd" * 17
        generation = Generation(model, prompt)
        self.assertEqual(generation.ids, encode_prompt(model, prompt))
        for step, length in enumerate((64, 48, 49)):
            result = generation.next(self.output / f"long_{step}", self.executable)
            self.assertEqual(result["cache"]["context_tokens"], length)
            full = forward(
                model,
                result["input_ids"],
                self.output / f"long_full_{step}",
                self.executable,
                last_only=True,
            )
            self.assertEqual(result["probabilities"][-5:], full["probabilities"][-5:])
            self.assertEqual(result["selection"], full["selection"])

    def test_expanded_feedforward_cache_and_rebuild_match_full_rtl(self):
        model = fixture(context=8, width=8, layers=2, heads=2, ffn_multiplier=4)
        generation = Generation(model, "abcdabc")
        phases = []
        for step in range(4):
            result = generation.next(
                self.output / f"ffn_expanded_{step}", self.executable, draw=0.37, temperature=0.7
            )
            phases.append(result["cache"]["phase"])
            full = forward(
                model,
                result["input_ids"],
                self.output / f"ffn_expanded_full_{step}",
                self.executable,
                last_only=True,
                draw=0.37,
                temperature=0.7,
            )
            self.assertEqual(result["probabilities"][-5:], full["probabilities"][-5:])
            self.assertEqual(result["sampling_probabilities"], full["sampling_probabilities"])
            self.assertEqual(result["selection"], full["selection"])
        self.assertEqual(phases, ["prefill", "decode", "rebuild", "decode"])

    def test_context_128_append_and_rebuild_match_full_recomputation(self):
        model = fixture(context=128, width=8, layers=2, heads=2)
        generation = Generation(model, "abcd" * 31 + "abc")
        for step, length in enumerate((127, 128, 96, 97)):
            result = generation.next(self.output / f"context128_{step}", self.executable, draw=0.3)
            self.assertEqual(len(result["input_ids"]), length)
            full = forward(
                model,
                result["input_ids"],
                self.output / f"context128_full_{step}",
                self.executable,
                last_only=True,
                draw=0.3,
            )
            self.assertEqual(result["probabilities"][-5:], full["probabilities"][-5:])
            self.assertEqual(result["selection"], full["selection"])

    def test_incremental_reference_exactly_matches_full_independent_reference(self):
        model = fixture(context=128, width=8, layers=4, heads=2)
        ids = [1, 0, 4, 2, 3, 1, 4, 2] * 16
        previous = None
        for length in list(range(1, 18)) + [126, 127, 128]:
            if previous and len(previous.ids) != length - 1:
                previous = None
            got, state = reference_forward(
                model, ids[:length], previous=previous, return_state=True
            )
            expected, full_state = reference_forward(model, ids[:length], return_state=True)
            self.assertEqual(got, expected)
            self.assertEqual(state, full_state)
            previous = state
        with self.assertRaisesRegex(ValueError, "reference history"):
            reference_forward(model, ids[:8], previous=previous)

    def test_independent_reference_catches_corrupted_cached_rtl_values(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        cache = KVCache()
        forward(
            model, [1, 2, 3], self.output / "cache_corrupt_prefill", self.executable, kv_cache=cache
        )
        key, values = cache.layers[0]
        cache.layers = ((key, tuple([bits(100)] * len(values))),) + cache.layers[1:]
        before = copy.deepcopy(cache)
        output = self.output / "cache_corrupt_append"
        with self.assertRaisesRegex(ValueError, "transformer output"):
            forward(model, [1, 2, 3, 4], output, self.executable, kv_cache=cache)
        self.assertEqual(cache, before)
        self.assertFalse((output / "result.json").exists())

    def test_cancelled_append_preserves_previous_cache_and_can_retry(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        generation = Generation(model, "abc")
        generation.next(self.output / "cancel_prefill", self.executable)
        before = copy.deepcopy(generation.cache)
        ids, text = list(generation.ids), generation.text
        output = self.output / "cancel_append"

        def stop(event):
            if event["type"] == "kernel_start" and event["stage"] == "block1_ffn_down_matmul":
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            generation.next(output, self.executable, stop)
        self.assertEqual(generation.cache, before)
        self.assertEqual(generation.ids, ids)
        self.assertEqual(generation.text, text)
        self.assertFalse((output / "result.json").exists())
        result = generation.next(output, self.executable)
        self.assertTrue(result["verified"])
        self.assertEqual(result["cache"]["reused_tokens"], 3)
        # A new prompt has no state from the completed/cancelled prompt.
        fresh = Generation(model, "a")
        self.assertEqual(fresh.cache, KVCache())
        self.assertEqual(fresh.ids, [1])

    def test_rejects_wrong_prefix_weights_and_rtl_without_committing(self):
        model = fixture(context=8)
        unversioned = self.output / "unversioned"
        unversioned.mkdir()
        (unversioned / "result.json").write_text('{"verified":true}')
        with self.assertRaisesRegex(ValueError, "RTL source fingerprint"):
            forward(
                model,
                [1, 2],
                unversioned,
                self.executable.executable,
                kv_cache=KVCache(),
            )
        self.assertFalse((unversioned / "result.json").exists())
        cache = KVCache()
        forward(model, [1, 2], self.output / "identity", self.executable, kv_cache=cache)
        changed = copy.deepcopy(model)
        changed["weights"]["wq"][0] += 0.1
        stale = copy.deepcopy(cache)
        stale.rtl_source_sha256 = "obsolete"
        truncated = copy.deepcopy(cache)
        truncated.layers = (((), ()),)
        for index, (weights, ids, state) in enumerate(
            (
                (model, [2, 1, 3], cache),
                (changed, [1, 2, 3], cache),
                (model, [1, 2, 3], stale),
                (model, [1, 2, 3], truncated),
            )
        ):
            before = copy.deepcopy(state)
            output = self.output / f"invalid_{index}"
            with self.assertRaisesRegex(ValueError, "KV cache does not match"):
                forward(weights, ids, output, self.executable, kv_cache=state)
            self.assertEqual(state, before)
            self.assertFalse(list(output.glob("*/program.asm")))

    def test_short_prefix_reference_is_causal_and_legacy_mode_is_available(self):
        model = fixture(context=8, width=8, layers=2, heads=2)
        ids = [1, 2, 3, 4, 2, 3, 4, 0]
        full = reference_forward(model, ids)
        for length in (1, 3, 5, 7):
            short = reference_forward(model, ids[:length])
            self.assertEqual(short["logits"], full["logits"][: length * 5])
        self.assertEqual(encode_prompt(model, "ab", pad=False), [1, 2])
        old = Generation(model, "ab", cache=False)
        self.assertIsNone(old.cache)
        self.assertEqual(old.ids, [0] * 6 + [1, 2])
        result = old.next(self.output / "legacy", self.executable)
        self.assertFalse(result["cache"]["enabled"])
        saved = json.loads((self.output / "legacy/result.json").read_text())
        self.assertEqual(saved["input_ids"], [0] * 6 + [1, 2])


if __name__ == "__main__":
    unittest.main()
