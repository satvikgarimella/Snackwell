"""Generation diagnostics must catch sentence loops as well as repeated characters."""

import unittest

from snackwell.cli.generation_quality import repetition
from snackwell.decoding import Decoder
from snackwell.model_preview import ModelPreview


class GenerationQualityTests(unittest.TestCase):
    def test_character_target_keeps_complete_subword_tokens(self):
        from tests.test_subword_model import fixture

        model = fixture()
        model["weights"] = {name: [0.0] * len(p) for name, p in model["weights"].items()}
        model["weights"]["lm_bias"][model["config"]["vocabulary"].index("ab")] = 10
        preview = ModelPreview(model, "scalar")
        self.assertEqual(preview.generate("aa", 8, Decoder("greedy"), characters=5), "aaababab")
        self.assertEqual(preview.generate("aa", 8, Decoder("greedy")), "aa" + "ab" * 8)

    def test_character_target_preserves_sample_sequence_across_rebuilds(self):
        from tests.test_model import fixture

        preview = ModelPreview(fixture(context=4), "scalar")
        full = preview.generate("abcd", 8, Decoder("sample", 1729, 0.7))
        limited = preview.generate("abcd", 8, Decoder("sample", 1729, 0.7), characters=5)
        self.assertEqual(limited, full[:9])
        for invalid in (0, -1, True, 1.5, 9):
            with self.assertRaises(ValueError):
                preview.generate("abcd", 8, Decoder("greedy"), characters=invalid)

    def test_repeated_speaker_and_sentence_are_detected(self):
        phrase = "ROMEO:\nI will not be a word.\n\n"
        text = "Another speaker.\n\n" + phrase * 3
        result = repetition(text)
        self.assertGreater(result["positions_starting_three_repeats"], 0)
        self.assertEqual(len(result["longest_tripled_substring"]), len(phrase))
        self.assertIn(result["longest_tripled_substring"] * 3, text)

    def test_roman_numeral_exposes_single_character_pattern(self):
        result = repetition("KING RICHARD III:\nGood night, my lord.")
        self.assertEqual(result["positions_starting_three_repeats"], 1)
        self.assertEqual(result["longest_tripled_substring"], "I")

    def test_two_copies_do_not_count_as_three(self):
        phrase = "ROMEO:\nI will not be a word.\n\n"
        self.assertEqual(repetition(phrase * 2)["positions_starting_three_repeats"], 0)

    def test_short_repetitions_and_empty_text(self):
        self.assertGreater(repetition("hahaha")["positions_starting_three_repeats"], 0)
        self.assertEqual(repetition("")["positions_starting_three_repeats"], 0)
        self.assertEqual(repetition("")["repeated_fourgram_fraction"], 0)
