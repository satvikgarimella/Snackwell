"""Supplement preparation excludes duplicate works and preserves dialogue boundaries."""

import string
import unittest

from snackwell.shakespeare_data import dialogue, supplement, word_windows


class ShakespeareDataTests(unittest.TestCase):
    def test_dialogue_skips_contents_stage_directions_and_speaker_punctuation(self):
        text = (
            "Contents\n\nACT I\n\nSCENE I. The house\n\nDramatis Personae\n\n"
            "KING, the ruler.\n\nACT I\n\nSCENE I. A hall.\n\n"
            "Enter Hamlet and\nHoratio.\n\nHAMLET.\nWho’s there?\n\n"
            "[_Exit._]\n\nHORATIO.\nA friend of Cæsar and Phœbus.\n"
        )
        alphabet = set(string.ascii_letters + " ,.:;'!?-\n")
        self.assertEqual(
            dialogue(text, alphabet),
            "HAMLET:\nWho's there?\n\nHORATIO:\nA friend of Caesar and Phoebus.\n",
        )

    def test_whole_held_out_play_and_overlapping_works_are_excluded(self):
        words = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million first second third fourth fifth"
        base = string.ascii_letters + " ,.:;'!?-\n" + (words + "\n") * 20
        titles = [
            "THE SONNETS",
            "KING RICHARD THE SECOND",
            "A NEW PLAY",
            "THE TAMING OF THE SHREW",
            "THE WINTER’S TALE",
            "A LOVER’S COMPLAINT",
        ]
        bodies = [
            "",
            words,
            "Good morning, my friend.\nI hope the sun will shine.",
            "Completely different punctuation and words.",
            words,
            "",
        ]
        source = "    Contents\n\n" + "\n".join("    " + title for title in titles) + "\n\n"
        for title, body in zip(titles, bodies):
            heading = (
                "THE LIFE AND DEATH OF KING RICHARD THE SECOND"
                if title == "KING RICHARD THE SECOND"
                else title
            )
            source += f"\n\n{heading}\n\nDramatis Personae\nKING RICHARD THE SECOND\n\nACT I\n\nSPEAKER.\n{body}\n\n"
        output, report = supplement(base, source)
        self.assertEqual([row["title"] for row in report["included"]], ["A NEW PLAY"])
        self.assertEqual(len(report["excluded"]), 3)
        self.assertIn("SPEAKER:\nGood morning", output)
        self.assertEqual(report["held_out_overlap_windows"], 0)

    def test_overlap_comparison_ignores_case_and_punctuation(self):
        text = "one two three four five six seven eight nine ten"
        self.assertEqual(word_windows(text, 4), word_windows(text.upper().replace(" ", ", "), 4))
