"""Dataset leakage and answer-token boundaries for the bounded language demo."""

import unittest

from snackwell.sentence_model import validate_prompt
from snackwell.sentence_task import Example, encode_example, examples, pair_split
from snackwell.tokenizer import Tokenizer


class SentenceTaskTests(unittest.TestCase):
    def test_demo_rejects_questions_outside_its_trained_scope(self):
        for prompt in (
            "My name is",
            "The ball is in the kitchen. The book is in the garden. Where is the cup?",
            "The ball is in the kitchen. The ball is in the garden. Where is the ball?",
            "The dog is in the kitchen. The book is in the garden. Where is the dog?",
            "The ball is in the kitchen. The book is in the kitchen. Where is the ball?",
        ):
            with self.assertRaises(ValueError):
                validate_prompt(prompt)
        for example in examples("test"):
            validate_prompt(example.prompt)

    def test_no_reserved_pairs_or_duplicate_prompts_in_training(self):
        seen = set()
        for split in ("train", "validation", "test"):
            data = examples(split)
            prompts = [e.prompt for e in data]
            self.assertEqual(len(prompts), len(set(prompts)))
            self.assertFalse(seen.intersection(prompts))
            seen.update(prompts)
            for example in data:
                self.assertEqual(
                    pair_split(example.query, dict(example.facts)[example.query]), split
                )
                for obj, location in example.facts:
                    if split == "train" or obj != example.query:
                        self.assertEqual(pair_split(obj, location), "train")
                self.assertEqual(len(set(obj for obj, _ in example.facts)), 2)
                self.assertEqual(len(set(room for _, room in example.facts)), 2)

    def test_balanced_query_order_and_counterfactual_locations(self):
        for split in ("train", "validation", "test"):
            data = examples(split)
            first = sum(e.facts[0][0] == e.query for e in data)
            self.assertEqual(first * 2, len(data))
        first = Example((("ball", "kitchen"), ("book", "garden")), "ball")
        swapped = Example((("ball", "garden"), ("book", "kitchen")), "ball")
        self.assertNotEqual(first.answer, swapped.answer)

    def test_training_tokenizer_handles_all_splits_without_crossing_boundary(self):
        tokenizer = Tokenizer.train("\n".join(e.prompt + e.answer for e in examples("train")), 256)
        for split in ("train", "validation", "test"):
            for example in examples(split):
                ids, targets = encode_example(tokenizer, example)
                self.assertEqual(len(ids), len(targets))
                self.assertLessEqual(len(ids), 32)
                self.assertEqual(
                    tokenizer.decode([x for x in targets if x != -100]), example.answer
                )
                self.assertEqual(tokenizer.decode(tokenizer.encode(example.prompt)), example.prompt)


if __name__ == "__main__":
    unittest.main()
