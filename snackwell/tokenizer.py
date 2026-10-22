"""Deterministic word-bounded BPE. Learn merges from the training split only."""

import collections
import functools
import re

PATTERN = r" ?[A-Za-z]+|[^A-Za-z]"
FORMAT = "snackwell-bpe-v1"


def merge(sequence, pair, replacement):
    result = []
    i = 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i], sequence[i + 1]) == pair:
            result.append(replacement)
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return tuple(result)


class Tokenizer:
    def __init__(self, vocabulary, merges, alphabet_size):
        self.vocabulary = vocabulary
        self.merges = [tuple(p) for p in merges]
        self.alphabet_size = alphabet_size
        self.alphabet = {c: i for i, c in enumerate(vocabulary[:alphabet_size])}
        self.ranks = {p: alphabet_size + i for i, p in enumerate(self.merges)}

    @classmethod
    def train(cls, text, size):
        vocabulary = sorted(set(text))
        if type(size) is not int or not 2 <= len(vocabulary) <= size <= 512:
            raise ValueError("BPE requires 2..512 characters and a vocabulary limit up to 512")
        alphabet_size = len(vocabulary)
        alphabet = {c: i for i, c in enumerate(vocabulary)}
        frequencies = collections.Counter(re.findall(PATTERN, text))
        words = [tuple(alphabet[c] for c in word) for word in frequencies]
        counts = list(frequencies.values())
        pairs = collections.Counter()
        locations = collections.defaultdict(set)
        for i, word in enumerate(words):
            for pair, count in collections.Counter(zip(word, word[1:])).items():
                pairs[pair] += count * counts[i]
                locations[pair].add(i)
        merges = []
        known_tokens = set(vocabulary)
        while len(vocabulary) < size and pairs:
            choices = [p for p in pairs if vocabulary[p[0]] + vocabulary[p[1]] not in known_tokens]
            if not choices:
                break
            pair = min(choices, key=lambda p: (-pairs[p], p))
            if pairs[pair] < 2:
                break
            replacement = len(vocabulary)
            vocabulary.append(vocabulary[pair[0]] + vocabulary[pair[1]])
            known_tokens.add(vocabulary[-1])
            merges.append(pair)
            for i in list(locations[pair]):
                old = words[i]
                new = merge(old, pair, replacement)
                for p, count in collections.Counter(zip(old, old[1:])).items():
                    pairs[p] -= count * counts[i]
                    if not pairs[p]:
                        del pairs[p]
                    locations[p].discard(i)
                for p, count in collections.Counter(zip(new, new[1:])).items():
                    pairs[p] += count * counts[i]
                    locations[p].add(i)
                words[i] = new
        return cls(vocabulary, merges, alphabet_size)

    @functools.lru_cache(maxsize=65536)
    def unit(self, word):
        result = tuple(self.alphabet[c] for c in word)
        while len(result) > 1:
            available = [p for p in zip(result, result[1:]) if p in self.ranks]
            if not available:
                break
            pair = min(available, key=self.ranks.__getitem__)
            result = merge(result, pair, self.ranks[pair])
        return result

    def encode(self, text):
        unknown = sorted(set(text) - self.alphabet.keys())
        if unknown:
            raise ValueError(f"prompt contains characters outside the vocabulary: {unknown!r}")
        return [token for word in re.findall(PATTERN, text) for token in self.unit(word)]

    def decode(self, ids):
        ids = tuple(ids)
        if any(type(i) is not int or not 0 <= i < len(self.vocabulary) for i in ids):
            raise ValueError("token ID outside vocabulary")
        return "".join(self.vocabulary[i] for i in ids)

    def metadata(self):
        return dict(
            kind=FORMAT,
            pattern=PATTERN,
            alphabet_size=self.alphabet_size,
            merges=[list(p) for p in self.merges],
            vocabulary=list(self.vocabulary),
        )

    @classmethod
    def restore(cls, data):
        keys = {"kind", "pattern", "alphabet_size", "merges", "vocabulary"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("invalid tokenizer metadata")
        if data["kind"] != FORMAT or data["pattern"] != PATTERN:
            raise ValueError("unsupported tokenizer format or word boundaries")
        vocabulary, merges, alphabet = data["vocabulary"], data["merges"], data["alphabet_size"]
        if (
            not isinstance(vocabulary, list)
            or not 2 <= len(vocabulary) <= 512
            or any(not isinstance(t, str) or not t for t in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or type(alphabet) is not int
            or not 2 <= alphabet <= len(vocabulary)
            or any(len(c) != 1 for c in vocabulary[:alphabet])
            or not isinstance(merges, list)
            or len(merges) != len(vocabulary) - alphabet
        ):
            raise ValueError("invalid BPE vocabulary or alphabet")
        seen = set()
        for index, pair in enumerate(merges, alphabet):
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(type(i) is not int or not 0 <= i < index for i in pair)
                or tuple(pair) in seen
            ):
                raise ValueError("BPE merges must reference earlier tokens exactly once")
            token = vocabulary[pair[0]] + vocabulary[pair[1]]
            if token != vocabulary[index] or re.fullmatch(PATTERN, token) is None:
                raise ValueError("BPE token does not match its merge or word boundary")
            seen.add(tuple(pair))
        return cls(list(vocabulary), merges, alphabet)
