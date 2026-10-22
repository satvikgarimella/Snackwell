"""Convert between Python numbers and raw IEEE 754 binary32 words."""

import struct

SENTINEL = 0xDEADBEEF
SHARED_WORDS = 1024


def bits(number: float) -> int:
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError("tensor elements must be numbers")
    return struct.unpack("<I", struct.pack("<f", number))[0]


def value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]
