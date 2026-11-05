"""Prepare supplemental dialogue without importing TinyShakespeare's held-out play."""

import hashlib
import re
import unicodedata


def word_windows(text, size=32):
    words = re.findall(r"[a-z]+", unicodedata.normalize("NFKD", text).lower())
    return {
        hashlib.sha256(" ".join(words[i : i + size]).encode()).digest()
        for i in range(len(words) - size + 1)
    }


def plays(text):
    """Read the ebook's main table of contents, then its unindented work headings."""
    contents = re.search(r"(?m)^ +Contents\s*$", text)
    first = text.index("\nTHE SONNETS\n", contents.end())
    titles = [line.strip() for line in text[contents.end() : first].splitlines() if line.strip()]
    last_play = titles.index("THE WINTER’S TALE")
    titles = titles[: last_play + 2]
    aliases = {"KING RICHARD THE SECOND": "THE LIFE AND DEATH OF KING RICHARD THE SECOND"}
    headings = []
    for title in titles:
        heading = aliases.get(title, title)
        matches = list(re.finditer(r"\n{3,}(" + re.escape(heading) + r")\n{2,}", text))
        if len(matches) != 1:
            raise ValueError(f"expected one work heading: {title}")
        headings.append(matches[0])
    starts = [match.start(1) for match in headings]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError("unexpected Gutenberg work headings")
    # Poems bracket the plays in this edition. Keep dialogue plays only.
    return {titles[i]: text[headings[i].end(1) : starts[i + 1]] for i in range(1, last_play + 1)}


def dialogue(text, alphabet):
    beginnings = list(re.finditer(r"(?m)^ACT I\.?\s*$", text))
    if not beginnings:
        raise ValueError("play has no first-act heading")
    # Some play-specific tables of contents also use unindented ACT headings.
    text = re.sub(r"\[.*?\]", "", text[beginnings[-1].start() :], flags=re.S)
    blocks = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or re.match(r"^(ACT |SCENE[ .]|Enter |Exit[ .]|Exeunt|Re-enter )", block):
            continue
        lines = block.splitlines()
        if re.fullmatch(r"[A-Z][A-Z ’',.\-]*", lines[0]):
            lines[0] = lines[0].rstrip(".") + ":"
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    text = text.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "“": "'",
                "”": "'",
                "—": "-",
                "–": "-",
                "…": "...",
                "æ": "ae",
                "Æ": "AE",
                "œ": "oe",
                "Œ": "OE",
            }
        )
    )
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = "".join(c if c in alphabet else " " for c in text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def supplement(base, source):
    train_end = int(len(base) * 0.9)
    alphabet = set(base[:train_end])
    known = word_windows(base)
    held_out = word_windows(base[train_end:])
    included, excluded, chunks = [], [], []
    for title, original in plays(source).items():
        # Both fixed held-out splits are from this play; exclude it in full,
        # regardless of punctuation, spelling or formatting in another edition.
        if title == "THE TAMING OF THE SHREW":
            excluded.append(dict(title=title, reason="entire held-out play"))
            continue
        cleaned = dialogue(original, alphabet)
        overlap = len(word_windows(cleaned) & known)
        if overlap:
            excluded.append(dict(title=title, reason="overlaps base corpus", windows=overlap))
            continue
        included.append(dict(title=title, characters=len(cleaned)))
        chunks.append(cleaned)
    result = "\n".join(chunks)
    if not included or word_windows(result) & held_out:
        raise ValueError("supplement is empty or overlaps held-out text")
    return result, dict(
        included=included,
        excluded=excluded,
        held_out_overlap_windows=0,
        overlap_window_words=32,
        characters=len(result),
        base_sha256=hashlib.sha256(base.encode()).hexdigest(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        output_sha256=hashlib.sha256(result.encode()).hexdigest(),
    )
