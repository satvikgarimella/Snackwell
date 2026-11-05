#!/usr/bin/env python3
"""Prepare additional Shakespeare plays for host training, keeping fixed evaluation splits."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snackwell.shakespeare_data import supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gutenberg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text, report = supplement(args.base.read_text(), args.gutenberg.read_text())
    report.update(
        source_url="https://www.gutenberg.org/ebooks/100",
        source_file_sha256=hashlib.sha256(args.gutenberg.read_bytes()).hexdigest(),
        note="Training supplement only. Base vocabulary, tokenizer and evaluation splits stay fixed.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(text)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(report['included'])} plays, {len(text):,} characters: {args.out}")
    print(f"Excluded {len(report['excluded'])} plays; no 32-word overlaps with held-out text.")


if __name__ == "__main__":
    main()
