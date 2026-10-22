"""Assemble Snackwell instructions into a padded $readmemh program image."""

import argparse
from pathlib import Path

from snackwell.assembler import AssemblyError, assemble, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        words = assemble(args.source.read_text())
        write_image(args.output, words)
    except (OSError, AssemblyError) as error:
        parser.exit(1, f"Assembly failed: {error}\n")
    print(f"Assembled {len(words)} instructions -> {args.output}")


if __name__ == "__main__":
    main()
