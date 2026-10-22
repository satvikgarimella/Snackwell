#!/usr/bin/env python3
"""Compare repetition in CPU-reference generation; this does not execute RTL."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.generation_quality")
