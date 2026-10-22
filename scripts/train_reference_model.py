#!/usr/bin/env python3
"""Train/export reference weights on the host; this is NOT Snackwell training."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.train")
