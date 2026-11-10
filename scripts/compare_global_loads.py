#!/usr/bin/env python3
"""Measure same-address load broadcasting with identical kernels and model weights."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.compare_global_loads")
