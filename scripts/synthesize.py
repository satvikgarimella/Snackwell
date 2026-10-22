#!/usr/bin/env python3
"""Synthesize the full Snackwell device and optionally test its gate netlist."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.synthesis")
