#!/usr/bin/env python3
"""Compile and run every Snackwell testbench with Icarus Verilog."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.tests")
