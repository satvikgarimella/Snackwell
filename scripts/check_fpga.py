#!/usr/bin/env python3
"""Check the FPGA host interface in RTL simulation (no physical board required)."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.fpga")
