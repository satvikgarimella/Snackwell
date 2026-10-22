#!/usr/bin/env python3
"""Assemble, execute, verify, and profile Snackwell kernels in RTL simulation."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.kernel")
