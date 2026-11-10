#!/usr/bin/env python3
"""Run and verify packed NVFP4 matrix multiplication through the SM."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.nvfp4_demo")
