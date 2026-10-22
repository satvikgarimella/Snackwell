#!/usr/bin/env python3
"""Compare cached and uncached RTL execution of identical character contexts."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.benchmark_cache")
