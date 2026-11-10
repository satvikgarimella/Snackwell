#!/usr/bin/env python3
"""Compare verified native NVFP4 inference with the unchanged FP32 model."""

from _bootstrap import run

if __name__ == "__main__":
    run("snackwell.cli.compare_nvfp4_model")
