"""Run a checkout-local module from any working directory, without installation."""

import runpy
import sys
from pathlib import Path


def run(module: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module(module, run_name="__main__")
