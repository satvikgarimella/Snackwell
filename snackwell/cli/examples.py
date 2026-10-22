#!/usr/bin/env python3
"""Export readable default assembly programs; --check verifies they are current."""

import argparse

from snackwell.assembler import assemble
from snackwell.kernels.inference import NAMES, build_case
from snackwell.paths import ROOT


def examples():
    for name in NAMES:
        case = build_case(name)
        header = [
            "; Shape-specialized example. Edit snackwell/kernels/inference.py and regenerate.",
            f"; Run: python3 scripts/run_kernel.py {name}",
        ]
        header += [
            f"; {key}: byte {start * 4}, {count} words"
            for key, (start, count) in case.layout.regions.items()
        ]
        header += [
            f"; constant at byte {address}: 0x{word:08x}"
            for word, address in case.layout.constants.items()
        ]
        for stage in case.stages:
            assemble(stage.source)
            yield f"{name}/{stage.name}.asm", "\n".join(header) + "\n\n" + stage.source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / "kernels/nn"
    if not args.check:
        output.mkdir(parents=True, exist_ok=True)
    stale = []
    count = 0
    for name, source in examples():
        count += 1
        path = output / name
        if args.check:
            if not path.exists() or path.read_text() != source:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
    if stale:
        parser.exit(1, f"Stale kernel examples: {', '.join(stale)}\n")
    print(f"{'Checked' if args.check else 'Wrote'} {count} assembly examples in {output}")


if __name__ == "__main__":
    main()
