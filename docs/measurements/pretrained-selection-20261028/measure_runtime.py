"""Measure representative real RTL launches; no language inference claim."""

import json
import time
from pathlib import Path

from snackwell.numeric import bits
from snackwell.pretrained_runtime import PretrainedRuntime
from snackwell.simulator import SimulatorSession, compile_model_runner

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "build/pretrained-selection-20261028/cost"


def main():
    simulator = compile_model_runner(OUT, "verilator", compact=True)
    results = []
    with SimulatorSession(simulator) as session:
        for width, columns in ((256, 1024), (576, 1024), (576, 1536)):
            runtime = PretrainedRuntime(OUT / f"matvec-{width}-{columns}", session)
            x = [bits((i % 13 - 6) / 16) for i in range(width)]
            weights = [bits((i % 31 - 15) / 128) for i in range(width * columns)]
            start = time.perf_counter()
            runtime.matmul("matvec", x, weights, 1, columns, width, transpose_b=True)
            results.append(dict(width=width, columns=columns, multiply_adds=width * columns,
                                seconds=time.perf_counter() - start,
                                cycles=sum(r['cycles'] for r in runtime.reports),
                                launches=len(runtime.reports),
                                verified=all(r['verified'] for r in runtime.reports),
                                max_words=max(r['allocated_words'] for r in runtime.reports),
                                max_instructions=max(r['instructions'] for r in runtime.reports)))
            print(results[-1], flush=True)
    report = dict(kind="Verified RTL matvec microbenchmarks; not full-model latency",
                  rtl_sha256=simulator.source_sha256, backend=simulator.backend, measurements=results)
    (Path(__file__).parent / "rtl-cost.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
