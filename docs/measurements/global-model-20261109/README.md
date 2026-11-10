# Global-memory model evidence — November 9, 2026

See [the runtime contract](../../GLOBAL_MODEL.md) for transport boundaries and
the remaining host responsibilities.

```sh
build/model-venv/bin/python scripts/compare_model_memory.py --cases 8
build/model-venv/bin/python scripts/compare_model_memory.py \
  --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'ROMEO:' --tokens 4 \
  --out build/inference/global-character-comparison
build/model-venv/bin/python -m unittest tests.test_global_model -v
build/model-venv/bin/python scripts/run_tests.py
```

The comparison commands also work with ordinary `python3`; the project venv
includes the optional dependencies used by the full regression.

- `sentence-comparison.json`: eight reserved test questions, seed 20261109.
  All 56 generated-token probability vectors and selections match between
  shared/global transport. Both modes answer 8/8 correctly. Global execution
  consumes 67,163,967 cycles versus 27,453,019 with direct shared-memory input.
  It completes 5,302,096 reads and 673,152 writes, with 5,975,248 responses.
- `character-comparison.json`: `ROMEO:`, four greedy tokens, identical
  probability vectors and continuation `\nMar` in both transports. Global
  execution consumes 7,552,044 cycles versus 3,477,228 shared.
- `sources.json`: fingerprints of the simulator transport, external endpoint,
  model orchestration, kernel generator, command-line entry points and tests.

Both comparisons use NVFP4 linear projections with FP32 attention and nonlinear
kernels, the unchanged `nvfp4-linear-row-v1` quantization policy, and latency 2.
They establish transport equivalence, not a new language-quality result or a
speedup. Model quality against FP32 remains documented in the separate
[precision comparison](../nvfp4-inference-20261109/README.md).

RTL/simulation source SHA-256:
`a96505f467e2ab25d15a221bec8c16234960fecc147152926cf2e39c39953a86`.
Full per-token stage reports remain in the two `build/inference/` directories
above. Reported wall times include quantization, verification and system load.

The eight focused integration tests exercise ordinary multiwarp kernels and
partial masks, native tile/K-chunk streaming, several memory latencies, cached
decode and rebuilds, sampling, mode isolation, fault/status handling, timeout
and restart, guard corruption, serial/overlapped copies, and both simulator
transports. Numerical acceptance criteria were not relaxed.

The first full run passed all 46 RTL benches and 241 of 242 Python methods.
One compatibility test rejected extra memory metadata in the default shared
report. The default report schema was restored; global mode alone adds those
fields. `initial-python-tests.log` and `initial-regression.log` retain the
failure. The failing case and all eight new integration tests then passed
together (`focused-final.log`). Additional artifact checks confirm that saved
global runs initialize shared memory with guard words, not model operands.

`terminal-smoke.log` records the public sentence CLI with
`--precision nvfp4 --memory global`. Its answer was ` The ball is in the kitchen.`,
with 8,395,437 simulated cycles and 1,253 launches. This also checks that the CLI
actually selects the external-memory runtime rather than merely labeling it.

Final regression: **46/46 RTL benches and 242 Python test methods passed**, with
no skips. The Python suite took 378.940 seconds. `regression.json`,
`regression.log` and `python-tests.log` retain the complete successful run.
Ruff checks/format verification passed for 138 Python files, and
`git diff --check` passed. Final source fingerprints match `sources.json`.
