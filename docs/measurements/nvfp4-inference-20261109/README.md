# NVFP4 inference evidence — November 9, 2026

These are new mixed-precision model measurements, separate from the earlier
matrix/projection demonstrations. See [the integration contract](../../NVFP4_INFERENCE.md).

```sh
build/model-venv/bin/python scripts/compare_nvfp4_model.py --cases 8
build/model-venv/bin/python scripts/compare_nvfp4_model.py \
  --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'ROMEO:' --prompt 'My lord,' --tokens 4 \
  --out build/inference/nvfp4-character-comparison
build/model-venv/bin/python -m unittest tests.test_nvfp4_model -v
build/model-venv/bin/python scripts/run_tests.py
```

The comparison commands also run with ordinary `python3`; the project venv is
used here so the full regression includes installed PyTorch dependencies.

- `sentence-comparison.json`: eight test-split questions sampled with seed
  20261109; 56 identical-prefix next-token comparisons plus independent greedy
  answer generation in each mode. Both modes answer 8/8 correctly. Top-1
  agreement is 56/56. Mean probability total variation is 0.002058; mean
  target-token NLL rises from 0.000164 to 0.002316.
- `character-comparison.json`: two prompts, eight paired next-character
  checks, 6/8 top-1 agreement, mean probability total variation 0.191159.
  Greedy continuations differ. This is not a language-quality acceptance test.
- `sources.json`: SHA-256 fingerprints for the new/changed runtime, numerical
  reference, kernel emitter, profiler and integration-test sources.

Both model modes use RTL SHA-256
`e96b19fc9a94aa79c246795f7697132776a59f876764f8c6df3943634421a5c0`.
Checkpoint hashes, prompts, teacher-forced input IDs, predictions, losses,
distribution errors, cycle counts and wall times are retained in the reports.
Wall timings include verification and system load; they do not establish a
hardware throughput improvement. Full launch results remain under the two
`build/inference/` directories used above.

The focused seven-test suite covers padding, row independence, bias, K chunks,
native/Python oracle parity, Icarus/Verilator parity, actual native opcode
issue, cache policy rejection/rebuilds, injected corruption and tensor-status
failures. A traced test found that the profiler omitted tensor-controller
state 5 (`PREPARE`); adding its explicit phase classification fixed the failure.
The numerical and execution checks were not relaxed.

Final regression: **46/46 RTL testbenches and 234 Python test methods passed**,
with no skips. The Python suite took 361.763 seconds. `regression.json`,
`regression.log`, and `python-tests.log` retain the results. Ruff checks and
format verification passed for 134 Python files; `git diff --check` passed.
