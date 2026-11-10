# Tensor-memory milestone validation, November 7, 2026

[validation-summary.json](validation-summary.json) records 40 passing RTL benches
and 195 distinct passing Python methods across runs, plus hashes of the tested
hardware, program, and integration tests. There are no unresolved skips.
This is RTL simulation evidence; no new physical FPGA fit is claimed.

- [Initial suite result](initial-suite-result.json) retains `passed: false`.
  All 40 RTL benches passed. The Python run executed 189 methods: 175 passed
  and 14 skipped because its interpreter lacked PyTorch. HTTP test class setup
  was blocked by the sandbox's loopback socket restriction.
- [Initial Python log](initial-python.log) preserves those details.
- [Environment follow-up](environment-followup.log) passed the 14 previously
  skipped methods plus all four HTTP methods, using the existing
  `build/model-venv/bin/python` and granted network access for the local server.
- [Final tensor tests](tensor-memory-final.log) passed all ten new tensor-memory
  tests. Eight were already included in the initial run; public-device-bus and
  Verilator/persistent-session checks were added afterward. Hence the distinct
  passing-method count is `175 + 18 + 2 = 195`, not the sum of all repetitions.
- [Demo result](demo-result.json) records the 4×8 by 8×4 product: 250 simulated
  cycles, zero tensor status, exact numerical agreement, and intact memory guards.
- [Verilator lint](verilator-lint.log) records no errors or new structural
  hazards. Two existing incomplete-case warnings are in reconvergence control.

Additional checks passed: `make lint check-examples RUFF=build/dev-venv/bin/ruff`
(115 formatted Python files and 36 generated assembly examples), and
`git diff --check`.

To repeat the full suite with optional PyTorch coverage, use:

```sh
build/model-venv/bin/python scripts/run_tests.py
```

The HTTP tests need permission to bind a local loopback socket. The summary is
an aggregate of the recorded runs, not a claim of one uninterrupted green run.
