# Global-memory validation, November 7, 2026

[validation-summary.json](validation-summary.json) records one complete passing
run of `build/model-venv/bin/python scripts/run_tests.py`: **43 RTL testbenches
and 204 Python test methods, with no skips**. PyTorch was available in the
existing environment, and localhost socket permission was granted for the HTTP
tests. The Python portion took 286.406 seconds.

- [Suite result](suite-result.json) and [Python log](python.log): full regression,
  including existing inference/training tests and nine new global-memory methods.
- [Unit log](unit.log): lane snapshots, address/error semantics, held responses
  and completions, ordering, and reset.
- [SM writeback log](sm-writeback.log): occupied writeback slot, captured warp/lane
  metadata, dependent FP32 arithmetic, and halt draining.
- [Device control log](device-control.log): abort before and after acceptance,
  watchdog, drain, coordinated link reset, and relaunch.
- [Demo result](demo-result.json), [log](demo.log), and [trace](demo-trace.csv):
  4×8 by 8×4 matrix multiplication through global/shared/tensor/global memory,
  with 64 reads, 16 writes, two matrix jobs, exact FP32 agreement, and all
  surrounding memory guards intact. Default timing: 1,013 simulated cycles.
- [Verilator lint](verilator-lint.log): no new structural warnings; two existing
  incomplete-case warnings in reconvergence control remain.

The independent random matrix case returned identical bits at configured memory
latencies 0, 3, and 19, taking 884, 1,088, and 2,392 simulated cycles respectively.
These settings are additional endpoint delay, not total access latency. Another
test issued 31 independent integer instructions while a global load waited.
The Python suite includes Icarus/Verilator comparison of the external-memory
host-bus runner and the Verilator abort/relaunch tests.

Additional checks passed: Ruff lint/format (119 Python files), 36 generated
assembly examples, `git diff --check`, and a source/port manifest comparison for
the Quartus project. The virtual-pin list now includes the global-memory bus.

Source hashes identify the tested local tree, which includes the previous
tensor-memory milestone. No new FPGA synthesis, Quartus fit, timing closure,
bitstream, or board execution is claimed. The simulated 64 KiB RAM is a test
endpoint, not synthesized on-chip storage.
