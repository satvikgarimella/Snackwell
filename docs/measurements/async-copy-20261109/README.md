# Asynchronous-copy validation, November 9, 2026

The full regression passed **45 RTL testbenches and 215 Python test methods,
with no skips**. This includes the existing inference/training checks and 11
new asynchronous-copy methods. The Python suite took 335.331 seconds. It ran
with the existing PyTorch environment and permission for the localhost HTTP
tests:

```sh
build/model-venv/bin/python scripts/run_tests.py
make lint check-examples RUFF=build/dev-venv/bin/ruff
python3 scripts/run_async_demo.py
```

- [Suite result](suite-result.json) and [Python log](python.log) record the
  complete regression. [Validation summary](validation-summary.json) identifies
  the checks and local source manifest.
- [Copy unit](copy-unit.log), [external-memory arbiter](arbiter.log), and
  [device-control log](device-control.log) cover held requests/responses,
  descriptor capture, ticket generations, consumer ownership, failed reads,
  abort/watchdog draining, coordinated reset, and relaunch.
- [Demo comparison](demo-result.json), [serial trace](serial-trace.csv), and
  [overlapped trace](overlapped-trace.csv) show the same A[4,12] @ B[12,4]
  computation in **1,582 versus 1,412 simulated cycles**: 170 fewer cycles,
  or 10.7%. The overlapped schedule has 139 cycles with copying and PE
  arithmetic active together, including 25 actual copy read/write handshakes.
- Both schedules check every shared/global memory word against an independent
  expected image, including the FP32 product, guard words, statuses, and
  generation tickets. Each performs three matrix jobs, 96 external reads,
  96 shared copy writes, and 16 external output stores.
- [Fresh Verilator control log](verilator-cold-control.log) and
  [kernel result](verilator-cold-result.json) record a separately rebuilt
  executable launched immediately after compilation. It passed the abort and
  relaunch tests and matched the Icarus output images and 1,412-cycle timing.
  The full regression subsequently passed the regular Verilator parity test.
- [Structural lint](verilator-lint.log) contains two existing incomplete-case
  warnings in reconvergence control; the structural-hazard checks passed.
  [Python lint/example checks](lint-and-examples.log) passed for 123 Python
  files and 36 generated assembly examples. The Quartus source manifest matches
  all 41 RTL implementation files.

Two earlier Verilator launches immediately after rebuilding the usual executable
ended without diagnostic output. The cause was not established. A repeat of
the focused test, the separate fresh build, and the final full regression all
passed; no Icarus/Verilator numerical or protocol mismatch was observed in those
passing runs.

The multiple-consumer case retains a scheduler regression: resource-stalled
warps must not prevent another eligible warp from releasing their buffer.
Randomized rectangle/matrix tests also cover different strides, multiple buffer
generations, delayed memory, normal-LSU contention, and live operand snapshots.

[SHA-256 source hashes](source-sha256.txt) identify the tested local RTL,
simulation, Python, test, kernel, and FPGA project sources, including earlier
uncommitted tensor-memory/global-memory work. The timing comparison uses two
additional endpoint-delay cycles and simulated request backpressure; it is not
a physical-memory or FPGA measurement. Existing model workloads still use the
earlier data path. No new synthesis, FPGA fit, timing closure, or board execution
is claimed.
