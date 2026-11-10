# NVFP4 validation, November 9, 2026

The final complete regression passed **46 RTL testbenches and 227 Python test
methods, with no skips**. The Python portion took 363.341 seconds. Existing
inference and training checks ran in the environment with PyTorch installed;
localhost socket permission was granted for the existing HTTP tests.

```sh
build/model-venv/bin/python scripts/run_tests.py
make lint check-examples RUFF=build/dev-venv/bin/ruff
python3 scripts/run_nvfp4_demo.py
python3 scripts/run_nvfp4_demo.py \
  --checkpoint models/character_transformer/checkpoint.json --prompt 'ROMEO:'
```

The [suite result](suite-result.json), [Python log](python.log), and
[validation summary](validation-summary.json) record the final run.
[Source hashes](source-sha256.txt) identify 268 local implementation, simulator,
test, kernel, and FPGA-project files, including earlier uncommitted milestones.

## Arithmetic and integration

- [Product unit](product-unit.log): 37,888 vectors cover every E2M1 element and
  scale byte on both operands, selected boundary-scale combinations, random
  products, signed zeros, and rejected scale encodings.
- [Tensor controller](tensor-controller.log): operand capture under delayed
  memory/array handshakes, accumulator carry between four internal passes,
  completion only after the final pass, ownership and reset.
- Twelve new Python methods check conversion midpoints and saturation, two-level
  quantization, partial/nonuniform commands, multiple scale groups and output
  tiles, packed strides/bounds, initial FP32 values and rounding, rejected scales,
  failure propagation across warps, guarded scale words, and buffer reuse during
  arithmetic. All memory words are compared, including guards and status/tickets.
- [FP32 result](decoded-fp32-result.json) and [NVFP4 result](decoded-nvfp4-result.json)
  belong to a parity test using the same decoded values. The test checks full
  output images and exact agreement between the two arithmetic paths.
- [Verilator control](verilator-nvfp4-control.log) and
  [kernel metrics](verilator-nvfp4-result.json) accompany exact Icarus/Verilator
  parity of the NVFP4 matrix program. The existing [device control bench](device-control.log)
  continues to cover abort, watchdog, draining, coordinated reset, and relaunch.

## Matrix and checkpoint experiments

| Experiment | Serial cycles | Overlapped cycles | Narrow MACs | Relative L2 error vs original FP32 inputs |
| --- | ---: | ---: | ---: | ---: |
| A[4,32] @ B[32,8] | 2,350 | 1,656 | 1,024 | 13.65% |
| A[4,64] @ B[64,8] query projection | 4,380 | 2,798 | 2,048 | 9.18% |

Both experiments match their quantized reference bit for bit. The error column
measures quantization, independently of RTL correctness. Serial and overlapped
schedules use identical inputs and external latency; the table is not an NVFP4
versus FP32 throughput comparison. Neither schedule launches the array's FP32
multipliers; SIMD FP32 multiplication still applies the tensor-wide epilogue.

- Matrix: [result](matrix-result.json), [inputs](matrix-inputs.json),
  [serial assembly](matrix-serial.asm), [overlapped assembly](matrix-overlapped.asm),
  [serial trace](matrix-serial.csv), [overlapped trace](matrix-overlapped.csv),
  [terminal log](matrix-demo.log). There are 538 cycles of copy/arithmetic
  overlap, including 99 actual copy read/write handshakes.
- Projection: [result](projection-result.json), [inputs](projection-inputs.json),
  [serial trace](projection-serial.csv), [overlapped trace](projection-overlapped.csv),
  [terminal log](projection-demo.log). There are 1,277 overlap cycles and 233
  copy handshakes during arithmetic.

The projection uses checkpoint `43f211daad7c` and the last four positions of
`ROMEO:`. Host-prepared normalized embeddings multiply the first eight columns
of the first block's query weights, before bias. The tensor-wide weight scale
uses the full query-weight tensor. This is one operation slice; full-model
quality and inference/training migration to this path remain open.

## Simulator launch failure and follow-up

The [initial full run](initial-suite-result.json) passed all 46 RTL benches and
226 Python methods; one Verilator launch failed without diagnostic output.
The [initial Python log](initial-python.log) and empty
[control log](initial-verilator-control.log) preserve the failure. The earlier
command helper did not record its exit code, so the cause is unconfirmed.

The same executable then passed a [diagnostic relaunch](verilator-launch-diagnostic.log).
Tests now remove the old executable before linking a new one while retaining
the generated C++ cache. The command helper records nonzero exit codes.
Three fresh-link/immediate-launch checks passed
([1](fresh-relink-1.log), [2](fresh-relink-2.log), [3](fresh-relink-3.log)), followed
by the complete passing regression above. Failures are not automatically
retried or ignored by the tests.

[Python lint and example checks](lint-and-examples.log) passed for 128 Python
files and 36 generated examples. [Verilator structural lint](verilator-lint.log)
passed; two existing incomplete-case warnings in reconvergence control remain.
The Quartus project lists all 42 RTL sources exactly once. No new synthesis,
FPGA fit, timing closure, or physical board execution is claimed.
