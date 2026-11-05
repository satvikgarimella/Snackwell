# Inference transport

`run_model.py` now defaults to summary artifacts with Verilator. Python sends
each assembled program, its 1,024 raw memory words and launch controls through
a pipe. The C++ driver resets and clocks the same SM, then returns all memory
words, registers and the cycle count. It performs no model arithmetic.

This removes repeated filesystem work between small kernels. It does not change
the generated assembly, SM scheduling, FP32 rounding or simulated cycle count.
It is a simulation-host improvement, not an FPGA speed measurement or a global
memory implementation.

Every kernel still runs the independent expected-value calculation and checks
its output. Each token's `result.json` retains all stage reports, numerical error
measurements, cycles, checkpoint identity and simulator source fingerprint.
Layer outputs and probabilities are also retained. A failed kernel saves its
assembly, input image, layout and launch controls; if RTL completed, its actual
output and registers are saved too. Failure never publishes a verified token.

```sh
python3 scripts/run_model.py --interactive --tokens 32
```

For full assembly and memory dumps for successful kernels as well:

```sh
python3 scripts/run_model.py --artifacts full --prompt 'ROMEO:' --tokens 4
```

`--trace` and the Icarus backend retain the full artifact path. The library API
also keeps full artifacts unless `compile_model_runner(..., compact=True)` is
explicitly requested. Existing scripts that inspect individual kernel files can
continue using full mode.

## Measured latency

These October 27 measurements use the same `ROMEO:` prompt, four generated tokens,
seed 92617660 and temperature 0.7. Each model runs in the order full, summary,
summary, full after training and evaluation jobs finish. Values are means across
the two runs per mode; cached decode averages the three subsequent tokens in
each run. They exclude checkpoint loading and compilation. A longer prompt or
cache rebuild can take substantially longer.

| Checkpoint | First output, full → summary | Cached output, full → summary |
| --- | --- | --- |
| Default character model | 1.35 → 0.84 seconds | 0.68 → 0.34 seconds/character |
| Subword model | 3.56 → 2.02 seconds | 2.46 → 1.30 seconds/token |
| Eight-block Shakespeare model | 9.88 → 5.59 seconds | 6.22 → 3.27 seconds/token |

All checked outputs, intermediate stage reports, selections and cycle counts
match exactly across modes and repeats. [Raw measurements](measurements/inference-transport.json)
include checkpoint hashes, simulator fingerprint, individual timings and result
hashes. This is approximately a 1.9–2.0× improvement in cached simulation
throughput. It does not improve the models' generated language.

## Verification

Transport tests compare every output word, register and cycle against Icarus and
the file-based Verilator path, including FP32 edge cases, masked lanes, tensor
rounding, nonzero entry PCs, resets and complete transformer inference. They
also check truncated requests, failed kernels, timeout recovery and switching
between traced and untraced execution. The terminal and KV-cache suites pass,
as do checks for complete training-step parity, finite-difference gradients and
inference using an updated checkpoint.

The simulator fingerprint includes its C++ host; changing that fingerprint here
does not imply a change to the SM RTL.
