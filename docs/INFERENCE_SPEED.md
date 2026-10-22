# Inference speed investigation

This page records the first optimization pass, before the FP/tensor RTL changes.
For current measurements and architecture changes, see [FP32 and tensor speed](FP32_SPEED.md).

The four-block checkpoint is unchanged (`43f211daad7c`). This work reduces
simulation overhead and matrix-kernel memory traffic while keeping the same
FP32 reduction order, numerical tolerances and sampling settings.

## Controlled measurement

The same prompt, checkpoint, seed 92617660, temperature 0.7 and Verilator backend
were used before and after. The prompt was `My lord, I have something to say`.
These are unprofiled, warm-build forward-pass times on the development Mac;
first compilation and CLI startup are excluded.

| Measurement | Before | After |
|---|---:|---:|
| Time for the first character | 36.24 s | 20.65 s |
| Simulated execution cycles | 34,236,950 | 32,235,401 |
| Kernel launches | 1,571 | 1,571 |
| Selected character | space | space |

The measured pass is 1.75 times as fast, with 43% less wall time and 5.8% fewer
execution cycles. Probabilities, sampling probabilities, greedy predictions,
selected token and numerical error metrics match the baseline exactly.
The run files are in `build/inference-speed/baseline-unprofiled` and
`build/inference-speed/optimized-native`. Profiled timings are stored separately
and are not used to calculate the speedup.

## Verification results

- All 37 RTL testbenches passed.
- All 79 Python test methods passed: 75 inside the sandbox, plus four local
  HTTP tests rerun with loopback permission after sandbox setup was denied.
- All 256 positions in the eight saved evaluation windows passed, with
  probabilities and predictions identical to the earlier RTL implementation.
- Four characters generated through interactive terminal mode matched the
  saved RTL sequence, including every raw/sampling probability and selection.
- The native scalar oracle matched Python on 68,487 edge, random and
  cancellation cases. Matrix tests also covered transposition and carried sums.
- Formatting, lint and all 36 generated assembly examples were checked.

The [verification record](../models/character_transformer/performance_verification.json)
stores source fingerprints, compiler settings, measured times and artifact paths.
Raw test logs, including the sandbox failure and successful focused rerun, are
preserved under `build/inference-speed/`.

## Changes and reasons

### Keep partial matrix sums in registers

The original tiled matrix kernel loaded and stored C after every four-term
product. R8 through R11 now hold the accumulated output tile throughout a K
chunk. `TMUL R0, R0, R4` gathers A before overwriting it, then four FADDs update
those accumulators. Only the completed tile is stored. A later K chunk loads
the preceding C words once and continues in the same order.

For a 16-by-16 multiplication, this reduces stores from 1,280 words to 256 and
loads from 3,072 words to 2,048. The 64 TMULs and FP32 arithmetic are unchanged.
The generated program shrinks from 625 instructions to 561. This change reduces
GPU execution cycles, independently of simulator overhead.

### Reuse the simulator process

The old runtime started one executable per kernel. A forward pass now uses one
Verilator process, sending a directory and launch controls over stdin. Each
request still resets the DUT, loads complete instruction/shared-memory images,
launches the selected warps, waits for completion, checks outstanding work and
exports memory and registers. This is a simulation host protocol, not a new
hardware launch interface or global-memory implementation.

Tests compare repeated sessions with independent Icarus launches, including
register reset, warp/lane masks, paths containing spaces, trace equality and
failure recovery. A failed run removes stale outputs. Context management closes
the simulator on success, errors and Ctrl+C. Single-launch execution remains
available to the kernel runner and Icarus fallback. Verilator's generated C++ is
also built with `-O3` instead of its default size optimization.

### Speed up verification without removing it

The Python matrix oracle performed roughly nine million scalar integer FP32
operations per character. An optional C++ oracle computes the same exact
integer arithmetic using five 64-bit limbs, enough to represent every aligned
binary32 sum. It implements nearest-even rounding, signed zeros, subnormals,
infinities and NaNs without host floating-point arithmetic or RTL code.

The C++ oracle is checked against the original Python oracle on random bit
patterns, special values, every normal exponent, cancellation, transposed
matrices and carried sums. If no compiler is available, Python verification
remains available. The whole-model scalar reference is still Python. Expected
outputs are only compared with RTL results; they never enter the simulated
model's memory or activations. Approximate-operation tolerances are unchanged.

Repeated assembly text is cached separately from input data. This avoids
reassembling identical kernel shapes; weights, activations and results are not
cached.

## Limits at this milestone

Twenty seconds per character is still slow. It is a measured improvement to
verified RTL simulation, not interactive chatbot speed or an FPGA throughput
measurement. Language quality is unchanged. Matrix arithmetic still dominates
the remaining cycles; the array waits for each multi-cycle step to finish.

We did not add a naive KV cache: the current model shifts a fixed window and
resets learned position indices after each character. Reusing old keys and
values under those semantics would change the model's calculations. Any future
cache needs an explicit position policy and equivalence tests.

The subsequent [RTL update](FP32_SPEED.md) addresses tensor scheduling and FP
latency. A real memory/launch interface and FPGA timing/resource measurements
remain needed to translate cycle improvements into hardware throughput.
