# FP32 and tensor execution speed

This update reduces RTL execution cycles with the existing four-block checkpoint
(`43f211daad7c`). Weights, model dimensions, assembly, FP32 reduction order,
sampling settings and numerical tolerances are unchanged.

This document records the FP32/tensor update. The later multiwarp, broadcast,
and runtime changes are measured separately in [RUNTIME_SPEED.md](RUNTIME_SPEED.md).

## Paired measurement

Prompt: `My lord, I have something to say`; seed: `92617660`; temperature: `0.7`.
Both runs use Verilator without tracing on the development Mac. Times cover one
checked forward pass, excluding compiler/startup time. No profiling or concurrent
regression/evaluation run was active during these two measurements.

| Measurement | Before this RTL update | After |
|---|---:|---:|
| First character, wall time | 22.44 s | 11.21 s |
| Simulated execution cycles | 32,235,401 | 11,847,367 |
| Kernel launches | 1,571 | 1,571 |

That is **2.00× faster** in this paired run and **63.25% fewer simulated cycles**.
All 3,142 memory/register output files match byte for byte. Raw probabilities,
sampling probabilities, predictions, the selected character and error metrics
also match exactly. The preceding [software optimization](INFERENCE_SPEED.md)
has its own historical measurements; its timings are not this run's baseline.

At this revision, the standard 4×4 comparison took **136 cycles for TMUL** and **329 cycles
for SIMD FMUL/FADD**, with matching independent FP32 results and memory guards.
TMUL's adapter takes four gather cycles, one launch cycle, 76 array cycles, four
result-row cycles and one final-commit cycle for those inputs. The earlier TMUL
baseline was 385 cycles, including 281 array cycles and 32+16 serial transfers.

## What changed

**Bounded FP arithmetic.** Add/subtract aligns significands with a sticky barrel
shift and normalizes using a leading-zero detector. The multiplier captures both
operands together, computes the exact 48-bit significand product, normalizes,
handles gradual underflow, then rounds. Each scalar unit completes within four
clocks after acceptance; special cases can complete sooner. Signed zeros,
subnormals, infinities, canonical NaNs and nearest-even rounding remain covered.
The old Dawson multiplier is retained with its notices but is outside the active
datapath. No host floating-point operation substitutes for RTL arithmetic.

**Overlapping multiply and add.** A PE can forward its operands after submitting
the rounded product to its adder. Its next multiplication can then overlap that
addition. A fast product waits if the adder is still occupied; a just-completed
sum can feed the next addition directly. The array drains outstanding additions
after its tenth transport step before presenting C. Products and sums still round
separately and sums retain their original k order.

**Direct tile transfers inside the SM.** The register adapter already gathers
complete A/B tiles. It now passes those buses directly to the array and reads C
as four vector rows. It no longer copies 32 input words through the standalone
word interface or rebuilds rows from 16 serial output words. Destination
reservations still last until the final row commits, including under writeback
backpressure. The standalone word interface remains available and tested.

## Verification

- 38 RTL testbenches and 79 Python test methods passed across the recorded runs.
- The scalar FP bench checked 38,784 results, including every normal exponent,
  randomized inputs, special values, cancellation, the four-clock bound, input
  capture, busy-request rejection and reset in every arithmetic phase.
- PE tests exercise overlapping arithmetic and same-edge sum forwarding. Array
  tests check operand alignment, drain-before-result, pauses and backpressure.
- Full-SM tests cover masks, aliases, dependencies, competing completions and
  final-row commit. Icarus and Verilator agree on memory, registers and traces.
- All 256 saved evaluation positions match the previous RTL probabilities and
  predictions exactly; held-out loss remains 1.6053 and accuracy remains 140/256.
- A real interactive terminal run generated four checked characters (` aga`),
  with all probabilities and selections identical to the saved earlier RTL run.

The initial full regression exposed one test's fixed 12-clock assumption, which
was replaced by observing the early lane's actual completion; its focused rerun
passed. The sandbox also blocked the four HTTP tests at localhost bind; all four
passed with loopback access. The initial failed run is preserved, alongside the
successful focused reruns and the combined result, rather than being reported as
one uninterrupted green invocation.

Run artifacts are under `build/fp32-speed/`. The permanent
[verification record](../models/character_transformer/rtl_performance_verification.json)
contains source hashes, compiler configuration and measurement/evaluation data.

## Reproduce

Exit an old interactive process with `/quit` before restarting, since it may
still hold a previously compiled RTL executable.

```sh
python3 scripts/run_model.py --interactive --tokens 4 --seed 92617660
```

Enter `My lord, I have something to say`. Four new characters are only the
beginning of the continuation; use `/tokens 16` for a longer check.

This remains slow RTL simulation, not real-time generation. The software host
still moves data between 1,571 launches, and each character recomputes the sliding
context. We did not add a KV cache that changes the learned-position semantics.
Wider shifts and detectors trade combinational logic for fewer cycles; FPGA area,
maximum clock frequency and throughput still require synthesis/timing evidence.
Training on Snackwell and FPGA deployment remain later milestones.
