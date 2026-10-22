# Snackwell report walkthrough

Work through these in order. For each choice, explain the constraint, follow one
concrete example through the code, then show the measurement that supports it.
The existing [whiteboard walkthrough](INFERENCE_WALKTHROUGH.md) supplies the
diagrams and tensor shapes for the forward pass.

## 1. What the project demonstrates

Snackwell executes programs on a custom GPU written in SystemVerilog. The full
SM has four warps, four lanes per warp, integer/FP32 execution, a 4×4 systolic
tensor array, divergence/reconvergence, and 4 KB of shared memory. The current
trained-model demonstration runs in RTL simulation.

Trace one instruction from fetch and decode through dependency checks, register
reads, execution and writeback. Then explain why a multi-cycle FP result needs
its warp, destination register and active-lane mask preserved until completion.
Use the [FP32](FP32.md), [tensor](TENSOR.md), and [memory](MEMORY.md) notes to
connect each block to its implementation.

## 2. How hardware became something we could program

The assembler maps readable instructions to Snackwell machine words. Kernels
are programs made from those instructions; they implement operations such as
vector addition, tiled matrix multiplication, LayerNorm, GELU and softmax.

Explain these implementation choices with their tradeoffs:

| Choice | What it enables / costs |
| --- | --- |
| 4×4 `TMUL` tiles | Larger products reuse the tensor array; kernels load tiles and accumulate partial results. |
| Separate FP32 multiplication and addition | Reuses the implemented units; rounding differs from a fused multiply-add. |
| Serialize shared-memory bank conflicts | Conflicting requests complete correctly, with extra service cycles. |
| Approximate exp, reciprocal and reciprocal square root with existing instructions | Supports softmax, GELU and LayerNorm; requires numerical error checks. |
| One active warp for model kernels | Gives each output a clear owner without inter-warp barriers; leaves multiwarp optimization for later. |

Read one generated `program.asm` beside its `layout.json`. Identify the input
addresses, instructions that perform the arithmetic, and where outputs land.
The [kernel guide](INFERENCE_KERNELS.md) documents the numerical algorithms,
rounding order and verification tolerances.

## 3. How kernels became a trained model

First, the [standalone transformer block](TRANSFORMER.md) combines tested
kernels using deterministic fixture weights. The trained model adds character
and position embeddings, a saved checkpoint, final normalization, an output
projection, and character selection.

The current checkpoint has four blocks, four heads per block, 32-character
context, width 64, feed-forward width 128, a 65-character vocabulary, and
144,449 FP32 parameters. It was trained on TinyShakespeare using PyTorch on the
CPU. Training on Snackwell remains future work.

The weights occupy 577,796 bytes, exceeding 4 KB shared memory. The runtime
splits work into bounded launches and copies actual output words between them.
It accumulates K chunks on the SM, retaining the original FP32 summation order.
Prefill computes the prompt and stores each layer's actual K/V words. Cached
decode computes only the new character in every block, padded to four rows for
the current kernels. Section 11 explains context rebuilds and position handling.
The [model upgrade notes](MODEL_UPGRADE.md) trace the new dimensions and tradeoffs.

Draw the responsibility boundary explicitly:

- **Host:** read the checkpoint, encode characters, lay out/copy raw words,
  assemble and launch programs, check results, display characters.
- **SM RTL:** embedding lookup, matrix products, normalization, activation,
  masking, softmax and character selection. For sampling, the host supplies a
  seeded uniform scalar; the SM performs the probability sum and selection.

Independent reference calculations check the result. Their answers do not
replace the SM's intermediate activations. The current testbench loads internal
memories and resets between launches; a hardware memory/launch interface still
has to replace that mechanism for deployment.

## 4. What caused repetition: the original small-model investigation

Original generation selected the highest-probability character at every step,
with no sampling. This was deterministic and made simulator comparisons
reproducible, but the baseline checkpoint often entered a repeating pattern.

In the recorded baseline `the sky today is` run, the four generated characters were a
space, `t`, `h`, then `e`. After the space, the leading probabilities were:

| Next character | Probability |
| --- | ---: |
| `t` | 11.16% |
| `a` | 8.56% |
| `w` | 7.69% |
| `m` | 7.36% |
| `h` | 7.29% |

An 11.16% candidate still wins if every other character has a lower probability.
After selecting `t`, `h` wins at 62.65%; after that, `e` wins at 45.99%.
These probabilities concern individual characters, not the whole word.

Additional checks compared five prompts' next-character RTL predictions to the
independent reference; all agreed. Longer reference-only continuations also
repeat “the”. This establishes a model/decoding limitation for these examples.
The small model learned local character patterns. Neither that model nor the
larger replacement demonstrates reliable instruction following.

We changed default generation to categorical sampling. The host provides a
seeded uniform draw in [0, 1); the SM sums the probabilities, scales the draw by
the total, and selects the first cumulative sum above that threshold. The kernel
handles zero-probability padding and FP32 rounding at the upper boundary. We
retain greedy decoding for deterministic evaluation and expose `--seed` for
repeatable sampled runs. There are no hard-coded words or repetition penalties.

Separately, further CPU training at the same model size improved validation loss
from 2.1824 to 2.1522 and held-out CPU loss from 2.1772 to 2.1435. A fresh AdamW
optimizer and cosine learning-rate decay were used; the best continuation was
selected on validation loss. Training metadata and the original checkpoint are
preserved. Greedy generation still repeats with the new weights: the repetition
improvement in that experiment came from sampling, while the loss improvement came from training.

On six fixed prompts, all six 32-character greedy continuations contained a
threefold repeated substring. None of the 18 sampled reference continuations
(three fixed seeds per prompt) did. Three 32-character sampled RTL runs match
their reference continuations exactly. Output still contains misspellings and
weak sentence structure; the repetition diagnostic does not measure fluency.

Longer generation also exposed a numerical failure at the whole-model logits
check. Eight repeated squarings in the exponential approximation amplified FP32
rounding. Using exp(x/32) and five squarings reduced the worst observed absolute
error on a 20,001-point grid from about 9.97e-6 to 1.32e-6. We added direct RTL
accuracy cases and reran the failed continuation, preserving all tolerances.

## 5. What changed to make inference faster

The original terminal run extended `My name is ` to `My name is the ` in
228.3 seconds. Profiling identified simulator execution and per-cycle CSV
processing as the main time costs.

The model runner now prefers Verilator, which compiles the same SystemVerilog
RTL and testbench into a native executable. The build is cached by source/tool
identity. Per-cycle CSV tracing is optional; output memory, registers, cycle
counts and all numerical checks remain enabled. Icarus remains available with
`--simulator icarus`, and `--trace` enables detailed profiling on either backend.

Four complete checked forward passes took 6.85 seconds with Verilator and
tracing off; an interactive terminal run displayed 7.0 seconds. Both exclude
the first simulator build. This is approximately 33× faster simulation on the
development Mac. Because both the simulator and tracing setting changed, the
combined measurement must not be attributed to either change alone.

Across all 612 launches, program words, input/output memory, registers,
predictions, probabilities and cycle counts matched the original run exactly.
Both runs used 3,909,821 execution cycles. The speed change did not modify RTL,
kernels, checkpoint weights or numerical tolerances. Reducing hardware cycle
counts was separate from that simulator change. The later exponential change
described above does alter kernel instruction work and cycle counts.

## 6. What the tests prove

The speed change passed 37 RTL testbenches and 57 Python test methods, including
cross-simulator checks for FP32 edges, lane masks, tensor rounding, full model
execution, and tensor traces.
The subsequent sampling/numerical update has its own current regression counts
and source fingerprint in `models/character_transformer/verification.json`.

On the previous small checkpoint's held-out evaluation, all 128 greedy predictions matched PyTorch.
Only 57 of those predictions matched the actual next characters in the text
(44.53%). Those measure different things: agreement with the reference checks
implementation correctness; agreement with the text measures prediction quality.
That evaluation uses eight windows, so it is a small model-quality sample.
The new checkpoint has a separate verification record with 32-character windows.

The [inference guide](MODEL_INFERENCE.md) records numerical tolerances, model
provenance and limitations. Local evidence from these investigations is under
`build/performance/`, `build/model-quality/`, `build/model-upgrade/`, and
`build/diagnostics/repetitive_predictions/`; build
artifacts are ignored by Git and should be copied with the report if needed.

## 7. Why we increased model capacity

Sampling alone did not repair spelling or sentence structure. We trained three
larger candidates and selected the two-block, width-64 model using validation
loss. On identical held-out next-character targets, accuracy rose from 38.48%
to 51.12%, with loss falling from 2.1530 to 1.7257. Those are CPU reference
measurements; full RTL checks independently establish numerical agreement.

Explain four separate changes: more features, four attention heads, a second
block, and a doubled context window. Then explain the cost: substantially more
matrix arithmetic, data movement and simulation time. The first checked
prediction took about 15 seconds. Generated text is improved, but still contains
errors. The full comparison and reproduction commands are in
[MODEL_UPGRADE.md](MODEL_UPGRADE.md).

Whiteboard a split K reduction and an attention chunk whose first query is row
28. Show why it must mask columns greater than its actual query index, not its
index within the chunk. Finally compare full versus final-row generation and
explain why earlier layers cannot discard the earlier tokens.

Also distinguish the saved checkpoint from the model loaded by a running process.
After the upgrade, an old browser server still produced `To be, or not to the `.
Its `result.json` identified the original checkpoint (`846b525279aa…`), not the
upgraded model (`d63bcfcb2933…`). Restarting loaded the new code and weights.
The terminal and browser now display checkpoint identity and reload checkpoints
between predictions. Tests cover changed and invalid checkpoint files. This
fix makes the active model observable; it does not guarantee fluent output or
prevent the ordinary word “the” from being sampled.

A later `Romeosed prove,` result did use the upgraded two-block checkpoint.
Reproducing the exact prompt and seed on the CPU separated that language-quality
problem from RTL arithmetic. We compared continued training with two four-block
candidates, selected context 32 using validation performance and simulation cost,
and added temperature-controlled sampling on the SM. The raw distribution remains
verified separately. A clearly labeled CPU preview makes text exploration fast.
Use [the quality notes](LANGUAGE_QUALITY.md) to explain the controlled comparisons,
retained bad outputs, test results and remaining grammar limitations.

## 8. Making inference faster without changing the model

We profiled the existing four-block checkpoint before changing it. Matrix
multiplication consumed about 84% of GPU cycles. Separate simulator processes,
repeated assembly and the Python integer checker also added wall time.

We retained C tiles in registers, reused the simulator process across reset
kernel launches, cached assembly and added a compiled integer checker. A matched
unprofiled run fell from 36.24 to 20.65 seconds, with identical probabilities
and sampling. GPU cycles fell by 5.8%; the larger wall-time saving also includes
host improvements. All mathematical checks remain enabled. Explain that
distinction in the report rather than calling the whole speedup a GPU gain.

See [the speed investigation](INFERENCE_SPEED.md) for the register allocation,
process protocol, oracle tests and why a naive KV cache would be incorrect for
the current sliding position window.

## 9. Reducing actual RTL cycles

The next profile still put about 82% of cycles in matrix multiplication. We
changed three hardware paths while holding the model, assembly and numerical
checks fixed:

1. Replace bit-at-a-time FP alignment/normalization with sticky barrel shifts and
   leading-zero detection. The multiplier also accepts both inputs together,
   removing its old handshake overhead. Scalar operations finish within four
   clocks after acceptance, including subnormal results.
2. Let a tensor PE forward its tokens once the product enters the adder, so its
   next multiplication can overlap that addition. Forward a just-completed sum
   when necessary, retain k order, and drain the final additions before exposing C.
3. Connect gathered SM tiles directly to the array and return four complete rows,
   removing the serial 32-word input and 16-word output transfers inside the SM.

The paired first-character run fell from 22.44 to 11.21 seconds. Simulated cycles
fell from 32,235,401 to 11,847,367 (63.25%), while all 1,571 launches produced
identical memory/register outputs and model probabilities. These are simulation
measurements. Wider combinational shifters can affect area and clock frequency;
we still need synthesis/timing results before claiming an FPGA throughput gain.
The test and measurement record is in [FP32_SPEED.md](FP32_SPEED.md).

For a whiteboard session, draw the old serial multiply/add timeline beside the
new overlapping one. Then trace the last product: token forwarding happens
before its sum is ready, which is why the array now needs a drain state.

## 10. Using the existing warps and reducing launch overhead

The previous matmul kernel used only warp 0. Its trace had no eligible warp in
about 70% of cycles, usually while an operation was still in flight. Independent
output tiles now run on up to four warps. Their K sums stay local and retain the
same FP32 ordering. This lets other warps load operands and accumulate finished
tiles while the single tensor unit works; it does not add more tensor units.

We also broadcast shared-memory loads of the same word, which removes repeated
constant-read serialization in GELU. Requests containing a store keep lane order.
Bias addition runs at the end of the final K chunk, removing separate bias
launches without changing where floating-point rounding happens.

The same first-character workload falls from 11,847,367 to 8,101,875 cycles and
from 1,571 to 1,406 launches. Block outputs and final probabilities match exactly.
The runner records its RTL fingerprint and reloads changed hardware between
prompts, preventing an old interactive process from silently using stale RTL.
It now prints first-character and subsequent-character latency separately.
At that stage, every character still recomputed the full window. See
[RUNTIME_SPEED.md](RUNTIME_SPEED.md) for those measurements; the next change
addresses repeated computation.

On a whiteboard, draw four warp timelines feeding one tensor unit, then mark
where independent memory and FP instructions overlap its work. Compare issue
stalls separately from the tensor unit's busy time; a blocked warp does not
mean the whole SM is idle.

## 11. Cache previous keys and values during generation

The old runner had no KV cache. It recalculated the entire sliding window for
every new character. Calling this a broken cache would misdescribe the starting
point: we added caching and a context policy that makes it valid.

The checkpoint uses learned absolute positions. Simply dropping the oldest K/V
and shifting the rest is incorrect: those activations were computed at other
positions and with different preceding context. We kept the checkpoint and
changed generation to append characters at fixed positions until position 31.
At overflow, all layers recompute the newest 24 characters at positions 0..23.
The next eight characters can reuse that state before another rebuild.

Prefill reads the prompt without artificial leading spaces. Each layer saves
only real K/V rows produced by RTL. Decode projects the new character's Q/K/V,
attends to cached keys and values, and runs the new row through the feed-forward
network. Host code stores/copies raw words; every arithmetic operation remains
on Snackwell. The cache reaches 64 KiB, stored on the host, not in 4 KB shared
memory. Padding fills the existing four-row kernels; it is never committed as
context or selected as a prediction.

Draw a row per layer and a column per character. Shade the cached columns, then
trace one new column through all layers. At overflow, cross out the cache and
redraw the retained window with new position labels. This makes the reason for
the rebuild explicit.

Correctness and language quality are separate checks. Compare cached results
with uncached RTL on identical contexts, including rebuilds. Compare the new
24..32-character policy with the original 32-character policy on identical
held-out targets. Outputs can differ between policies even when both are
correct. The checkpoint is unchanged; this is inference work, not training.
See [KV_CACHE.md](KV_CACHE.md) for timings, tests and reproduction commands.

## 12. Make cached decoding process one character

The first cache reused K/V correctly, but assembly still processed four rows
for each new character. We added a one-row path, enlarged its output-column
chunks, combined Q/K/V projections, and stopped multiplying unused vocabulary
columns. Draw a single useful row entering the 4×4 array: the other three rows
receive zeros, and only the useful result is accumulated and stored. The array
itself is unchanged; this is kernel and runtime work.

The independent verifier was another source of repeated computation. It now
keeps its own double-precision history, computed solely from weights and IDs.
Draw this separately from the raw RTL cache. An intentionally corrupted RTL
cache fails the independent check. Both histories advance only after a verified
step, so cancellation cannot leave one ahead of the other.

Keep runtime measurements separate from model experiments. Compare old and new
runtime with identical weights, prompt and random draws. Compare candidate
weights on identical validation targets, then evaluate the selected candidate
on separate test text. Longer context and more training are experiments, not
automatic improvements. See [DECODE_OPTIMIZATION.md](DECODE_OPTIMIZATION.md).

## 13. Improve the checkpoint separately from the hardware

The faster runtime reproduced the same weak text because its weights were
unchanged. Compare CPU-reference continuations first: if both implementations
produce that text, investigate model quality rather than labeling the cache
broken. Then select candidates on identical validation targets and report a
separate test comparison. Save unsuccessful experiments as well.

An experimental model increased width from 64 to 128 and context from 32 to 64,
while retaining four transformer blocks. It has 555,073 parameters. A shallower
wide candidate did not improve validation results. Distilling the selected
four-block model into the original-sized model also failed to improve it.
Both alternatives were rejected. The wider model also proved unsuitable as a
default: it lowered held-out loss but still wrote poor sentences and took more
than twice as long. We restored the original width-64 checkpoint.
See [MODEL_CAPACITY.md](MODEL_CAPACITY.md) for measurements and raw diagnostics.

On the whiteboard, keep the physical four-lane SM and 4×4 array unchanged, and
draw a larger logical matrix split across more launches. Count the model's
features separately from hardware lanes. Explain how looped LayerNorm fits the
instruction ROM, and why splitting elementwise GELU preserves feature order.
The wider model's K/V cache reaches 256 KiB on the host, so the diagram must
show the existing transfers through 4 KiB shared memory.

This is CPU reference training followed by verified Snackwell inference.
Backpropagation on Snackwell is still a separate milestone. Numerical agreement
proves implementation correctness for the checked inputs; it does not prove that
a generated sentence is meaningful.

## 14. Match the kernel to single-character decoding

A cached step has one input row. A full TMUL still occupies the 4×4 array, even
though three result rows are unused. The new SIMD matrix-vector kernel assigns
one output column to each lane. It computes four independent products, adds them
in their original order, then accumulates that four-term sum. The grouping is
necessary to preserve FP32 rounding and exactly reproduce the tensor path.
Prefill's multirow products still use the tensor array.

For LayerNorm, draw all lanes reading the same row and calculating the same
sequential mean and variance. Then split the affine outputs across lanes:
lane 0 handles columns 0, 4, 8, ...; lane 1 handles 1, 5, 9, ... . There is no
new parallel reduction or reassociation. Residual and bias addition distribute
independent columns the same way. See [INFERENCE_REPAIR.md](INFERENCE_REPAIR.md)
for tests and separate measurements of the rollback and kernel optimization.

The terminal now says `RTL/reference match`. That describes what verification
establishes. Neither the checkpoint rollback nor faster execution fixes its
remaining language-quality problem.

## 15. What remains

Explain the current host-mediated data movement before proposing global memory
and a persistent launch interface. Use cycle profiles to guide kernel and
architecture optimization, then recheck both numerical results and cycle counts.
For language-model improvements, track output quality independently of simulator
speed. Backpropagation and weight updates on Snackwell remain to be implemented;
the planned FPGA synthesis, resource/timing checks and board demonstration follow
training support.

For the report walkthrough, start with one whiteboard session per section.
Predict what should happen before opening the code or checking the measured
result. That makes each design choice something you can explain and defend.
