# Trained character-transformer inference

`scripts/run_model.py` executes a saved character model through Snackwell's SM
in RTL simulation. Token lookup, model arithmetic, masking, softmax and character
selection run on the SM. The host loads/copies raw words, launches kernels,
checks results and displays characters. It does not supply model outputs.

## Run the saved checkpoint

Python 3.10+ and an RTL simulator are sufficient. Inference prefers Verilator
when installed and falls back to Icarus Verilog. On macOS, Verilator needs a C++
compiler and `make`; install it with `brew install verilator` and the Xcode
command-line tools. PyTorch is optional and used only for host reference training
and diagnostics.

```sh
python3 scripts/run_model.py --interactive --tokens 4
python3 scripts/run_model.py --stream --prompt 'ROMEO:' --tokens 4 --seed 1731
python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
```

The first Verilator run builds a native simulator. Later runs reuse the cache
under `build/simulators/verilator/`, keyed by sources, tool versions and options.
Interactive mode accepts `/tokens N`, `/temperature T`, `/seed N`, `/quit` and
Ctrl+C to cancel a prediction.
Each character appears only after its arithmetic and selection checks pass.
Startup prints the checkpoint path, architecture and SHA-256 prefix. The current
default should show `4 blocks`, `width 64`, `context 32` and `checkpoint 43f211daad7c`. Each prediction
also identifies the loaded model and decoding mode. Interactive mode reloads and
validates the checkpoint before each prompt's prediction, then holds it fixed
until that prediction finishes. Invalid replacement files fail explicitly.
The runner also prints the RTL source fingerprint and reloads changed RTL
between prompts. It reports first-character latency, average subsequent latency,
cached decode and context rebuild time separately. Timings include verification
and exclude compilation.
Restart an older Python runner once to load these runner/kernel changes.
Cached decode uses dedicated one-row paths. Prefill and periodic rebuilds take
longer. Host load and simulator affect timing; see
[the current measurements](INFERENCE_REPAIR.md).

Generation defaults to categorical sampling. The host supplies an exactly
representable uniform FP32 draw from a seeded PRNG. An SM kernel sums the
probabilities, scales the draw by that sum and selects the first cumulative
probability above the threshold. Zero-probability padding cannot be selected.
`--seed 1731` repeats a run; without a seed, each interactive prompt receives
a new seed printed and saved with its results. `--decode greedy` selects the
maximum-probability character instead. Greedy output can still loop.

Sampling defaults to temperature 0.7. A second softmax on the SM scales the
logit differences by `1 / temperature`, then the SM samples from that distribution.
Raw model probabilities and greedy predictions remain verified and saved;
held-out evaluation always measures the unmodified distribution. `--temperature 1`
reproduces the original sampling rule. The accepted range is 0.1..2.0.

For fast text-quality checks, `build/model-venv/bin/python scripts/preview_model.py
--interactive` runs the optional CPU reference. It explicitly does not test RTL.
Use matching checkpoint, prompt, seed and temperature for subsequent hardware
checks. See [the language-quality investigation](LANGUAGE_QUALITY.md).

## Model and quality

The default checkpoint has **four pre-LayerNorm transformer blocks, four attention
heads per block, 32-character context, width 64, feed-forward width 128, 65
characters and 144,449 FP32 parameters**. It was trained on
[TinyShakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
with CPU PyTorch. Backpropagation and weight updates on Snackwell remain future
work, followed by FPGA deployment.

A wider candidate reduced loss on 8,192 matched test targets from 1.6063 to
1.5401 nats/character, but still wrote poor sentences and more than doubled
simulation time. It is archived in `models/character_transformer/wide-context64/`;
the faster original checkpoint is restored. See the [capacity experiment](MODEL_CAPACITY.md)
and [rollback](INFERENCE_REPAIR.md). No new weights or sampling changes were
installed during the rollback.

Output still contains spelling errors and weak sentence structure. The model completes Shakespeare-like text;
it does not know personal facts or follow instructions. Its 32-character window
cannot retain a conversation. Characters outside the saved vocabulary are
rejected. Short prompts now use their actual length, starting at position zero.
Generation appends one character at a time using cached keys/values from all
layers. At overflow, it rebuilds the newest 24 characters at positions 0..23,
then reuses that state for eight more steps. Retaining old K/V after shifting
learned positions would be incorrect.

This policy differs from a sliding window, so generated text can differ.
`--no-kv-cache` uses space-padded sliding windows at the selected checkpoint's
context length; the CPU preview accepts the same flag. The older context-32
policy's quality measurements are retained in [KV_CACHE.md](KV_CACHE.md).

## Memory and execution

Parameters occupy 577,796 bytes, exceeding the SM's **4 KB shared memory**.
The host keeps tensors as raw 32-bit words and transports bounded pieces:

- Matrix launches cover up to eight output rows, sixteen output columns and
  K=32. Products with larger K carry the actual partial C words into subsequent
  launches. `TMUL` and `FADD` on the SM preserve the four-term tile reduction
  order; Python never accumulates partial products. Within a launch, partial
  tiles stay in registers until the K chunk finishes. Single-row decode covers
  up to 28 output columns per launch, and combines Q/K/V projection columns.
- Each attention head uses sixteen features. Splitting and concatenation copy
  words; dot products, scaling, softmax and weighted sums execute on the SM.
- Attention scores are processed in query-row chunks. Causal masking retains
  the original query index, including for a chunk starting at row 28.
- Embedding tables are split into four feature columns for prefill and eight
  for one-token decode when capacity permits. Token indexing and positional
  addition happen on the SM.
- Row kernels use up to four rows per launch and in-place buffers where safe.
  Single-row additions distribute columns across the four lanes. LayerNorm
  retains sequential mean/variance reductions in each lane, then distributes its
  independent affine outputs. Softmax and sampling retain lane-zero reductions.
  The largest one-row matrix launch uses 984/1,024 memory words. The largest
  measured program for the wider model is attention masking at 609/1,024
  instructions. Width-128 LayerNorm uses counted loops. The 256-feature GELU
  buffer is processed in bounded contiguous chunks.
- The 65-class projection computes 68 columns, then copies into the 128-column
  softmax layout. A kernel
  masks unused classes to negative infinity. Their probabilities must be zero.

Full evaluation computes every position. Prefill/rebuild computes all context
rows in earlier layers and only the final block's last real query row, while
retaining every real K/V row for future tokens. Cached
decode computes one real row per layer. One-row products now use SIMD FMUL/FADD,
with one output column per lane and the same four-term reduction order as TMUL.
Multirow matrix products still execute on the systolic array.
Each layer's K/V cache contains only actual RTL output words
for real characters. It reaches 64 KiB at 32 characters, held on the host and
transported in chunks; it is not an on-chip cache. Attention columns are padded
to a power of two, with future/padded columns masked on the SM.
The saved reports identify `cache`, `last_only`, `output_rows` and
`output_row_offset`. Verification compares those positions with the independent
full-model oracle. Cached and uncached runs must agree bit-for-bit on their
shared output rows under the same context policy. A cancelled or failed step
does not commit partial cache state. Each new prompt starts with an empty cache;
changed checkpoints or RTL builds cannot reuse stale state.

The independent double-precision reference keeps its own causal history to
avoid recalculating old rows. It never receives RTL intermediates. Tests compare
that history and all outputs with full independent recomputation; deliberately
corrupted RTL cache values must fail verification.

Matrix kernels distribute independent output tiles across up to four warps,
with all four lanes active in each. Each warp retains the original K reduction
order, and bias is added once in the final K chunk before storing the result.
Row and embedding kernels currently use warp 0. The testbench resets and loads
internal memories between launches. This host-mediated execution demonstrates
functional inference, but does not implement global memory, DMA, a persistent
launch interface or FPGA throughput. Those remain separate hardware work.

## Training provenance

The text is split contiguously into 90% training, 5% validation and 5% test.
Vocabulary comes from the training split. Dataset SHA-256:

`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`

The restored checkpoint used CPU PyTorch 2.14.1, seed 1729, batch size 64 and
20,000 AdamW updates, with cosine learning-rate decay from 0.002 to 0.0002.
Validation selected step 19,500. The checkpoint stores its original training
history and data hash. This rollback does not retrain it; the trainer has since
acquired additional options documented in the historical experiments.

```sh
python3 -m venv build/model-venv
build/model-venv/bin/python -m pip install -r requirements-model.txt
build/model-venv/bin/python scripts/train_reference_model.py --download \
  --width 64 --context 32 --heads 4 --layers 4 --steps 20000 --batch-size 64 \
  --learning-rate .002 --min-learning-rate .0002 --seed 1729 \
  --out build/new-candidate
```

The trainer also accepts `--device mps` for an available Apple GPU. CPU remains
the default. Exports return to CPU FP32 and are checked against the independent
scalar model. The checkpoint records the training device. Both modes train the
reference model on the host; neither implements backpropagation on Snackwell.

Inference never starts training or downloads data. Earlier checkpoints remain
under `models/character_transformer/baseline/`, `previous-small/`,
`previous-two-block/` and `previous-context32/`, with their
original evaluation and verification records. V1 checkpoints remain supported;
V2 explicitly records block/head counts and allows the larger dimensions.

## Correctness checks

Each launch computes an independent expected output before execution, then
checks the actual result and all non-output memory guards. Matrix products,
embedding, additions, masking and selection use exact FP32 comparisons. Approximate
kernels retain their tighter operation-specific tolerances. Expected answers
are never copied into the executed model's activations.

The separate scalar whole-model reference reconstructs the network from the
checkpoint and input IDs in Python double precision. It checks embeddings,
each block, final normalization, logits and probabilities:

```text
abs(actual-reference) <= 5e-5 + 5e-4 * abs(reference)
```

Greedy predictions must match the independent reference. Sampling additionally
checks the exact FP32 cumulative-sum rule using the supplied draw. These are
unchanged test criteria, not universal error bounds for all possible models.

The restored checkpoint has eight exported evaluation windows of 32 targets
each. Their saved PyTorch outputs are compared with the actual RTL results and
the independent whole-model reference. The current verification record contains
the fresh evaluation result.
These smaller RTL evaluation windows differ from the 8,192-target CPU comparison.
Reference agreement checks implementation correctness; target accuracy measures
prediction quality. Do not confuse the two.

The [verification record](../models/character_transformer/verification.json)
contains checkpoint/source fingerprints, the held-out results, regression
counts and sampled-generation evidence. The regression covers real RTL launches,
exact K-chunk accumulation, transposed products, masks, multihead causality,
wide normalization, cached/full equivalence and cross-simulator checks. No RTL source changed; earlier standalone
RTL regression counts are historical, not a new run claimed by this update.

`quality/repair/` contains the current evidence. Older root-level quality files
belong to the wider-model experiment, identified by their checkpoint hashes. Continuations
are checked character by character; probability agreement is not a fluency claim.

## Simulation speed

The latest host implementation and its checks are described in
[INFERENCE_HOST.md](INFERENCE_HOST.md). The older measurements below retain
their original checkpoint and simulator context.

Verilator compiles the SM RTL to a native executable. Untraced model sessions
use `sim/session.cpp` to clock it directly; traced sessions and one-shot kernel
runs use the original SystemVerilog testbench. Both execute the same SM and
check that selected warps halt with no pending work. Per-cycle CSV tracing is off by default.
Programs, input/output memory, registers, cycles and all numerical checks remain
saved. Use `--trace` for detailed cycle profiles or `--simulator icarus --trace`
for the original diagnostic path.

The current model and isolated latency comparison are documented in
[INFERENCE_REPAIR.md](INFERENCE_REPAIR.md). The wider experiment remains available
with `--checkpoint models/character_transformer/wide-context64/checkpoint.json`.
All speed measurements below are historical and use earlier model profiles.

The earlier software optimization keeps matrix partial sums in registers, reuses one
simulator process per forward pass and accelerates the independent integer
matrix checker. A matched run dropped from 36.24 to 20.65 seconds, with exactly
the same probabilities and selection. Execution cycles fell from 34,236,950
to 32,235,401. The launch count remains 1,571. Every launch still resets and
loads the DUT, and all numerical checks remain enabled. See the
[speed investigation](INFERENCE_SPEED.md) for implementation and verification.
The [speed verification record](../models/character_transformer/performance_verification.json)
records 37 passing RTL benches, 79 passing Python methods, all 256 evaluation
positions and four sampled terminal characters matching the preceding RTL runs.

The subsequent RTL optimization bounds FP arithmetic latency, overlaps successive
multiply/add operations in the tensor PEs, and passes complete tiles between the
SM adapter and array. A new paired run fell from 22.44 to 11.21 seconds and from
32,235,401 to 11,847,367 cycles, with the same checkpoint and 1,571 launches.
See [the FP32 RTL measurements](FP32_SPEED.md) for numerical comparisons,
tests and the remaining synthesis/timing limitations.

The next kernel/runtime pass distributes output tiles across the existing warps,
broadcasts shared-memory reads of the same word, and adds bias inside the final
matrix kernel. The same prediction takes 8,101,875 cycles and 1,406 launches.
The terminal reports the RTL source fingerprint, first-character latency and
the average for subsequent characters. Long-lived terminal/browser runners
check for changed RTL before a new prediction and rebuild when needed.
See [the runtime measurements](RUNTIME_SPEED.md) for paired timings and tests.

KV caching is the next optimization. It avoids repeating old-token projections,
attention queries and feed-forward work between rebuilds. Terminal, one-shot CLI
and browser generation use the same policy. Reports separate `prefill`, `decode`
and `rebuild`; average subsequent latency includes rebuild cost. See
[KV_CACHE.md](KV_CACHE.md) for the paired benchmark and quality comparison.

Before those changes, the first checked width-64 model prediction for `My lord, I have something to say`
took 34,236,950 execution cycles across 1,571 launches and 37.5 seconds. Timing
excludes the first simulator build and is not an estimate of FPGA performance.
The earlier two-block model took about 15 seconds per character; the extra blocks
increase the work. A four-block context-64 candidate took 89.6 seconds and did
not improve the paired validation comparison, so context 32 was retained.

The earlier final-row optimization reduced a checked full pass from 43,497,346
to 34,185,392 cycles (21.4%), with bit-identical probabilities for the shared rows.
The comparison is saved alongside the checkpoint evidence. CPU preview is the
fast way to explore text quality; it never substitutes for RTL verification.

Historically, the original 4,657-parameter checkpoint took 228.3 seconds to
produce four characters with Icarus and tracing. Verilator without tracing took
6.85 seconds for the same program/memory/register/cycle results across all 612
launches. This demonstrated approximately 33× faster simulation for that old
checkpoint. It is **not** the performance of the current larger model.

## Artifacts and tools

Only the top-level `result.json` confirms that the entire requested run passed.
It is removed before a rerun, so a failed or interrupted run cannot leave stale
success. Runs save per-launch assembly, memory images, cycle counts and checks
under `build/inference/character_transformer/`. Interactive runs have unique
subdirectories under `terminal/`.

An optional `--live` browser view runs locally at `127.0.0.1:8768`. It displays
real kernel progress and the final block's first attention head. The page shows
the loaded architecture, decoding mode and checkpoint ID, updating them when a
run starts. Like interactive mode, it reloads and validates the checkpoint before
each prediction and keeps that model fixed during the run.

**Python code changes still require a process restart.** Quit an old interactive
session with `/quit` and rerun the command; stop an old browser server with Ctrl+C,
restart it and reload the page. Editing files does not update a running Python
process. Terminal use does not require the browser.

- `character_model.py`: validated schema, encoding and scalar mathematical reference.
- `torch_model.py`: optional host training/reference implementation; never imported by RTL inference.
- `model_runner.py`: model execution, output checks and held-out evaluation.
- `kv_cache.py`: per-prompt cache lifecycle, context rebuild policy and timing summary.
- `model_runtime.py`: bounded launches and raw-word transfers.
- `kernels/model.py`: embedding, looped softmax, masking and selection programs.
- `cli/train.py`: reference training and export.
- `cli/model_quality.py`: identical-target CPU quality comparison.
- [Whiteboard guide](INFERENCE_WALKTHROUGH.md), [report walkthrough](REPORT_WALKTHROUGH.md),
  [upgrade decisions](MODEL_UPGRADE.md).
