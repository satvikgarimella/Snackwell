# One-character decode optimization

The timings on this page use the earlier width-64, context-32 checkpoint.
These runtime improvements also support the experimental wider model. Its
quality and latency comparison is in [MODEL_CAPACITY.md](MODEL_CAPACITY.md).
The width-64 model is now restored; subsequent SIMD decode changes are in
[INFERENCE_REPAIR.md](INFERENCE_REPAIR.md).

This follows the first [KV cache implementation](KV_CACHE.md). The cache was
reusing prior keys and values correctly, but the decode kernels still processed
four rows for each new character. The independent scalar checker also repeated
the entire prefix. Both costs remained after adding the first cache.

## Measured result

The same default checkpoint, prompt `My lord, I have something to say`, seed
92617660 and temperature 0.7 were run twice with each runtime, in
before/after/after/before order. No training, test suite or quality job ran
concurrently. These times include verification and exclude compilation and
cycle tracing. Both versions produced ` 'twas. There's ` after the prompt.

| Work | Before | After |
| --- | ---: | ---: |
| Mean cached character | 1.93 s | 1.20 s |
| First character including prompt | 7.81 s | 7.71 s |
| Mean context rebuild | 5.92 s | 5.91 s |
| Mean subsequent character, including rebuilds | 2.47 s | 1.83 s |
| All 16 characters | 44.79 s | 35.10 s |
| Simulated cycles | 38,060,603 | 32,970,230 |
| Kernel launches | 8,692 | 7,242 |

Cached latency fell 38%, total latency fell 22%, and simulated cycles fell 13%.
The first-character and rebuild timings are essentially unchanged; they remain
the main pauses. This is a 1.28× overall speedup, not an interactive-speed or
FPGA-performance claim. Every generated last-row probability, sampling
distribution and selected character matched exactly across runtime versions.

A representative 25-character cached step uses 989,797 cycles and 288 launches,
down from 1,357,674 cycles and 394 launches. Matrix operations account for about
71% of its remaining cycles. Its independent reference calculation takes about
11 ms; the rest includes simulation, launch preparation, per-kernel checks and
artifact writes. This cycle breakdown is not a wall-time stall breakdown.

The 98-method regression passed. After the final one-row output optimization,
all 10 cache and multihead/full-versus-last-row checks passed again. All 256 saved
held-out positions retain their previous probabilities, predictions and saved
intermediates. The RTL source is unchanged; the earlier 38-bench hardware run is
historical evidence, not a fresh run. Full measurements, hashes, quality trials
and test records are in
[`decode_optimization_verification.json`](../models/character_transformer/decode_optimization_verification.json).

## Changes to executed kernels

- Decode now transports one row. The existing 4×4 tensor array receives zeros
  in its other three rows, and assembly loads, accumulates and stores only the
  useful row. The hardware array itself has not changed.
- A one-row matrix launch covers up to 28 output columns and 32 input features.
  With bias, its largest layout occupies 984 of the 1,024 shared-memory words.
  Partial products still accumulate on the SM in the original FP32 order;
  bias is added once after the final K chunk.
- Q, K and V weight columns are concatenated for one-row decode. The runtime
  copies the resulting words into three tensors. This reduces launches while
  preserving each dot product's arithmetic and bias.
- One-token embedding launches cover eight features. LayerNorm, softmax and
  other row operations use one active lane for a single row, preserving the
  original scalar reduction order and avoiding duplicate stores. GELU and
  embedding continue to distribute features across four lanes.
- The language-model projection computes 68 columns for 65 characters, then
  copies those words into the 128-column softmax layout. An RTL kernel masks
  unused classes. There is no reason to calculate 60 extra zero-weight columns.
- Sampling processes only the final real row.
- Prefill and rebuild still compute every K/V row needed by future tokens.
  In the last block, only the last real query needs attention output, the
  feed-forward network, final normalization and vocabulary projection. This
  uses its actual position even when the prompt ends in a partial four-row tile.

## Independent verification

Every launch retains its numerical reference and memory-guard checks. There is
also a separate double-precision model calculation from checkpoint weights and
token IDs. Its cache stores only CPU reference calculations, never RTL values.
Its full uncached path remains available and is compared with incremental
results in tests, including every intermediate and K/V tensor.

The runtime checks model identity, RTL identity and prefix before reusing either
cache. It commits both only after all checks succeed. Cancellation or a failed
check cannot advance either history. A corruption test changes the stored RTL
values and verifies that the independent reference rejects the next character.
No numerical tolerances were relaxed. No CPU prediction is used as an executed
Snackwell output.

## Longer contexts

The portable checkpoint and runtime support contexts through 128 characters,
while model width remains at most 64. Embedding launches split long contexts into
32-row chunks. The attention-mask kernel uses a counted loop beyond 64 columns
to stay within the 1,024-instruction ROM. Tests cover column 127 and cache
rebuilds from 128 to 96 characters.

Supporting a larger context does not mean its trained weights are better. Model
candidates are compared on identical validation targets before considering a
replacement. CPU reference training does not implement training on Snackwell.

## Model experiments and the deployment decision

Four new CPU-trained candidates were evaluated, alongside the current default
and the earlier context-64 model. All use four blocks, width 64 and four heads.
The 8,192 validation target positions are identical across all six models;
each gets its own context length. This comparison is separate from the training
log's average over all positions in sampled windows.

| Candidate | Validation loss | Next-character accuracy |
| --- | ---: | ---: |
| Current context-32 default | 1.46368 | 56.02% |
| Continue context-32 training, 4,000 updates | 1.46213 | 56.68% |
| Continue earlier context-64 model, 6,000 updates | 1.46962 | 56.34% |
| Extend current model to context 64, 8,000 updates | 1.48512 | 55.42% |
| Extend current model to context 128, 6,000 updates | 1.53291 | 53.85% |
| Earlier context-64 model | 1.47606 | 56.04% |

The lowest-validation-loss candidate was then compared with the current model
on 8,192 identical, separate test targets. Loss changed from **1.6218 to 1.6236**
(worse), while accuracy changed from **52.94% to 53.38%**. Fixed-prompt diagnostics
retain all outputs for eight prompts and three sampling seeds, plus additional
seeds for the selected candidate. Spelling coverage improved in that diagnostic,
but grammar and repetition remained weak. Corpus word coverage is not fluency.

These mixed results do not establish a useful language-quality improvement.
**The default checkpoint remains `43f211daad7c`.** The trained candidates and
their evaluation data remain under `build/decode-quality/`; none silently
replaces the model. The best candidate also passed 256 exported RTL evaluation
positions, which establishes implementation agreement, not better language.
Its exported windows differ from the default's windows, so their evaluation
losses must not be compared directly.

The continued candidates use seed 1733, fresh AdamW and cosine learning rates
from 0.0001 to 0.00001 (batch 64 for context 32, batch 32 for context 64).
The extensions use seed 1729 and repeat the old learned position table only for
initialization, then train all parameters: context 64 uses batch 32 and rates
0.0002 to 0.00002; context 128 uses batch 16 and rates 0.0003 to 0.00003.
All use the same training/validation/test split and corpus hash as the baseline.
This is CPU reference training; Snackwell backpropagation remains future work.

## Reproduce the checks

```sh
python3 -m unittest tests.test_decode_kernels tests.test_kv_cache -v
python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
python3 scripts/benchmark_kv_cache.py --out build/kv-cache/current
```

The benchmark compares cached execution with full RTL recomputation of the same
prefixes and checks exact probabilities and selections. `--checkpoint` selects a
particular model. Use identical prompt, seed, temperature and weights when
comparing runtime revisions; changing the model changes the workload too.

## Remaining limits

All timings are verified RTL simulation on the development Mac. The tensor
array still physically computes a 4×4 tile, and the design has one tensor unit.
The host still transports weights and activations through bounded shared-memory
launches. Learned absolute positions still require context rebuilds on overflow.
These changes do not implement global memory, training on Snackwell, or FPGA
deployment, and do not establish FPGA throughput.
