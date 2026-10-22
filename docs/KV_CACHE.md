# Cached character decoding

This records the first cache implementation and its measurements. The subsequent
[one-row decode optimization](DECODE_OPTIMIZATION.md) preserves its checkpoint
and context policy while reducing kernel and verification work.

Snackwell generation now caches each layer's keys and values. The checkpoint
`43f211daad7c` is unchanged. All model arithmetic, including new K/V, attention,
normalization, feed-forward work and selection, still executes through the SM
RTL. The host stores and transports the resulting raw FP32 words.

## Try it

Restart an already-running Python session to load the new runner:

```sh
python3 scripts/run_model.py --interactive --tokens 16 --seed 92617660
```

Enter `My lord, I have something to say`. The terminal reports first-character
latency, cached decode time and context rebuild time separately. The average for
subsequent characters includes rebuilds. Use `--no-kv-cache` for the original
padded, sliding-window policy. The default one-shot CLI, terminal and browser
share the same generation implementation. CPU preview uses the matching context
policy and accepts the same flag, but does not execute RTL.

## Why context rebuilds are necessary

The model learned absolute position embeddings for positions 0..31. Previously
each new character shifted the window and reset those positions. Keeping K/V
after that shift would reuse values computed at different positions and with
different earlier context, especially in deeper layers.

The new policy is explicit:

1. Prefill uses the newest at most 32 prompt characters, without artificial
   leading spaces. Compute and cache real K/V for every layer.
2. Append each generated character at the next position. Compute only its new
   activations, attending to the old cached K/V and its own new K/V.
3. When the next input exceeds 32 characters, retain the newest 24, including
   the new character. Recompute all retained layers at positions 0..23. There
   are then eight cached steps before the following rebuild.

Other supported context sizes retain three quarters of their capacity. Short
prefixes are already part of the causal training objective: each position was
trained using only its prefix. No weight conversion or retraining is needed.

This changes context policy, so outputs need not match the old sliding-window
sequence. Cached and uncached execution must match when given the **same active
prefix**. That is the comparison performed by the paired RTL benchmark.

## Execution and state

Each cached step performs one real row of Q/K/V projections and feed-forward
work per layer. The initial kernels required four rows, so three transport rows were
padded. Only real rows enter the cache or prediction output. Attention pads key
columns to a power of two and uses the SM causal mask to exclude future/padded
columns. The actual last row supplies sampling and attention visualization.

At maximum capacity, K/V occupy `2 × 4 layers × 32 characters × 64 × 4 bytes`,
or **64 KiB**. This storage is on the host; the SM still has 4 KiB shared memory.
It does not add global memory, DMA or persistent on-chip storage.

Cache updates commit only after all launch and whole-model checks pass. A
cancelled or failed append leaves the previous cache intact. New prompts own new
caches. A cache is bound to its token prefix, checkpoint configuration/weight
fingerprint and RTL source fingerprint. Mismatches fail before a kernel launch.

## Verification

The 16-character paired run produced
`My lord, I have something to say 'twas. There's `.
Every generated character's probabilities and selection matched uncached RTL
bit-for-bit under the same context policy.

| Measured work | Cached | Recompute identical context |
| --- | ---: | ---: |
| First character, including prefill and selection | 7.93 s | 7.87 s |
| Mean of the 13 incremental steps | 1.99 s | 7.55 s |
| Mean of the two rebuild steps | 6.15 s | 6.07 s |
| All 16 characters, summed verified forward time | 46.11 s | 118.12 s |
| Total simulated cycles | 38,060,603 | 117,891,652 |
| Total launches | 8,692 | 21,528 |

Cached steps use 394 launches, versus 1,366 or 1,406 for the matched uncached
contexts, and 81.85% fewer simulated cycles on average. Including both rebuilds,
the run is 2.56× faster in wall time and uses 67.72% fewer cycles. Mean latency
after the first character, including rebuilds, is 2.55 seconds. These results
do not claim that every subsequent character takes two seconds.

Every launch still checks independently computed results and memory guards.
In this first version every generated character ran the independent full-model
scalar oracle from scratch, even when RTL computation was cached.
No tolerance has been relaxed and no CPU result is used as an executed output.

The cache tests cover:

- One-character prefill and consecutive prefixes across 4/8-row padding limits.
- Multiple heads/layers, exact K/V words and exact output probabilities against
  independently recomputed RTL prefixes.
- Repeated overflow rebuilds, 64-character contexts and long-prompt truncation.
- Cancellation after partial work, retry, new prompts and stale-state rejection.
- Causal prefix references, CPU-preview policy agreement and legacy execution.

All 89 Python regression methods passed, including HTTP, terminal and
Icarus/Verilator comparisons. After the final simulator-fingerprint/stale-success
guard, all six cache methods passed again. All 256 saved held-out positions and
their intermediate outputs match the previous RTL results exactly. The RTL
source fingerprint is unchanged; the previous 38-bench hardware regression is
preserved rather than claimed as a new run.

```sh
python3 -m unittest tests.test_kv_cache -v
python3 scripts/benchmark_kv_cache.py
build/model-venv/bin/python scripts/evaluate_cache_policy.py
```

The benchmark runs 16 generated characters and an uncached RTL comparison after
each, using identical IDs, checkpoint, draws and temperature. It checks exact
last-row probabilities, sampling probabilities and selected character. Both
paths retain every numerical check. Compilation and cycle tracing are excluded;
no other regression or quality process ran during these paired measurements.
Reported times sum verified forward execution, rather than interactive idle time.

The CPU quality check uses 64 held-out sequences of 32 next-character targets,
the same 2,048 targets for both policies. It uses the final 5% of TinyShakespeare,
checks the corpus hash, and keeps training weights fixed.

| CPU quality metric | Sliding 32 | Periodic 24..32 |
| --- | ---: | ---: |
| Loss, nats/character | 1.47946 | 1.47651 |
| Perplexity | 4.39058 | 4.37765 |
| Next-character accuracy | 57.03% | 57.08% |

The small differences support similar quality on this sample, not an improvement
in fluency. Greedy predictions differ on 4.44% of targets, confirming that the
context-policy change can alter text. This remains a small Shakespeare character
model, not an instruction-following model.

Full timing and regression results are recorded in
[`kv_cache_verification.json`](../models/character_transformer/kv_cache_verification.json).
Detailed runs are under `build/kv-cache/`. Earlier checkpoint and runtime evidence
is preserved. [REPORT_WALKTHROUGH.md](REPORT_WALKTHROUGH.md) section 11 gives the
whiteboard explanation for the report.

## Remaining work

The first version still padded a single row into a 4×4 tensor shape. Weight transport,
host launch/reset overhead, verification, shared-memory access and the single
tensor unit still cost time. Rebuilds remain necessary for this position policy.
These are verified RTL simulation measurements, not FPGA throughput. Training
on Snackwell and FPGA deployment remain separate roadmap work.
