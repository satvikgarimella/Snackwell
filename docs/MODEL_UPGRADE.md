# Improving the character model

This records the earlier upgrade to two blocks. The current four-block checkpoint
and temperature-controlled sampling are described in
[Language quality after the first model upgrade](LANGUAGE_QUALITY.md).

The first model learned weak character patterns. Greedy decoding repeatedly
chose phrases such as “the”; sampling broke those loops but produced many
misspelled words. Matching an independent reference established that the RTL
was executing that model correctly. It did not establish that the model was a
good language predictor.

## What changed

The new checkpoint has two transformer blocks, four attention heads per block,
32-character context, 64 features per character, and a 128-feature feed-forward
layer. Its 77,505 parameters replace the previous 4,657-parameter model. The
vocabulary remains 65 characters. Training still happens in CPU PyTorch;
forward computation and character selection happen in Snackwell RTL.

Each head receives 16 of the 64 Q/K/V features, computes its own causal attention,
and produces 16 output features. The four outputs are concatenated, then the
output projection mixes their information. A second block operates on the first
block's contextual representations. No new hardware instructions were needed.

Three candidates used the same training split, seed 1729, batch size 64,
16,000 AdamW updates, weight decay 0.01, gradient clipping at 1.0, and cosine
learning-rate decay from 0.002 to 0.0002. Selection used validation loss only.

| Candidate | Parameters | Best validation loss |
| --- | ---: | ---: |
| 1 block, width 64, 4 heads, context 32 | 44,033 | 1.7295 |
| 2 blocks, width 32, 4 heads, context 32 | 22,401 | 1.7784 |
| **2 blocks, width 64, 4 heads, context 32** | **77,505** | **1.6080** |

The larger two-block model was selected. The one-block candidate was faster,
but its reference continuations still contained substantially more malformed
words. Full candidate training histories remain in `build/model-upgrade/`.
The previous default checkpoint and its original evidence are preserved in
`models/character_transformer/previous-small/`.

## Making sure the upgrade is actually loaded

A subsequent report of the same `the ` continuation came from a browser server
that had been running for 21 hours. Its saved run for `To be, or not to ` used
the original checkpoint (`846b525279aa…`), 16-character context and old greedy
decoding code. An interactive terminal process was also older than the updates.
Replacing the source and checkpoint files had not changed either process's
already-loaded model. This was a deployment mistake, separate from the new
checkpoint's language quality.

The server was restarted with checkpoint `d63bcfcb2933…`. Both interfaces now
show the model identity and decoding mode. They reload and validate the checkpoint
before each prediction, retaining one model for the complete run; an invalid
replacement produces an error instead of silently reusing old weights. Regression
tests replace checkpoints between actual RTL predictions in both interfaces.
Python source changes still require restarting the process.

A fresh four-character browser run used the upgraded checkpoint with sampling
and produced `To be, or not to purg`. All four RTL predictions passed numerical
checks; the run took 62.1 seconds across 2,632 launches. This checks the deployed
model identity and execution, not fluency. The seven focused interface tests and
the before/after run identities are recorded in
[`session_verification.json`](../models/character_transformer/session_verification.json).

## Measuring language quality

The old and new models were compared at the **same 2,048 held-out next-character
positions**. Each model uses its own context length; neither receives training
gradients from those positions. These are CPU-reference measurements, separate
from the smaller full RTL evaluation.

| Model | Loss (nats/character) | Next-character accuracy |
| --- | ---: | ---: |
| Previous default | 2.1530 | 38.48% |
| One block, width 64 | 1.8217 | 47.07% |
| Two blocks, width 32 | 1.8319 | 47.12% |
| **Selected two blocks, width 64** | **1.7257** | **51.12%** |

The fixed-prompt diagnostic generates 64 characters from six prompts with three
sampling seeds, retaining every continuation. Of the sampled words counted
excluding the first and last words, 68/167 from the old model and 168/192 from the
new model occurred in the training corpus. This is a spelling diagnostic, not
a measure of grammaticality or understanding. Greedy generation still repeats
phrases. Sampling remains the default and uses the same unmodified selection
rule as before.

For example, the new CPU reference extends `My name is ` with seed 1731 to:

```text
my kindred.
You will say him more or try, noble, I could teal th
```

That contains recognizable phrases and an invented word. It demonstrates an
improvement, not reliable conversational ability. A character model trained on
Shakespeare is not trained to answer personal questions or follow instructions.
Short prompts are still left-padded with spaces to the fixed context length.

## Fitting it onto the existing SM

The parameter storage is 310,020 bytes, but the SM still has only 4 KB shared
memory. The host copies raw words between launches. All learned arithmetic and
all reductions stay on the SM:

1. Products with K greater than 32 are split into K chunks. Later launches load
   the actual previous C words and keep accumulating with `FADD`. A host-side
   addition would change both where computation happens and its FP32 rounding.
   Tests require exact agreement with the original four-term tile summation order.
2. Attention scores are split by query rows. The causal mask uses each query's
   original position, so chunking cannot accidentally hide valid earlier tokens
   or expose future tokens.
3. Multiple heads are split/concatenated by copying words. Each head's QK product,
   scaling, softmax and probability-times-V product execute as SM kernels.
4. Embedding launches reuse each four-feature table slice for every context row.
   Matrix launches fit up to eight output rows, sixteen columns and K=32 in
   896 memory words. The widest LayerNorm program uses 954 instructions, below
   the 1,024-instruction ROM limit.

## Keeping generation practical

Evaluation computes every position. Generation needs only the final prediction,
so the final block computes the last four query rows and subsequent row operations
(the hardware uses four-lane/four-row tiles). Earlier blocks still compute all
32 positions, and the last block still computes every key and value. Nothing
needed by the last prediction is skipped. This is not a KV cache.

Tests compare the shortened computation's probabilities and sampled selection
bit-for-bit with the corresponding full computation. The first checked prediction
for `My name is ` used 13,629,445 execution cycles and 658 launches, taking about
15 seconds on the development Mac with Verilator and tracing off. That is slower
than the old small model: quality improved by doing substantially more work.
On a held-out window, final-row generation reduced execution cycles from
22,965,230 to 13,647,553 (40.6%), and launches from 1,032 to 657. Probabilities
and greedy predictions for the four retained rows matched the full computation
bit-for-bit. The comparison is saved in `generation-optimization.json`.
These times are simulation measurements, not projected FPGA throughput.

The independent expected result for each launch is computed once and reused for
verification, rather than recomputed twice. Memory guards, exact FP32 checks,
whole-model tolerances and greedy prediction checks remain enabled.

## Reproduce and inspect

```sh
build/model-venv/bin/python scripts/train_reference_model.py \
  --width 64 --context 32 --heads 4 --layers 2 --steps 16000 \
  --learning-rate 0.002 --min-learning-rate 0.0002 \
  --out build/reproduced-model

build/model-venv/bin/python scripts/compare_models.py \
  --checkpoints models/character_transformer/previous-small/checkpoint.json \
                models/character_transformer/checkpoint.json \
  --data build/data/tinyshakespeare.txt --out build/quality-comparison.json

build/model-venv/bin/python scripts/evaluate_generation.py \
  --checkpoint models/character_transformer/checkpoint.json --backend torch \
  --data build/data/tinyshakespeare.txt --tokens 64 --out build/generation-quality.json

python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
python3 scripts/run_model.py --interactive --tokens 4
```

The optional PyTorch generation diagnostic is explicitly CPU-only. Ordinary
`run_model.py` does not import it and never substitutes its predictions for RTL
results. The portable scalar mathematical reference remains independent of both
PyTorch and the assembly execution path.

For the report, draw a 32×64 activation matrix, split its features into four
32×16 heads, then follow one output row through both transformer blocks. Next
trace a 4×128 by 128×16 product through its four K chunks, showing where C lives
between launches. Finally separate the evidence for model quality, numerical
correctness and simulation speed; each answers a different question.
