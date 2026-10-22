# Full transformer training on Snackwell

The complete causal character transformer now runs forward propagation, mean
next-character cross-entropy, backward propagation and SGD updates on the SM in
RTL simulation. Every parameter is trainable: token/position embeddings, Q/K/V/O
projections and biases, both feed-forward layers, all LayerNorm scales/biases, and
the final vocabulary projection.

This advances beyond the earlier [two-layer GELU training demo](TRAINING.md).
Training correctness and language quality are separate: a short training run can
prove the loss falls and every gradient is correct without producing fluent text.

The [2026-10-21 training audit](TRAINING_AUDIT.md) checks the installed
669,952-parameter subword model against PyTorch autograd and independent FP32
updates, checkpoint reload and inference, plus a fresh 100-step RTL learning run.

## Terminal commands

From the repository root, using the local TinyShakespeare corpus:

```sh
python3 scripts/train_transformer.py
```

The default starts from random weights: one block, two heads, width 16, context
16, 65 characters and 4,657 parameters. It runs 100 SGD updates, one sequence per
step, with learning rate 0.05. `--data PATH` selects another UTF-8 corpus with
2..128 characters. The runner never downloads text or trains with PyTorch.
Imported [subword checkpoints](SUBWORD_MODEL.md) can instead use up to 512 token
IDs. Their saved tokenizer is preserved through training and resume; tokens may
contain multiple characters.

To train the existing larger character model:

```sh
python3 scripts/train_transformer.py \
  --checkpoint models/character_transformer/checkpoint.json \
  --steps 3 --evaluation-windows 2
```

Imported weights keep their architecture; the default learning rate is 0.001.
Every step computes the full context and all gradients. Training does not reuse
an inference KV cache across parameter updates.

The terminal prints the new run directory and checkpoint. Continue that exact run:

```sh
python3 scripts/train_transformer.py \
  --resume build/transformer-training/<run>/checkpoint.json --steps 100
```

Resume preserves the corpus hash, seed, learning rate and update count. It selects
the same next sequence as uninterrupted execution. It rejects a different corpus.
The runner checks the entire step before atomically replacing the checkpoint.
Ctrl+C or a failed check preserves the last committed update. `--out` must name a
new directory, preventing stale results from being mistaken for a successful run.

Use the exported weights with the existing inference command:

```sh
python3 scripts/run_model.py \
  --checkpoint build/transformer-training/<run>/checkpoint.json \
  --interactive --tokens 4
```

The default inference checkpoint is not overwritten. The optional `--simulator
icarus` and `--trace` flags retain all numerical checks. Verilator is selected
when available; an initial compilation takes additional time.

## What a step does

The forward pass is the same architecture as the existing inference model:

```text
token + position embeddings
  -> LayerNorm -> causal multi-head attention -> projection + residual
  -> LayerNorm -> linear -> GELU -> linear + residual
  -> repeat for each block
  -> final LayerNorm -> vocabulary logits -> cross-entropy
```

The backward pass starts with `(probability - one_hot_target) / context_length`.
For each dense projection it computes `dW = X^T dY`, `db = sum_rows(dY)` and
`dX = dY W^T`. Residual branches add their contributions on the SM. Attention
computes gradients for both products, `Q K^T` and `P V`, including the
`1/sqrt(head_width)` scale. Softmax backward uses
`dS = P * (dP - sum(P*dP))`; masked positions have zero probability and zero
gradient. The first causal row has a constant one-element distribution and zero
score gradient.

LayerNorm backward computes row means and inverse standard deviations in RTL,
then its input gradient and per-row scale gradients. Column reductions produce
the scale and bias gradients. Token IDs that occur repeatedly accumulate all
their embedding gradients; unseen IDs receive zero gradient. Position embeddings
receive the corresponding row gradients.

All gradients use the weights from the start of the step. SGD then updates each
parameter with separately rounded FP32 multiplication and subtraction. The output
preserves the imported checkpoint's format and tokenizer. The same inference
runner accepts the resulting weights.

## Stable loss without a logarithm instruction

The loss kernel evaluates
`max(logits) - target_logit + log(sum(exp(logits - max(logits))))`.
It never takes the logarithm of a target probability that may have underflowed to
zero. The exponential sum lies between 1 and the padded vocabulary size, up to
512. The kernel extracts its binary
exponent, looks up the exponent times `log(2)`, and approximates the mantissa's
logarithm with a bounded odd polynomial. These operations all use existing integer
and FP32 instructions. Tests include a correct-class probability that underflows
while the correct loss remains finite at about 2,000.

GELU, softmax and LayerNorm backward approximate derivatives of the smooth model
functions. They are checked numerically; they are not derivatives of discrete
FP32 rounding or of each individual approximation instruction.

## Verification

Each launch verifies outputs and every memory word outside the output region.
Exact FP32 arithmetic uses the existing integer oracle. New approximate kernels
have fixed absolute/relative limits: LayerNorm backward 1e-5/3e-4, softmax backward
3e-6/2e-4, and cross-entropy 5e-6/3e-4. The independent complete training graph
checks every observed forward tensor, every parameter gradient, the embedding
activation gradient and every updated parameter with absolute tolerance 2e-5 plus
relative tolerance 2e-3. Existing inference and feed-forward training tolerances
were not loosened.

The complete reference is a separate double-precision reverse-mode implementation
built from the original checkpoint and token IDs. It never reads RTL activations
or gradients. Tests compare it with PyTorch autograd for one, two and four blocks,
and one, two and four heads. All 241 parameter gradients in the smallest model
are also checked using central finite differences.

For the existing four-block, four-head, width-64, context-32 checkpoint, all
**144,449 RTL gradients** were additionally checked directly against PyTorch
float64 autograd. The largest absolute error was **3.94e-7**. That step took
19.3 seconds, 31,339,083 simulated cycles and 5,731 launches. Its raw evidence is
under `build/transformer-training/current-model-step/`.

The CLI also completed three updates of that existing model. Its fixed training
sample loss fell from 1.304591 to 1.284718; validation changed from 1.492523 to
1.492698 on 64 targets. That small validation increase gives no evidence of
improved language quality, so the default inference checkpoint was preserved.
The three updates plus initial/final evaluations took 122.9 seconds; the checked
export is in `build/transformer-training/current-model-3/checkpoint.json`.

A subsequent [100-update experiment on the existing model](RTL_FINETUNING.md)
checked 8,192 fixed validation targets and 32 fixed generation cases. All updates
passed, but next-character loss changed from 1.449532 to 1.450448 and the samples
still contained grammar problems. The original inference checkpoint was kept.
This is separate from the smaller model's 100-step learning run below.
The [follow-up training experiments](RTL_FINETUNING.md#follow-up-training-and-model-selection)
add explicit quality reporting and test alternative host-training recipes and
feed-forward capacity. Host-trained candidates and verified RTL updates remain
separately identified.

The tests also cover Icarus/Verilator agreement for complete-step tensor words
and cycle counts, repeated embeddings, zero-variance normalization, shared-memory
chunk boundaries, causal masking, invalid inputs, deliberately corrupted backward
results, cancellation, exact resume, and inference using updated weights.

The full Python regression passed **137 test methods**, including 11 new
transformer-training tests, with no skips. The optional PyTorch and Verilator
checks both ran. The saved 100-step export also passed a separate CLI resume
check at step 101 and runs through the normal inference command.

## Measured learning run

The default 100-step TinyShakespeare run produced:

| Measurement | Before | After |
| --- | ---: | ---: |
| Training cross-entropy | 4.755490 | 3.473169 |
| Validation cross-entropy | 4.758931 | 3.241008 |

It took 75.8 seconds with a cached Verilator build, including reference checks and
artifact writes, and executed 92,386,171 cycles. These are simulation measurements,
not FPGA throughput. Most individual updates took about 0.65 seconds for this
4,657-parameter model. The larger 144,449-parameter model is substantially slower.

The corpus is split contiguously into 90% training, 5% validation and 5% reserved
test text. The same eight evaluation windows per split are used before and after
training: 128 target characters each. Validation windows never contribute
gradients; the reserved test split is not used in this run. The sample is too small
to establish broad language quality. Per-step losses vary with the sampled text.

The raw run is `build/transformer-training/shakespeare-100/`. A portable copy and
compact evidence are in `models/rtl_transformer/`. Each run saves the starting and
final checkpoints, source/corpus hashes, per-step loss history, gradients,
intermediates, assembly, memory images and simulator logs. Full launch artifacts
are substantial: this 100-step run used about 1.7 GB under the ignored `build/`
directory. Add `--trace` only when cycle-level diagnostics are needed.

## Remaining work

The full SGD training path now works in simulation. Remaining work includes longer
training and evaluation, language quality, larger batches/gradient accumulation,
optimizer choices, kernel throughput and memory movement. Hardware global memory,
physical FPGA deployment and FPGA resource/timing measurements are still separate
unfinished tasks. The current simulation host moves raw tensors through 4 KiB of
shared memory, rearranges layouts, supplies labels and constants, and performs
independent checks; it does not supply execution results or gradients.
