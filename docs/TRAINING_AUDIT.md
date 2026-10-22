# Training verification — 2026-10-21

The current training implementation passed a fresh audit. It computes correct
gradients and SGD updates for the installed subword model, preserves checkpoints
through reload, and learns from random initialization in a separate small-model
run. These checks exercise RTL simulation. They do not establish FPGA execution
or fluent language generation.

## Installed model

The audited checkpoint is `models/subword_transformer/checkpoint.json`, SHA256
`964d603faa39de1b90c0bd31e2a5e0e66402d5cd10e2f1fc61ae183431cb7ba8`.
It has four blocks, four heads, width 128, context 64, a 512-token vocabulary,
and 669,952 parameters in 70 tensors. Its original pretraining used CPU PyTorch
and AdamW. This audit performs two additional, separate **RTL SGD** updates;
it does not replace the installed weights or claim that pretraining ran on RTL.

Both updates used full 64-token training windows, seed 1729 and FP32 learning
rate 0.001. The second update started from the first update's saved and reloaded
checkpoint. The corpus SHA256 is
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
The Verilator RTL fingerprint is `592ce28d04d3...`.

For each update:

- Every forward tensor, loss, parameter gradient and updated parameter passed
  the existing independent double-precision reference checks.
- All 669,952 parameter gradients also passed a direct comparison with PyTorch
  float64 autograd. The existing tolerances were unchanged.
- Every updated FP32 word matched separate PyTorch FP32 multiplication and
  subtraction using the RTL gradient, bit for bit. Updated weights also passed
  comparison with updates computed from the independent autograd gradients.
- All kernel outputs and shared-memory guards passed their existing checks.
- Saving and reloading preserved every weight and the tokenizer exactly.

| Check | Update 1 | Update 2 |
| --- | ---: | ---: |
| Largest absolute gradient error against PyTorch | 9.25e-7 | 3.62e-7 |
| RTL loss before update | 2.180006 | 3.080021 |
| Independent same-batch loss before update | 2.180006 | 3.080021 |
| Independent same-batch loss after update | 2.161469 | 3.039702 |
| Parameter values that changed | 604,775 | 604,020 |
| Simulated cycles | 231,374,367 | 231,369,949 |
| Kernel launches | 37,791 | 37,775 |

The same-batch before/after losses use the independent PyTorch model. Each row
pair compares identical inputs and targets; the two updates use different
training windows. Unseen embedding rows legitimately have zero gradient, and
some small updates round to the previous FP32 value.

The final checkpoint also passed the ordinary RTL inference path on eight
input tokens. A fixed CPU evaluation of 16 validation windows (1,024 target
positions) changed from 3.046036 to 3.045843. That small change after two updates
does not establish a useful language-quality improvement. The candidate remains
in the audit directory, and the installed checkpoint's hash is unchanged.

## Learning from random weights

A separate run used the normal training CLI with its 4,657-parameter model:
one block, two heads, width 16, context 16, 100 RTL SGD updates, learning rate
0.05 and seed 1729. It evaluated the same 32 windows per split before and after
training, giving 512 sampled target positions per split. Training windows came
only from the training split; no test targets were used.

| Mean cross-entropy | Before | After |
| --- | ---: | ---: |
| Training sample | 4.688200 | 3.414652 |
| Validation sample | 4.638922 | 3.491582 |

All 100 forward/backward/update checks passed. Including the initial and final
RTL evaluations, this run executed 132,994,946 cycles and 37,365 launches.
The reduced held-out loss establishes learning on this sample; it does not
establish fluent text generation or transfer this improvement to the larger
pretrained model.

## Regression coverage and change

All 38 focused test methods passed with PyTorch and Verilator enabled, with no
skips. They cover finite differences, independent autograd, Icarus/Verilator
agreement, repeated embedding IDs, causal masking, zero-variance normalization,
memory chunk boundaries, exact uninterrupted/resumed equivalence, tokenizer
preservation, invalid inputs, deliberately corrupted backward results and safe
cancellation.

The cross-entropy underflow test was extended from vocabulary sizes 4 and 128
to include 512, matching the installed subword model. That expanded test passed
separately. It covers extreme logits, a correct-class probability that underflows
while its loss remains finite, uniform logits and shifted logits. No arithmetic,
optimizer, runtime or numerical-tolerance change was needed.

## Reproduction and evidence

From the repository root, with the existing PyTorch environment:

```sh
build/model-venv/bin/python -m unittest \
  tests.test_transformer_training tests.test_subword_model \
  tests.test_training tests.test_reference_training -v

python3 scripts/train_transformer.py \
  --steps 100 --evaluation-windows 32 --log-every 10 --simulator verilator
```

To run two verified SGD updates on the installed subword model through the
ordinary CLI, including initial/final RTL evaluation:

```sh
python3 scripts/train_transformer.py \
  --checkpoint models/subword_transformer/checkpoint.json \
  --steps 2 --evaluation-windows 1 --log-every 1 --simulator verilator
```

The audit's additional direct PyTorch comparisons and raw evidence are retained
under `build/training-audit-20261021/`:

- `audit_current_model.py`: the exact full-size audit driver.
- `current_model/result.json`: per-tensor autograd errors, update and reload
  checks, validation losses, source/checkpoint identities and inference result.
- `current_model/step_001/` and `step_002/`: actual RTL intermediates, gradients,
  updated parameters, assembly, memory images and simulator output.
- `current_model/checkpoint_step_002.json`: the experimental updated weights.
- `learning/result.json` and `learning/history.jsonl`: all 100 updates and
  before/after RTL evaluation results.
- `tests.log`, `loss-boundaries.log`: regression results.
- `source-manifest.json`, `verified-source-manifest.json`: hashes before and
  after the audit; the only source change was the added 512-token test case.

The small and full-size runs overlapped. Their wall times are diagnostic records,
not isolated speed benchmarks. No installed model was promoted or overwritten.
