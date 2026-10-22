# Training on Snackwell

Snackwell can train a two-layer GELU network in RTL simulation. Both dense layers,
the activation, loss, parameter/input gradients and SGD updates execute through
the real SM. This document records that first training milestone. The complete
character transformer now also supports backward propagation and SGD; see
[full transformer training](TRANSFORMER_TRAINING.md). Training exports separate
checkpoints and preserves the existing default character-model checkpoint.

## Run it in the terminal

From the repository root:

```sh
python3 scripts/train_snackwell.py
```

The default network has 8 inputs, 16 hidden units, 4 outputs and 212 trainable
parameters. It starts from seeded random weights, trains for 200 updates with
batch size 16 and SGD learning rate 0.2, then evaluates every training and held-out
example. Python's standard library and an RTL simulator are sufficient. Verilator
is selected when available; use `--simulator icarus` to choose Icarus explicitly.
`--trace` saves cycle traces. Both modes run all numerical checks.

Example from the verified run:

```text
Start: train loss 0.172689 | held-out 0.137247
Step    1: batch loss 0.163583 | 73,843 cycles | gradients + updates verified
Step  100: batch loss 0.049805 | 73,931 cycles | gradients + updates verified
Step  200: batch loss 0.042842 | 73,787 cycles | gradients + updates verified
Train loss:    0.172689 -> 0.033176
Held-out loss: 0.137247 -> 0.038508
```

Batch loss varies because each update uses a different selection of examples.
The before/after losses use the same complete datasets and are directly comparable.
The held-out examples never participate in gradient updates. Lower held-out loss
demonstrates learning on this regression task; it says nothing about the language
quality of the separately trained character model.

The final captured run took 12.7 seconds with a cached Verilator build on this Mac, including
verification and artifact writes. It executed 15,146,210 simulated cycles,
including initial/final evaluation. This is host simulation time, not FPGA timing
or transformer-training performance. First-time compilation takes additional time.

The terminal prints a run directory under `build/training/`. Resume from its
checkpoint for another 100 updates:

```sh
python3 scripts/train_snackwell.py --resume build/training/<run>/checkpoint.json --steps 100
```

Resume preserves the dataset seed, learning rate and completed-step count, so the
next batch and all parameter words match uninterrupted execution. `--out` accepts
a new directory; an existing directory is rejected to protect earlier evidence.
Ctrl+C stops the current run while preserving the last fully committed checkpoint.
Any failed numerical check stops training with a nonzero exit status.

## The task

Inputs are eight FP32 numbers sampled uniformly from [-1, 1]. The four targets are:

```text
y0 = sin(x0) + 0.25*x1*x2
y1 = 0.5*(x2 + x3)^2 - 0.3
y2 = x4*tanh(x5)
y3 = 0.5*x6 - 0.25*x7 + 0.3*x0*x4
```

The host creates 64 training examples and 32 separate held-out examples. These
formulas generate the labels; they do not implement the network's prediction.
The terminal shows before/after predictions for the first output of four examples.
All four outputs of every example are checked and saved.

## One training step

For batch size N and output width O:

```text
z    = X W1 + b1
a    = GELU(z)
y    = a W2 + b2
loss = sum((y - target)^2) / (2*N*O)

dy  = (y - target) / (N*O)
dW2 = transpose(a) dy       db2 = sum_rows(dy)
da  = dy transpose(W2)
dz  = da * GELU'(z)
dW1 = transpose(X) dz       db1 = sum_rows(dz)
dX  = dz transpose(W1)

parameter = parameter - learning_rate * gradient
```

All gradients use the parameters from the start of the step. The four parameter
tensors are updated only after gradient computation. The new parameter state is
committed to disk after every launch and the complete graph have passed checks.
The caller's old parameters are never mutated in place. Input gradients are
computed and verified even though the demo does not train its input data; they
are needed when composing this network with earlier layers.

Dense products reuse the existing tiled `TMUL` kernels. Their FP32 accumulation
order is unchanged. New assembly kernels implement the loss, its derivative,
GELU backward, column reductions for bias gradients and SGD. SGD multiplication
and subtraction round separately; the tensor array also uses separate multiply
and add operations.

GELU uses the existing tanh approximation. Its backward kernel approximates the
analytic derivative of that same smooth tanh-GELU function. It uses
`exp(-abs(v))/(1+exp(-abs(v)))^2` for the sigmoid derivative to avoid cancellation
in the tails. This is an approximation to the smooth derivative, not a derivative
of the discrete FP32 rounding operation. The forward and backward approximations
are checked against independent mathematical references.

The host initializes parameters and labels, selects batches, transports raw tensor
words and rearranges transposed layouts. It also runs independent verification.
No host-computed predictions, gradients or updates are fed into the execution
path. Tensors larger than 4 KiB are split across launches, with raw partial sums
carried between chunks. This still uses simulated shared memory and host transport;
it does not add hardware global memory.

## Verification and evidence

Each launch checks the entire output and all memory outside the output region.
Matrix products, loss, MSE gradient, bias reductions and SGD use an integer FP32
oracle that follows the documented rounding/accumulation order. GELU backward
uses absolute tolerance 3e-6 plus relative tolerance 2e-4. An independent
double-precision forward/backward graph starts from the original parameters,
inputs and targets and checks every intermediate, gradient and updated parameter
with absolute tolerance 5e-6 plus relative tolerance 5e-4. It never consumes RTL
intermediates. Existing inference tolerances were not changed.

`tests/test_training.py` covers:

- Central finite differences for all 212 parameter gradients and 32 input gradients
  in a four-example batch; optional independent PyTorch autograd comparison.
- Identical complete-step tensor words, launch counts and cycles in Icarus and
  Verilator.
- SGD rounding, signed zero/subnormals, GELU's negative and saturated tails, and
  reduction/vector/matrix splits across shared-memory limits.
- A deliberately wrong backward result rejected by the whole-graph check even
  after each individual launch passes its own contract.
- Rejection of nonfinite inputs, overflow, invalid learning rates and checkpoints;
  cancellation leaves weights and the last committed checkpoint intact.
- Repeated RTL updates lowering both training and held-out loss, plus saved/resumed
  execution reproducing uninterrupted updates.

The full Python regression passed all 126 test methods, including 13 training
tests, with no skips. The optional PyTorch and Verilator comparisons both ran.
Across the captured 200-step run, the largest absolute parameter-gradient error
against the independent double-precision graph was 1.63e-8.

Each run saves `data.json`, `run.json`, `initial_checkpoint.json`, `checkpoint.json`,
`history.jsonl` and the final `result.json`. Every step directory contains
`step.json` with raw intermediate/gradient/updated tensor words and error metrics,
along with each launch's assembly, layout, memory images and simulator log.
Evaluation directories contain predictions, RTL loss and their checks.

The captured 200-step checkpoint and compact results are in
[`models/gelu_network/`](../models/gelu_network/). The recorded raw run is
`build/training/verified-ffn-200-final/`. Its `run.json` records the RTL identity and
hashes of the Python sources; the saved verification record also hashes the data
and checkpoint. A preceding 200-step run reproduced the same checkpoint words
and cycle count.

## Following milestone

LayerNorm backward, attention/softmax backward, residual gradient accumulation,
embedding/output-projection gradients and cross-entropy loss are now connected
and checked as a [complete transformer training step](TRANSFORMER_TRAINING.md).
Longer runs, language quality, more advanced optimizers, sustained throughput and
memory movement still need work. FPGA work remains after training in the roadmap.
