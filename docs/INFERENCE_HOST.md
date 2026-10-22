# Inference host and model experiments, October 16

This update improves the simulation host. It does not change the SM, ISA,
executed kernels, default weights, sampling policy, or numerical tolerances.
The default checkpoint remains `43f211daad7c…` (four blocks, width 64,
context 32). Its sentence quality remains limited.

## Direct clock driver

Untraced Verilator sessions now use `sim/session.cpp`. The host sets inputs,
evaluates the compiled RTL on both clock edges, and waits for all selected
warps to halt. It rejects completion with pending memory, FP32, tensor or
writeback work. This removes the general timed-testbench scheduler from this
path; it does not calculate any model result in C++.

The simulation wrapper exposes raw program/shared-memory loading and result
readback. Each launch resets the SM and loads complete images. Input parsing
rejects truncated images, unknown digits, overflow and invalid launch controls.
Every launch still saves its inputs, outputs, registers, assembly and checks.
The host also converts each weight tensor and memory image in bulk, preserving
the exact FP32 words.

Traced sessions and one-shot launches retain the original SystemVerilog
testbench. Both host sources enter the build cache and model-run fingerprint.
Enabling tracing on an existing direct session restarts it with the traced
testbench. A failed or cancelled launch closes its process; it cannot advance
the model cache.

## Measured speed

Two runs of each host used the same default checkpoint, prompt
`My lord, I have something to say`, seed 92617660 and temperature 0.7, in
before/after/after/before order. Training, evaluation and tests were stopped.
Compilation and cycle tracing were excluded; all numerical checks and artifact
writes were included.

| Work | Previous host | Direct-clock host |
| --- | ---: | ---: |
| First character, including prompt | 7.64 s | 5.16 s |
| Mean cached character | 1.098 s | 0.741 s |
| Mean context rebuild | 5.89 s | 3.95 s |
| All 16 characters, including two rebuilds | 33.69 s | 22.70 s |
| Simulated cycles | 31,380,631 | 31,380,631 |
| Kernel launches | 7,242 | 7,242 |

Overall wall time falls 32.6%; cached latency falls 32.5%. Every probability,
sampling distribution, selected character and error metric is identical across
all four runs. Their unchanged continuation is ` 'twas. There's `.
The hardware executes the same number of cycles. This is faster RTL simulation,
not a hardware-throughput or FPGA measurement. Prefill and context rebuilding
still cause pauses.

## Verification

All 113 Python test methods pass with PyTorch and Verilator installed, with no
skips. They include cross-simulator FP32 edge cases, lane masks, tensor rounding,
memory/register/cycle equality, repeated resets, nonzero/invalid PCs, timeouts,
trace fallback, malformed input rejection, and full/cached model execution.
The first regression attempt was blocked only in localhost server setup by the
execution sandbox; the complete rerun passed with local socket access.

All eight exported default-model windows (256 targets) pass again against
PyTorch and the independent scalar model. Their total is 81,981,271 simulated
cycles, with the same 1.605263 held-out loss. These checks establish numerical
agreement for the tested inputs. They do not establish fluent language.

The standalone RTL bench sweep was not rerun for this host change. Its preceding
39-bench result is separate historical evidence.

The [verification record](../models/character_transformer/host_verification.json)
records hashes, tests, timing and model-comparison evidence. The preceding
`verification.json` retains its original historical results.

## Reference-training experiments

The trainer now supports `--device mps` on an available Apple GPU; CPU remains
the default. It exports portable CPU FP32 weights and checks them against the
independent scalar model. This is reference training on the Mac, not training
on Snackwell.

A new width-64/context-128 model was trained from scratch for 30,000 updates.
On 8,192 identical validation targets it scored loss 1.5818 and accuracy 52.25%,
versus 1.4637 and 56.02% for the default. Its 32 sampled continuations also had
more malformed words and repeated substrings. It was rejected.

A separate run continued the archived width-128/context-64 model for 12,000
updates, with dropout 0.05 and a decaying learning rate. Validation selected
update 11,500. The following comparison uses its own shared target positions
for all three models; do not mix its default score with the context-128 trial.

| Model | Validation loss | Validation accuracy | Test loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Default, width 64/context 32 | 1.4755 | 56.52% | 1.6063 | 53.85% |
| Archived width 128/context 64 | 1.4212 | 57.48% | 1.5401 | 54.50% |
| Continued width 128/context 64 | 1.3805 | 59.05% | 1.5015 | 56.41% |

The continued model improves test loss by 6.52% relative to the default. Across
eight fixed prompts and four sampling seeds at temperature 0.7, 736/744 completed
words occur in the training corpus, versus 725/748 for the default. This is only
a spelling diagnostic. The full 128-character continuations still contain
grammatical and semantic errors; greedy decoding repeatedly loops. It is a
better candidate, not a fluent language model, and has not replaced the faster
default. All continuations, including failures, are retained under
`models/character_transformer/quality/host/`.

The continued wider model also completed a 16-character RTL run with every
numerical check enabled. It took 15.40 seconds for the first character and
2.22 seconds per cached character (48.63 seconds total), using 67,795,042 cycles
and 16,453 launches. Its text matched the same-seed CPU diagnostic's prefix:
` 'I,' will not s`. This is one execution check, not a full held-out RTL
evaluation of the candidate. Its latency and remaining grammar problems do
not justify replacing the faster default. Candidate weights and export data
remain in `build/inference-quality-v4/wide-refined/`.

Training on Snackwell still needs backward kernels, loss computation, gradient
checks, parameter updates and a verified repeated-step training run. Reference
retraining does not complete those tasks. FPGA work follows inference and
training.
