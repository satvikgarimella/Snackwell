# Improving the character model

**Historical experiment:** the width-128 checkpoint described as selected below
was subsequently removed as the default. Better held-out loss did not establish
acceptable sentence quality, and it more than doubled generation time. It is
preserved in `models/character_transformer/wide-context64/`. The width-64 model
is the default again; see [the rollback and kernel changes](INFERENCE_REPAIR.md).

The earlier speed work preserved the checkpoint. It therefore preserved weak
sentences too. The CPU reference reproduced the bad continuations; numerical
verification was passing because Snackwell was executing that model correctly.
This experiment changes the trained model, separately from the GPU architecture.

## Candidate selection

All models use the same TinyShakespeare corpus and contiguous 90/5/5 training,
validation and test split. The comparison samples 8,192 identical target positions
within each held-out split. Each model receives its own available context. The
target lists are saved with the results; these are different samples from older
reports, so compare rows within this experiment.

| Model | Parameters | Context | Validation loss | Validation accuracy | Test loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous: 4 blocks, width 64 | 144,449 | 32 | 1.4755 | 56.52% | 1.6063 | 53.85% |
| 2 blocks, width 128 | 290,113 | 64 | 1.5007 | 55.24% | Not used for selection | — |
| Selected: 4 blocks, width 128 | 555,073 | 64 | 1.4212 | 57.48% | 1.5401 | 54.50% |

Loss is natural-log next-character cross-entropy, lower is better. The selected
candidate reduces test loss by about 4.1%. The two-block candidate demonstrates
why adding parameters alone does not establish an improvement: its matched
validation results were worse, and it was rejected.

The wider models were trained from scratch in CPU PyTorch, seed 1847, batch 32,
four attention heads and feed-forward width 256. Initialization uses standard
deviation 0.02, with residual projections scaled by `1/sqrt(2 * layers)`.
AdamW decays matrix weights by 0.1, excluding biases and normalization parameters.
Learning rate warms up over 300 updates, then follows cosine decay from 0.001
to 0.0001. Gradient norm is clipped at 1.0. The two-block run used 16,000 updates
and dropout 0.1; the four-block run used 20,000 updates and dropout 0.15. Both
selected their final update by validation cross-entropy. Dropout is disabled
for evaluation, export and inference; it introduces no new RTL operation.

An additional attempt distilled the wider model into the original four-block,
width-64, context-32 model. It resumed the old weights for 8,000 updates, mixing
true-target cross-entropy with the teacher's probability distribution. Every
trained checkpoint had worse validation cross-entropy than the starting model.
The selected step was zero and its weights are exactly the original weights.
This attempt is rejected; it does not establish a faster quality upgrade.
The trainer records teacher checksum, mixing weight and temperature. With a
teacher, `train_batch_loss` is the mixed objective; validation/test losses always
remain ordinary cross-entropy against real next characters.

## What changed in the runtime

Model width means features per character. Snackwell still has **four physical
SIMD lanes and one 4×4 tensor array**. The larger matrices run in pieces through
the same hardware; no RTL source changed in this experiment.

- The checkpoint validator and embedding path accept width 128.
- LayerNorm uses counted loops above width 64 so its program fits the
  1,024-instruction ROM. The mean and variance retain their sequential FP32
  summation order. The looped and unrolled implementations produce identical
  bits on shared supported shapes.
- The 256-feature GELU result is split into bounded contiguous chunks. GELU is
  elementwise, so this reshaping does not change the math or feature order.
- Cache identity is still recomputed from the actual configuration and weights
  at each step. Hashing binary numeric values avoids formatting hundreds of
  thousands of weights as decimal text. Even an in-place edit smaller than an
  FP32 rounding unit changes the signature; the code does not trust a stale
  checkpoint-file checksum.

The selected model holds 2,220,292 bytes of FP32 parameters. Its full 64-character
K/V cache occupies 256 KiB on the host. Both exceed the hardware's 4 KiB shared
memory and are transported in bounded pieces. A complete evaluated window uses
at most 609 instructions and 1,024 memory words in any single launch. Global
memory, DMA and a persistent hardware launch interface are still future work.

## Language checks and limitations

Generation diagnostics retain all eight prompts, four sampled seeds and
96-character continuations, plus greedy output. Sampling uses temperature 0.7
for both old and new models. In those 32 sampled cases, mean repeated-fourgram
fraction fell from 4.94% to 2.82%; threefold short repetition occurred in two old
runs and zero new runs. Greedy generation still loops and its repeated-fourgram
fraction gets worse, from 44.89% to 70.03%.

Training-corpus word coverage was 527/543 for the old sampled outputs and
532/551 for the new ones. This is a spelling proxy, not evidence of improved
grammar or meaning. The selected model has better held-out prediction loss,
but its continuations still contain malformed words, awkward grammar and weak
connections to the prompt. It completes Shakespeare-like text; it is not an
instruction-following chatbot and does not know personal facts.

The longer context uses the existing cache rule: append at fixed learned
positions through character 64, then rebuild the newest 48 characters from
position zero. Short prompts use their actual length. The larger context avoids
rebuilds during a 16-character continuation of the original 32-character prompt.
Changing model/context changes predictions, so this is a quality comparison,
not an output-preserving performance optimization.

## Verification and timing

The installed checkpoint is `933a263b2d03…`. All 105 Python regression methods
pass, with the optional PyTorch tests enabled and no skips. Eight exported
64-character windows pass every RTL launch and whole-model check: 512/512
greedy IDs match PyTorch, with maximum probability difference 1.55e-6.
Target accuracy on these particular windows is 277/512; agreement with a
reference and prediction accuracy are different measurements.

Eight additional trained-model comparisons match full RTL recomputation exactly
at context lengths 3, 4, 5, 6, 63, 64, 48 and 49, including the rebuild.
The source fingerprint remains `75031f1eb062…`. No numerical tolerances were
relaxed. [The historical verification record](../models/character_transformer/wide-context64/verification.json)
includes source/checkpoint hashes, complete comparison files and test evidence.

Timing uses the original prompt `My lord, I have something to say`, seed
92617660 and temperature 0.7, generating 16 characters per run. Two repetitions
per model ran sequentially in old/new/new/old order, after training and tests
finished. Both use the same current runtime and Verilator build, with tracing
off. Compilation is excluded; independent verification is included.

| Metric | Previous model | Wider default |
| --- | ---: | ---: |
| First character | 8.19 s | 24.30 s |
| Cached character | 1.19 s | 3.58 s |
| Subsequent character, including rebuilds | 1.83 s | 3.58 s |
| Complete 16-character run | 35.69 s | 78.06 s |
| Rebuilds in this run | 2 | 0 |
| Simulated cycles | 32,970,230 | 74,031,325 |
| Kernel launches | 7,242 | 16,453 |

These are verified RTL-simulation timings on the development Mac, not FPGA
performance. The wider model is about 2.2 times slower over this run. The
longer window avoids two rebuilds, but larger matrix products outweigh that
saving. Both repetitions produce identical probabilities and selections within
each model. The wider model's actual RTL continuation is:

```text
My lord, I have something to say all to see you:
```

That is the unedited output from the original prompt/seed. It illustrates the
remaining language limitation even though held-out loss improves. All prompts
and longer CPU continuations, including failures, are saved in `quality/`.

The previous faster profile remains available:

```sh
python3 scripts/run_model.py --interactive --tokens 16 \
  --checkpoint models/character_transformer/previous-context32/checkpoint.json
```

CPU training and distillation do not implement training on Snackwell. Model
arithmetic and sampling remain on the SM; the CPU independently checks the
results and transports raw words between bounded launches.

## Reproduce the selected training run

```sh
build/model-venv/bin/python scripts/train_reference_model.py \
  --width 128 --layers 4 --heads 4 --context 64 --batch-size 32 \
  --steps 20000 --learning-rate .001 --min-learning-rate .0001 \
  --warmup-steps 300 --dropout .15 --initialization gpt \
  --weight-decay .1 --seed 1847 --out build/reproduced-wide-model
```

This uses the optional CPU PyTorch environment and the local corpus at
`build/data/tinyshakespeare.txt`. Inference needs neither PyTorch nor a download.
The saved checkpoint includes training configuration, validation history and
dataset fingerprint. Elapsed times in that history are machine-dependent.
