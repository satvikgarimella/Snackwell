# Language quality after the first model upgrade

This page records the earlier width-64 model and its historical measurements.
The width-128 experiment is described in [the capacity experiment](MODEL_CAPACITY.md).
It was subsequently removed as the default; the width-64 checkpoint is restored.
See [the rollback and current runtime](INFERENCE_REPAIR.md).

The later [decode optimization experiments](DECODE_OPTIMIZATION.md) tested four
additional trained candidates on 8,192 matched validation targets. Their mixed
results did not justify replacing the context-32 default. That section records
the rejected candidates and separate test comparison; faster execution alone
does not improve language quality.

The terminal run with checkpoint `d63bcfcb2933`, seed `92617660` and prompt
`My lord, I have something to say` produced ` Romeosed prove,`.
Unlike the earlier stale-server problem, this run used the intended weights.
The CPU reference reproduced it with the same seed. Numerical agreement was
working; the learned distribution and sampling still produced poor language.

## Training and inference are separate

`train_reference_model.py` uses CPU PyTorch to compute gradients and update the
weights. Snackwell training is still future work. `run_model.py` loads fixed
weights and executes the forward pass and character selection on the SM in RTL
simulation. Retraining the reference model improves the input checkpoint for
that inference demonstration; it does not implement GPU backpropagation.

The original two-block model was continued for 8,000 updates at a smaller
learning rate, with a fresh AdamW optimizer and cosine decay. Its best validation
loss improved only from 1.6080 to 1.5974; on the same 2,048 test targets,
loss changed from 1.7257 to 1.7070 and accuracy from 51.12% to 51.81%.
That modest change did not resolve malformed words and weak sentences.

## Sampling temperature

Sampling at temperature 1 uses the entire predicted distribution. A less likely
character can begin a misspelled word, and later predictions condition on that
mistake. Lower temperature concentrates probability on more likely characters.
It changes how the model's predictions are used, not its trained weights.

The SM computes a second output softmax for sampling:

```text
sampling_probability = softmax((logit - maximum_logit) / temperature)
```

The scaling is an `FMUL` inside the softmax assembly loop. Cumulative probability
selection also executes on the SM. The CPU supplies only the scalar random draw
and the temperature constant. Raw logits, original probabilities and greedy
predictions remain checked and saved separately. Evaluation loss always uses
the original distribution, regardless of the generation temperature.

There are no hard-coded completions, word filters or prompt-specific weights.
Lower temperature is not a guarantee of grammatical output; too little randomness
can bring repetition back. Diagnostics retain all prompts, seeds and outputs.
Word coverage against the training corpus is only a spelling proxy, not a score
for meaning or sentence structure.

## Fast quality checks

```sh
build/model-venv/bin/python scripts/preview_model.py \
  --prompt 'My lord, I have something to say' --tokens 64 \
  --seed 92617660 --temperature 0.7

build/model-venv/bin/python scripts/preview_model.py --interactive

build/model-venv/bin/python scripts/evaluate_generation.py \
  --checkpoint models/character_transformer/checkpoint.json --backend torch \
  --data build/data/tinyshakespeare.txt --tokens 64 --temperature 0.7 \
  --seeds 92617660 1729 1730 1731 --out build/generation-quality.json
```

The preview explicitly runs on the CPU and does not claim hardware verification.
It never replaces an RTL result. Use the same checkpoint, prompt, seed and
temperature in the RTL command when checking the implementation:

```sh
python3 scripts/run_model.py --stream \
  --prompt 'My lord, I have something to say' --tokens 16 \
  --seed 92617660 --temperature 0.7
```

In an updated interactive session, `/temperature T` and `/seed N` adjust these
settings without restarting. Python code changes still require a restart.

## Longer-context model support

The checkpoint schema now permits four blocks and a 64-character context, with
the same width-64 arithmetic and four attention heads. A 64-row embedding lookup
is split into two groups of 32 rows per four-feature table slice. Each group
uses the correct absolute positional embeddings. Even a 128-character vocabulary
fits in 800 shared-memory words per lookup launch.

Tests cover these row boundaries, causal-mask offsets through row 63, four-block
execution, and full-versus-last-row generation. Temperature tests check masked
classes, large logits, probability sums and independent whole-model agreement.

## Selecting the replacement

Both larger candidates were trained from scratch for 20,000 AdamW updates on
the same training split, seed 1729, batch size 64 and cosine learning-rate
schedule from 0.002 to 0.0002. Checkpoint weights were selected by validation
loss. The model comparison used identical 2,048 next-character targets within
each split, letting each model use its own context length.
Training's logged validation loss averages all positions in each window; the
comparison below scores the next character after the full available context.

| Model | Validation loss | Validation accuracy | Test loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original 2 blocks, context 32 | 1.5217 | 55.62% | 1.5982 | 53.52% |
| Continued training, same shape | 1.5119 | 55.32% | 1.5871 | 53.81% |
| 4 blocks, context 64 | 1.4411 | 57.42% | 1.5653 | 54.93% |
| **4 blocks, context 32** | **1.4398** | **58.15%** | **1.5284** | **55.76%** |

The four-block, context-32 candidate wins the validation comparison and requires
less work than the longer-context model. It has 144,449 parameters. Its first
checked sampled RTL character took 34,236,950 cycles and 37.5 seconds; the
context-64 model took 74,344,773 cycles and 89.6 seconds on the same prompt.
The shorter model is selected for deployment. These are simulator timings,
not FPGA projections. It is still slower than the previous two-block model.

The test target positions differ from the earlier upgrade's comparison because
the candidate set now includes context 64. The table above compares all models
on exactly the same saved positions; its scores should not be mixed with the
older comparison's different targets.

Across eight fixed prompts and four seeds, 294/336 completed sampled words from
the original model at temperature 1 appeared in the training corpus (87.5%).
The selected model at temperature 0.7 scored 326/340 (95.9%); these 32 runs had
no threefold repeated substring. With four additional seeds, it scored 349/359
(97.2%) with one repetition case in 32 runs. All outputs are retained, including
errors. This improvement combines retraining and decoding; it is not solely an
effect of the extra blocks. At temperature 1, the selected model scored 294/323
(91.0%). Sentence grammar remains inconsistent.

## RTL verification and deployment

The timings below record the model-quality upgrade. Later kernel and simulator
optimizations preserve its outputs and reduce a matched character run from
36.24 to 20.65 seconds; see [the speed investigation](INFERENCE_SPEED.md).

The installed checkpoint is `43f211daad7c…`. The full regression passed 37 RTL
testbenches and 74 Python tests; four focused browser integration tests also
passed after extending their checks to temperature-adjusted probabilities.
All 256 positions across eight exported windows passed numerical verification,
with 256/256 greedy IDs matching PyTorch and a maximum probability error of
2.81e-6. Actual next-character accuracy on those smaller windows was 140/256.

The original user prompt and seed `92617660`, using the new checkpoint and
temperature 0.7, produced these 16 characters on RTL:

```text
My lord, I have something to say against my word
```

All sixteen predictions matched the CPU diagnostic, averaging 38.6 seconds per
character. This is a checked short continuation, not a claim of reliable grammar
or conversational ability. The full outputs and fingerprints are stored in
`models/character_transformer/verification.json` and `generation_checks.json`.
The prior checkpoint and evidence are preserved under `previous-two-block/`.

The browser backend was restarted and its checkpoint/temperature verified.
An existing interactive terminal must exit with `/quit` and rerun `run_model.py`
once, because checkpoint reload alone cannot update already-imported Python code.
