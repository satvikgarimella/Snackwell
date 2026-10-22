# Subword language-model experiment

The candidate is saved separately at `models/subword_transformer/checkpoint.json`.
The default character checkpoint remains unchanged. To try the candidate:

```sh
python3 scripts/run_model.py --checkpoint models/subword_transformer/checkpoint.json \
  --interactive --tokens 16 --seed 92617660 --temperature 0.7
```

For this model, `--tokens` counts subword predictions. Sixteen predictions can
produce more than sixteen characters. The existing browser display is character
only; use the terminal for subword inference.

The character model's numerical checks pass, but its sentences often break down.
Another 100 verified RTL SGD updates did not improve its validation loss. This
experiment changes the text representation and trains a larger candidate on the
host before testing that candidate on Snackwell.

The candidate has four blocks, four heads, width 128, feed-forward width 256,
64 tokens of context, and 669,952 parameters. Its vocabulary has 512 entries:
65 characters and 447 learned combinations of letters, sometimes preceded by a
space. A prediction can emit part of a word or a whole word. Training text averages
2.17 characters per token; generated text can have a different ratio.

The 90%/5%/5% corpus split happens **before** learning the tokenizer. Merges are
learned from training text only. The checkpoint stores the alphabet, ordered
merges and vocabulary, so inference and resumed training use exactly the same IDs.
Unknown characters are rejected instead of silently replaced.

## Training and selection

These weights were trained with **PyTorch on the CPU**, not on Snackwell: 16,000
AdamW updates, batches of 32 sequences, seed 2631, dropout 0.15, matrix weight
decay 0.1, GPT-style initialization, gradient clipping at 1, and cosine learning
rates from 0.0015 to 0.00015. Fixed full-context validation targets selected
step 16,000. The dataset SHA256 is
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

The portable checkpoint SHA256 is
`964d603faa39de1b90c0bd31e2a5e0e66402d5cd10e2f1fc61ae183431cb7ba8`.
The training sources, full history, CPU samples and experiment records are under
`build/subword-repair-20261020/`. Export changes format identifiers and records the
prototype checksum; it does not change weights.

## Language quality

Character and subword token losses are not directly comparable. This measurement
scores the same 16 nonoverlapping, newline-bounded passages with each model's own
context and the actual periodic cache-rebuild policy. It reports the negative log
likelihood of the canonical token sequence, divided by target characters. It does
not sum over alternate tokenizations and is not a character accuracy score.
Lower is better.

| Model | Validation, 6,533 characters | Reserved test, 6,494 characters |
| --- | ---: | ---: |
| Existing character default | 1.372564 | 1.542716 |
| Subword candidate | 1.280377 | 1.454787 |

These are 6.7% and 5.7% reductions. Test passages were scored after validation
selected the candidate. A smaller width-64 subword trial scored 1.380540 on the
validation passages and was rejected. The gain combines representation, context,
capacity and training changes; this experiment does not isolate each cause.

All eight fixed prompts, three seeds and at least 96 generated characters per
case are retained in `final-generation.json`. Some continuations are readable;
others still invent words or use bad grammar. **This is an improvement in measured
prediction, not a fluent or instruction-following model.** It learned Shakespeare
dialogue, not answers to arbitrary questions.

## Runtime changes

- Embedding lookup and classifier kernels support up to 512 vocabulary entries.
  Large rows run individually; embeddings stream one or two feature columns at
  a time to stay inside 4 KB of shared memory.
- Counted loops keep programs inside the 1,024-instruction ROM. Large address
  increments are split to fit the ISA's signed 14-bit immediate.
- Cross-entropy's logarithm constants cover exponential sums up to 512. Gradient
  accumulation and SGD still execute through the existing SM instructions.
- Cache identity includes tokenizer metadata. Generated IDs are appended without
  retokenizing the text, including when caching is disabled.
- Terminal output distinguishes tokens from characters and reports both rates.
  Existing character checkpoints keep their previous behavior.

No RTL arithmetic, numerical tolerance, or hardware memory capacity was changed.
The host moves data, encodes text, and verifies results. RTL performs embeddings,
the complete forward pass, probability calculation and token selection. In RTL
training, it also computes the loss, all gradients, and updates.

The tokenizer format validates merge order, token composition and vocabulary
identity. Tests cover malformed checkpoints, 512-entry embeddings/classifiers,
partial lane masks, full-model gradients against PyTorch, checkpoint resume, and
cached versus full recomputation across a rebuild.

## RTL generation measurement

Both runs used `My lord, I have something to say`, seed 92617660, temperature
0.7, Verilator, no cycle trace, and all numerical checks enabled. They ran
sequentially on the same Mac, stopping at the first whole token reaching at least
96 new characters. The RTL fingerprint is `592ce28d04d3...`.

| Measurement | Character default | Subword candidate |
| --- | ---: | ---: |
| Generated characters / tokens | 96 / 96 | 99 / 43 |
| First prediction | 5.12 s | 7.34 s |
| Cached prediction | 0.71 s/character | 2.53 s/token |
| Cached text throughput | 1.40 characters/s | 0.92 characters/s |
| Overall text throughput | 0.90 characters/s | 0.87 characters/s |
| Context rebuilds | 11 | 0 |

The larger model is slower per cached character. Its longer effective context
avoids the rebuilds in this particular run, bringing total throughput close to
the default. This is not a general speedup or an FPGA measurement. Both complete
RTL continuations exactly matched CPU diagnostic generation. Full outputs and
individual step timings are retained in `models/subword_transformer/benchmark.json`.

## Completed verification

`models/subword_transformer/verification.json` records the final checks:

- 59 distinct focused test methods passed across the regression and interface
  runs, including ten methods for larger vocabularies and subword models.
- Two complete 64-token validation windows passed against the independent
  double-precision reference and exported PyTorch probabilities. Maximum
  absolute probability error against PyTorch was 0.00000334.
- A rebuild to 48 tokens and the next cached prediction at 49 tokens were
  bit-identical to full RTL recomputation.
- One full training step on the actual 669,952-parameter candidate passed every
  forward, loss, gradient and SGD-update check. All 70 parameter-gradient tensors
  also passed comparison with PyTorch autograd. The RTL loss was 2.18000555,
  versus 2.18000625 in PyTorch.

That step executed 231,374,367 cycles and 37,791 launches. It took 170.2 seconds
while other numerical checks ran concurrently, so this is not an isolated training
benchmark. Its updated checkpoint remains under the experiment's
`rtl-training-step/` directory; it was not substituted for the validated inference
candidate. The candidate's 16,000 pretraining updates were on the CPU. This single
RTL step establishes that its complete training path works, not that one update
improves language quality.

The normal `python3 scripts/run_model.py --stream` path was also checked with the
installed checkpoint, without PyTorch. It correctly reports subword tokens and
character throughput. Lint and formatting checks passed.

## Reproduce the training recipe

The production host trainer supports the same tokenization and recipe:

```sh
build/model-venv/bin/python scripts/train_reference_model.py \
  --bpe-vocabulary 512 --width 128 --context 64 --layers 4 --heads 4 \
  --steps 16000 --batch-size 32 --seed 2631 --initialization gpt \
  --dropout 0.15 --weight-decay 0.1 \
  --learning-rate 0.0015 --min-learning-rate 0.00015 \
  --eval-every 1000 --evaluation-targets 2048
```

This recreates the recipe, not a byte-identical export of the earlier prototype:
the production trainer includes its own provenance and export checks. It writes
to a new experiment directory and never overwrites the installed model.

`scripts/compare_text_models.py` compares character and subword checkpoints on
shared passages. Its default split is validation; use `--split test` after
selection. `scripts/compare_models.py` remains a character-only comparison and
rejects subword checkpoints instead of assigning incorrect character IDs.
