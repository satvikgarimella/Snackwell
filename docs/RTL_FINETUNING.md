# Continuing the character model's training on RTL

This experiment continues the existing 144,449-parameter character model for
100 SGD updates on Snackwell's simulated SM. The original checkpoint was trained
on the host with PyTorch; its metadata records selection at step 19,500 from a
20,000-step AdamW run with batches of 64 sequences. These additional updates run
forward propagation, cross-entropy, backpropagation and every parameter update
on RTL.

This is a controlled fine-tuning experiment, not training a language model from
random weights. It is separate from the 4,657-parameter demonstration described
in [the training guide](TRANSFORMER_TRAINING.md).

## Fixed setup

- Four blocks, four heads, width 64, context 32, vocabulary 65.
- Starting checkpoint: `models/character_transformer/checkpoint.json`, SHA256
  `43f211daad7c733d69c34f1d2588dace14de4504a46cf785ec64e6722053622e`.
- 100 updates, one 32-character sequence per update, SGD learning rate 0.001,
  seed 1729. This processes 3,200 training targets, with possible overlap among
  sampled sequences.
- TinyShakespeare uses a contiguous 90% training, 5% validation and 5% test split.
  This experiment does not consult the test split.
- All existing numerical tolerances and memory-write checks remain enabled.
- The default inference weights remain available throughout the experiment.

The simulation host supplies raw data, layout transformations and independent
checks. It does not substitute reference activations, gradients or updated
weights for RTL results. Training recomputes the full sequence after each weight
update; it does not reuse an inference KV cache across updates.

## Before-and-after comparisons

The training command checks 16 fixed windows from each split before and after
training. Each split therefore contributes 512 target characters. Both the
forward calculation and loss reduction run on RTL. This loss includes every
position in each sequence, including positions with little preceding context.

A separate CPU-reference quality diagnostic scores 8,192 identical validation
positions with seed 2718, using a full 32-character context before each target.
It saves each target's position, loss and prediction correctness. This score is
not directly interchangeable with the RTL window average. Its purpose is to
measure language quality on a larger sample, not hardware throughput.

Generation uses eight fixed prompts, greedy decoding and three sampled seeds
(92617660, 1729 and 1730), with 64 generated characters and temperature 0.7. All
32 cases are retained, including malformed words and repetitions. Corpus word
coverage is a spelling diagnostic, not a measure of fluency. Selected complete
continuations are also run through RTL after training.

## Reproduction and evidence

From the repository root:

```sh
python3 scripts/train_transformer.py \
  --checkpoint models/character_transformer/checkpoint.json \
  --steps 100 --evaluation-windows 16 --log-every 1
```

The current experiment is under
`build/transformer-training/large-model-100-20261020/`. `experiment.json` records
the comparisons before training results are available; `run.json` records the
dataset and source hashes. The runner preserves starting/final checkpoints,
per-step loss, all checked tensors and launch artifacts.

## Quality measurements

All 100 RTL updates passed. The run, including its before-and-after RTL
evaluations, took 2,731.9 seconds (45.5 minutes), executed 3,805,221,092 simulated
cycles and performed 684,244 kernel launches. Individual updates averaged
22.11 seconds. The final inference checks ran concurrently with the final
evaluation, so these times are not isolated performance benchmarks.

The fixed RTL window evaluations produced:

| Measurement | Before | After |
| --- | ---: | ---: |
| Training loss, 512 targets | 1.363018 | 1.336123 |
| Validation loss, 512 targets | 1.523647 | 1.524170 |

The larger CPU-reference comparison uses exactly the same 8,192 validation
targets for every checkpoint:

| RTL updates | Next-character loss | Next-character accuracy |
| ---: | ---: | ---: |
| 0 | 1.449532 | 56.4331% |
| 25 | 1.449547 | 56.4941% |
| 50 | 1.449621 | 56.4453% |
| 75 | 1.450386 | 56.4453% |
| 100 | 1.450448 | 56.5186% |

Loss worsened slightly, while the final model predicted seven additional targets
correctly out of 8,192. These mixed changes do not establish a useful language
quality improvement. The default inference checkpoint is preserved.

The paired mean loss change was +0.000916. A diagnostic block bootstrap groups
nearby targets into 256-character corpus blocks and resamples those blocks 5,000
times with seed 20261020. Its 95% interval for the change was
[-0.000231, +0.002115]. It includes zero; this sample does not establish a clear
generalization gain or a clear regression. The procedure and raw per-target
scores are saved alongside the run.

Thirty of the 32 fixed continuations changed. In the sampled cases, completed-word
coverage changed from 258/266 to 257/263; neither model produced a threefold repeat
in these short samples. That spelling diagnostic does not capture the remaining
grammar problems.

Two 64-character continuations from the final checkpoint were also executed on
RTL. Both passed all numerical checks and exactly matched their CPU diagnostic
continuations. With seed 92617660 and temperature 0.7, the results were:

```text
ROMEO:
And would not present but the repetition
In his head and sorrow
```

```text
My lord, I have something to say 'twas. There's send the true is not my ignorant make his miscip
```

Each stops at the requested 64 new characters. The second example still contains
clear grammar problems before the cutoff. A change in sampled wording is not
evidence of improved fluency.

The final checkpoint SHA256 is
`792b910b2aaafe2ea96ca4507000f5e1a973f1cc09c3c4ba37d746bd76399b03`.
It remains an experimental candidate under the run directory. Numerical checks,
the original checkpoint and Python execution sources were unchanged throughout
the experiment.

## Outcome and remaining work

The experiment completed successfully as an RTL training verification run. It
did not demonstrate improved language quality. The original checkpoint remains
the default, and the candidate is kept for inspection and exact continuation.
No arithmetic implementation or numerical tolerance was changed for this run.

`comparison-summary.json` contains the measurements, validation trajectory,
generation summaries and checked RTL continuations. `decision.json` records why
the candidate was not promoted. The final weights are in `checkpoint.json`;
the snapshots at updates 25, 50 and 75 are also retained. The reserved test split
was not used to tune or select this candidate.

Before committing to a much longer simulator run for language quality, compare
training recipes in the CPU reference with larger batches and suitable optimizer
settings. Any promising recipe still needs its arithmetic implemented and
verified on RTL. This experiment does not establish that simply running more
single-sequence SGD updates will fix the generated sentences.

## Follow-up: training and model selection

The subsequent repair experiments are in `build/language-repair-20261020/`.
These use **CPU reference training**, not RTL training. They test training recipes
before spending another long simulator run on an unpromising checkpoint.

The host trainer now selects weights using fixed next-character targets, each
with a full context. The previous selector averaged all positions in sampled
windows, including the first character with almost no preceding context. Both
measurements are legitimate, but they answer different questions. The new score
more closely matches generation after prefill. Validation selects weights; the
reserved test split is only evaluated when `--report-test` is explicitly passed.
Exported numerical-check examples otherwise come from validation.

Training writes to a new experiment directory under `build/reference-training/`
by default and refuses to overwrite an existing output directory. `run.json`
records arguments, input hashes and source hashes; `history.json` records progress.
An experiment that selects step zero explicitly says it found no improvement.
RTL training also reports whether its sampled validation loss improved and keeps
that separate from passing gradient/update verification.

`--loss-tail N` optionally emphasizes later predictions while retaining the whole
causal input. Its objective is 20% all-position loss plus 80% loss over the last
N targets, unless `--prefix-loss-weight` changes that mixture. It does not discard
the prefix or its indirect gradients. The default still trains on all positions.
This is an experimental host-training option, not a new RTL training objective.

The model format also accepts an optional `ffn_multiplier: 4`, retaining 2 as the
default for all existing checkpoints. Feed-forward width is limited to 256, the
existing bounded GELU path's supported width. With width 64 and four blocks,
this adds capacity from 144,449 to 210,497 parameters. The SM still has the same
four lanes and one 4×4 array; it executes the larger matrices in more pieces.
Forward execution, cached decoding, the independent references, and complete
RTL backpropagation all support the expanded shape.

For a resumed model, `--expand-ffn --ffn-multiplier 4` appends independently
initialized hidden units whose output weights start at zero. The original
function is preserved before training. The new output weights receive gradients
immediately; their input weights can learn after the output weights change.
Tests check preservation, trainability, every parameter gradient against PyTorch,
and exact cached-versus-recomputed RTL results across a context rebuild.

These capabilities do not establish a better language model. Checkpoint
promotion still requires a useful held-out quality improvement and verified RTL
execution. Failed candidates remain experiments; the existing inference model
is not silently replaced.

### Measured host comparisons

Each row below uses 8,192 identical validation targets for its before/after
comparison, seed 2718. The context-64 row has different target positions because
it reserves 64 preceding characters; compare within rows, not across rows.

| Experiment | Baseline loss | Candidate loss | Decision |
| --- | ---: | ---: | --- |
| 2,000 AdamW updates, emphasize last 8 positions | 1.449532 | 1.450722 | Reject |
| Matched 2,000-update all-position control | 1.449532 | 1.452787 | Reject |
| Fresh context-64 model, 16,000 updates, GPT initialization and dropout | 1.475543 | 1.583968 | Reject |
| Expanded feed-forward layers, matched 2,000-update recipe | 1.449532 | 1.452440 | Reject |

The first experiment looked slightly better on its 2,048-target pilot sample;
the larger check did not confirm that gain. The three short fine-tuning runs'
paired 256-character-block bootstrap intervals all include zero. The fresh
context-64 run is clearly worse on its matched targets. Another expanded
feed-forward run with dropout 0.05 and matrix weight decay 0.1 selected update
zero after 6,000 updates: none of its trained snapshots beat the starting model.

All eight fixed prompts, three sampling seeds, greedy outputs and 96-character
continuations are saved for the baseline and the tail/expanded candidates. The
samples still contain broken grammar. Slight changes in word coverage or
repetition are not enough to override the validation result. No reserved test
targets were used in these comparisons, and none of these weights was promoted.

`experiment.json`, the individual comparison JSON files and
`comparison-summary.json` retain the settings, per-target losses, checksums and
bootstrap intervals. The host reference trainer's seven tests, eleven complete
training tests, and model/cache/wide-model checks passed: 45 distinct focused
test methods in total. Two CLI tests were repeated after the final reporting
changes. No arithmetic tolerance or RTL source was changed in this follow-up.

### Depth and decoding experiments

A CPU prototype appended four identity-initialized residual blocks to the
width-64 model and trained for 8,000 updates. Its best snapshot, update 6,500,
reduced fixed validation loss from 1.449532 to 1.441594 (0.55%). The paired
diagnostic interval for the loss change was [-0.013727, -0.002261]. Longer samples
still contained broken grammar, and twice as many blocks would add substantial
work. This prototype was not deployed; eight layers remain unsupported by the
portable RTL checkpoint validator. Its source and results are retained under
`depth-eight/`, `grow_depth.py`, and `score_depth.py` in the experiment directory.

Temperatures 0.5 and 0.6 were also checked with the original weights using the
same eight prompts, three seeds and 96-character continuations. They reduced
misspellings in this diagnostic, but did not resolve broken grammar. Temperature
0.5 increased repeated four-grams. The default remains 0.7.

### A better-scoring wider candidate

Continuing the existing width-128, context-64, four-block model did improve
held-out prediction. This run used 4,000 additional **CPU AdamW** updates,
batch 32, seed 2301, dropout 0.1, matrix weight decay 0.1, gradient clipping at 1,
and cosine learning rates from 0.0001 to 0.00001. Fixed validation targets
selected update 4,000. The test split was scored only after this run's candidate
was selected; no training settings were subsequently tuned on those results.

Each split below uses 8,192 identical target positions across the three models.

| Checkpoint | Validation loss | Test loss | Test accuracy |
| --- | ---: | ---: | ---: |
| Current width-64 default | 1.475543 | 1.606291 | 53.8452% |
| Previous width-128 model | 1.421207 | 1.540052 | 54.5044% |
| Continued width-128 model | 1.398522 | 1.515285 | 55.5664% |

The candidate's test loss is 5.7% lower than the fast default's, and 1.6% lower
than the previous wider model's. The gain over the default combines extra model
capacity with additional training. It is not all attributable to this latest
training run. The paired test loss change versus the fast default is -0.091007,
with a diagnostic block-bootstrap interval of [-0.112119, -0.070694].

The candidate is saved at
`build/language-repair-20261020/wide-continued/checkpoint.json`, SHA256
`ad9a703f938b626340ea1872ec87476923b6eac7765765efcfa9f5fde2d4944a`.
It has 555,073 parameters. It was trained on the host; its subsequent **inference
runs on the SM RTL**. The fast default remains unchanged.

Both a 16-character `ROMEO:` continuation and a 64-character continuation of
`My lord, I have something to say` passed every numerical check and exactly
matched the CPU diagnostic, including two cache rebuilds in the longer run.
Two complete 64-character validation windows also passed, checking all 128
positions against exported PyTorch probabilities and the independent whole-model
reference. The full export contains eight windows; only the first two were run
in this check. The RTL source fingerprint remains `592ce28d04d3…`.

Sequential 16-character `ROMEO:` runs, seed 92617660 and temperature 0.7,
measured the cost on the same Mac with verification enabled and tracing off:

| Metric | Fast default | Continued wider candidate |
| --- | ---: | ---: |
| First character | 1.34 s | 4.31 s |
| Cached character | 0.69 s | 2.02 s |
| Complete 16-character run | 11.75 s | 34.59 s |

These are single RTL-simulation runs, not FPGA measurements. The longer candidate
run took 208.70 seconds for 64 new characters: cached steps averaged 2.30 seconds,
and its two context rebuilds averaged 26.06 seconds. Its actual output was:

```text
My lord, I have something to say amends.

LEONTES:
Most thou good are the gods of my soul of fel
```

The final word stops at the requested character limit, but the grammar problems
before that cutoff are real. **The sentence-quality problem is not solved.**
The candidate is retained for further experiments, rather than replacing the
faster default with a roughly three-times-slower model that still fails the
desired language-quality bar. `rtl-verification.json`, the per-run result files,
and `decision.json` record the completed checks and this decision.
