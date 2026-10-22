# Language-quality experiments, October 21

Training correctness and language quality are different measurements. The
[training audit](TRAINING_AUDIT.md) checked forward passes, losses, every gradient,
updates, and checkpoint reloads. These experiments measure whether different
weights make better predictions and continuations. They do not change the RTL
arithmetic or its numerical tolerances.

The baseline is the installed subword checkpoint with SHA256
`964d603faa39de1b90c0bd31e2a5e0e66402d5cd10e2f1fc61ae183431cb7ba8`:
four blocks, four heads, width 128, feed-forward width 256, context 64, vocabulary
512, and 669,952 parameters. It learned Shakespeare continuation, not instruction
following.

## Comparison protocol

Candidate training runs use CPU PyTorch AdamW. RTL execution is verified
separately; CPU-generated diagnostic text is never presented as an RTL run.
Experiments write to new directories and do not replace installed checkpoints.

Selection uses validation data only:

- 8,192 identical token targets, sampled with seed 2718. Each model sees its own
  context ending immediately before the same target. The saved tokenizer is
  identical across candidates.
- Sixteen identical validation passages, totaling 6,533 target characters,
  scored with the deployed periodic cache-rebuild policy. Nats per character
  means token-sequence negative log likelihood divided by target characters.
- Eight fixed prompts with greedy decoding and sampled decoding at temperature
  0.7, using seeds 92617660, 1729, and 1730. Each continuation reaches at least
  192 characters. All 32 cases are retained, including poor outputs.

The paired loss comparison resamples groups of adjacent targets in 128-token
corpus blocks, with 5,000 bootstrap replicates and seed 20261021. Its interval is
a diagnostic for this validation sample, not a guarantee about all language.
Word coverage and repeated substrings help locate problems; neither measures
fluency. Complete samples must also be read.

## Initial trials

Both trials start from the installed checkpoint, use 6,000 additional updates,
batch size 32, dropout 0.15, matrix weight decay 0.1, and seed 7210. The optimizer
starts fresh. Learning rates warm up for 100 updates and then decay with a cosine
schedule.

| Trial | Context | Learning rate, start to end | Validation token loss | Passage nats/character |
| --- | ---: | --- | ---: | ---: |
| Installed baseline | 64 | Original training | 2.833217 | 1.280377 |
| Continue training | 64 | 0.0002 to 0.00004 | 2.828441 | 1.272028 |
| Extend context | 128 | 0.0003 to 0.00005 | 2.827173 | 1.269104 |

The longer-context trial initially repeats the learned position table, then
trains those positions. Its parameter count is 678,144. On the matched targets,
the loss reductions are only 0.17% and 0.21%. Both paired diagnostic intervals
include zero. Their samples still have grammar errors. Neither was promoted.

An exploratory top-k sampling comparison at k=32, 16, and 8 also left grammar
problems; k=32 increased repetitive cases. No sampler or default was changed.

## Wider feed-forward trial

The next trial expands the longer-context model's feed-forward width from 256
to 512, for 941,312 parameters. Existing units are preserved; new units have
random input weights and zero output weights. Initial logits matched exactly in
a float64 check of four complete 128-token contexts. The new units can learn
without discarding the starting function.

This requires only extending the host model validation and launcher limits.
GELU already supports splitting independent elements into bounded launches.
Memory remains 4 KB and the instruction ROM remains 1,024 instructions.
Thirty focused tests passed, including width-512 gradients against PyTorch,
element ordering, memory and ROM bounds, full-versus-cached RTL inference,
checkpoint handling, and tokenizer behavior. Lint and formatting checks passed.

The trial completed 6,000 updates with the longer-context trial's recipe. Its
matched validation token loss was 2.809898 versus 2.833217 for the installed
baseline, a 0.82% reduction. The paired loss-change interval was
[-0.035677, -0.010199]. Passage nats per character fell from 1.280377 to 1.257953,
a 1.75% reduction. These are modest gains in prediction, not a fluency result.

All 32 complete continuations were inspected. Grammar and meaning still break
down. Among 24 sampled continuations, short triple repetitions occurred in three
cases versus two in the baseline. Words absent from the training corpus occurred
13 times among 855 completed words, versus 15 among 850 in the baseline. These
small diagnostic differences do not establish better prose. For example, the
standard prompt continued with `And then I know who have been not that yet done.`

The wider checkpoint remains an experiment under `build/`; it was not promoted
or committed as a replacement. Its full-checkpoint RTL forward/training checks
and latency benchmark have not run. The 30 passing support tests establish the
new shape's execution support, not verification of these particular trained
weights. No new reserved-test evaluation was used, no sampling default was
changed, and the installed baseline remains intact. Better generated language
is still unfinished.

## Evidence

Local evidence lives under `build/quality-improvement-20261021/`:
`experiment.json` records the recipes; `comparison-validation.json` records the
matched comparisons; `generation-*.json` retains complete generated samples;
`expansion-preservation.json` and `support-verification.json` record the wider
model checks. The experiment's evaluation scripts are retained there as well.
