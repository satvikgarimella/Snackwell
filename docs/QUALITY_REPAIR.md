# Language-quality repair, October 26–28

## October 28: a compact, bounded language target

After the pretrained candidates also produced grammar and context errors, the
user chose a compact model and a simpler text-generation target. A separate
[sentence demo](SENTENCE_DEMO.md) trains a small transformer to answer location
questions from two facts. Its complete evaluation records correct answers and
remaining failures. This does not change the open-ended Shakespeare model's
quality or default checkpoint. Host training and RTL inference are reported
separately.

## October 27: sentence quality remains unresolved

The subsequent [decoding repair](DECODING_REPAIR.md) fixes subword prompt-boundary
handling and adds an optional, RTL-verified probability filter. Neither the
sampling comparisons nor those fixes resolve the models' grammar defects.
Checkpoint selection and the default model remain unchanged.
To reproduce the older generation tables with current tools, use
`--no-token-healing --min-p 0`.

No new checkpoint was promoted. More training, wider representations, additional
depth and a prose supplement improved some likelihood measurements, but none of
these trials produced consistently grammatical continuations. The installed
character-model default and the optional model below remain unchanged.

All results use the same 8,192 validation token targets and 16 text passages.
The reserved test split was not used again. Each primary generation comparison
contains eight fixed prompts, greedy decoding and three sampled seeds at
temperature 0.7, generating at least 192 characters. The wider models also have
20 additional cases from five previously used prompts. Every continuation was
reviewed and retained in the [evidence directory](measurements/quality-repair-20261027).

| Candidate | Parameters | Token loss | Token accuracy | Passage nats/character |
| --- | ---: | ---: | ---: | ---: |
| Previous optional 8-block, width 128 | 1,734,400 | 2.576525 | 40.26% | 1.160392 |
| Continued 8-block, width 128 | 1,734,400 | 2.498692 | 42.80% | 1.125519 |
| 4-block, width 256 | 3,454,976 | 2.432561 | 43.62% | 1.092264 |
| 8-block, width 256 | 6,614,016 | 2.437366 | 44.01% | 1.091493 |
| 4-block, width 256, added prose | 3,454,976 | 2.414269 | 44.19% | 1.082828 |

The continued model received 24,000 additional host AdamW updates. Its samples
still include `Sir, I will be a hundred home with thee.` The wider four-block
model trained for 24,000 updates from a fresh initialization. Spelling improved,
but it still generated `I cannot give it it.` and repeated dialogue exchanges.

The deeper model copied those four blocks and added four initially inert residual
blocks. Its starting FP32 logits matched the parent at 1, 16 and 128 tokens.
After 16,000 updates, its shared token loss was slightly worse than its parent's,
and its samples still contained broken clauses and invented words. It was
rejected after all 52 continuations were reviewed. Its remaining CPU scalar
export checks were intentionally interrupted after rejection; it has no new RTL
verification claim.

The prose trial continued the wider four-block model for 12,000 updates using
the Shakespeare training corpus plus 5,448,924 characters from ten public-domain
books. The [source manifest](measurements/quality-repair-20261027/prose-manifest.json)
records the URLs and hashes. Short dialogue was preserved during cleaning; no
normalized 32-word window overlapped the original held-out text. The final weights
passed the trainer's independent scalar export checks. However, the samples mix
play dialogue and prose, still break grammar, and include a greedy continuation
that repeats `to be able` through the output. The lower loss does not justify
releasing this as a quality fix.

The generation diagnostic now accepts `--characters 192` to compare character
and subword models over comparable amounts of text. It finishes the last emitted
token. All 32 outputs from the previous optional model were reproduced exactly.

Runtime support now permits 256 model features and 1,024 feed-forward features.
Matrix products remain tiled; GELU operates on bounded slices. Forward LayerNorm
uses one 256-feature row per launch. Its backward kernel finishes the reductions
before replacing each consumed input/gradient pair with the output gradients.
The 4 KB shared-memory and 1,024-instruction limits remain enforced.

The trained four-block width-256 model passed a full 128-token RTL forward pass.
Its probabilities differed from the independent float64 PyTorch calculation by
at most 3.48e-6. Rebuilding a 96-token cache and decoding the 97th token matched
fresh RTL computations bit for bit. The [audit](measurements/quality-repair-20261027/rtl-wide4-window.json)
records the checkpoint and source hashes. It is a numerical check, not a language
quality result or an isolated latency benchmark.

The two width-256 regression methods check independent gradients for a
793,861-parameter fixture, exact FP32 SGD updates, cache behavior, original input
preservation, and Icarus/Verilator normalization parity including zero variance.
The four existing width-128 tests and 16 focused training/generation tests passed;
all 36 existing assembly examples remained unchanged. No tolerances were relaxed.

The [experiment record](measurements/quality-repair-20261027/continued-experiment.json)
contains the exact recipes, model hashes and decisions. These trials trained on
the Mac with PyTorch; they are not reported as full Snackwell RTL training runs.

### Pretrained feasibility comparison

Two published models were evaluated separately on the CPU after these trials
failed to resolve grammar. Neither is installed in Snackwell. Both used the same
13 prompts, greedy decoding and three sampling seeds at temperature 0.7, with a
128-token context and a rebuild retaining 96 tokens. A continuation stops after
192 characters, the end token, or 256 generated tokens. Consequently, cases that
emit the end token are shorter than the Shakespeare comparisons.

- [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M), revision
  `77f1b168e219585646439073245fe87e56b3023e`, has 3,745,984 total parameters
  including embeddings. Its [52 cases](measurements/quality-repair-20261027/pretrained-tinystories-1m.json)
  have clearer wording but substantial repetition and semantic errors.
- [TinyStories-8M](https://huggingface.co/roneneldan/TinyStories-8M), revision
  `8612e3b15c66ffa94eaa6ee0de5c96edd2d630af`, has 19,702,528 total parameters
  including embeddings. Its [52 cases](measurements/quality-repair-20261027/pretrained-tinystories-8m.json)
  more often form grammatical sentences, but still repeat, contradict themselves
  and mishandle some prompts. It generates simple stories rather than Shakespeare.

The model names are not their total parameter counts. This is a qualitative
feasibility comparison, not evidence of Snackwell execution or our own training.
The eight-block width-256 model has 16 heads and a 50,257-token vocabulary;
Snackwell currently supports four heads and 512 vocabulary entries. The tokenizer,
attention semantics and output projection would need explicit integration and
numerical verification. Its RTL latency has not been measured. The choice to
change the demo's subject and use pretrained weights remains open.

A separate [CPU input-boundary check](measurements/quality-repair-20261027/trailing-space-diagnostic.json)
shows that the trailing space in `My name is ` becomes its own BPE token. For the
pretrained model, omitting that trailing space changes the leading predictions
from `____` and `and` to names including ` Tom` and ` Bob`. Integration should
handle this token boundary while preserving the user's text. This does not
explain or fix the broader grammar failures in the Shakespeare trials; no
production tokenization behavior was changed during this diagnostic.

## October 26 optional checkpoint and historical comparison

**Follow-up:** neither the eight-block model nor the later character-model trial
solves sentence quality. The October 27 trial described below was rejected.
The separately verified [transport improvement](INFERENCE_TRANSPORT.md) reduces
simulation overhead without changing any model's output.

This experiment improves the Shakespeare continuation model without changing the
SM arithmetic. Model training uses PyTorch on the Mac's GPU. Separate RTL runs
check exported weights, cache behavior, backpropagation, SGD updates and checkpoint
reloads. Host training is not reported as Snackwell training.

The comparison starts from the installed subword checkpoint,
`models/subword_transformer/checkpoint.json`, SHA256
`964d603faa39de1b90c0bd31e2a5e0e66402d5cd10e2f1fc61ae183431cb7ba8`.
It has four blocks, four heads, width 128, feed-forward width 256, context 64,
a 512-token vocabulary and 669,952 parameters. The faster character-model default
is a different checkpoint.

## Training data

The original TinyShakespeare split, its tokenizer and its held-out text are
unchanged. The new training supplement contains 29 additional plays and 3,606,389
characters from [Project Gutenberg's Shakespeare collection](https://www.gutenberg.org/ebooks/100).
Nine plays were excluded because they overlap the base corpus or contain its
held-out text. The entire *Taming of the Shrew* is excluded, including passages
whose spelling or punctuation differs between editions. A normalized 32-word
window comparison found no overlap between the supplement and held-out text.

`scripts/prepare_shakespeare.py` removes contents, cast lists and stage directions,
converts speaker headings to the original corpus's format, and normalizes
punctuation and ligatures. It records included/excluded plays and file hashes.
The source file's SHA256 is
`eb61e33cab23dbe712ad57091ded6c7292ba91ac3bb2764f51a3223ec0068fe5`;
the prepared supplement's is
`9496a2f95175a9aa6e5a7514eb2233ac516b1610fc9b1a96374c474ff5c71401`.

The trainer's repeatable `--extra-training-data` option appends data only to the
training split. It does not refit the tokenizer or alter validation/test inputs.
The checkpoint records each supplemental file's hash and token count. This run
uses 2,185,821 training tokens, including separators.

## Selection protocol

The shared validation comparison uses 8,192 identical token targets, sampled
with seed 2718. Each model uses its own context ending before the same target.
Sixteen identical passages, totaling 6,533 target characters, are also scored
with the deployed periodic context-rebuild policy. Nats per character is the
canonical token sequence's negative log likelihood divided by characters;
it is not character accuracy.

The generation check retains every continuation from eight prompts, greedy
decoding, and sampling at temperature 0.7 with seeds 92617660, 1729 and 1730:
32 cases per model, each at least 192 new characters. All cases are inspected,
including broken sentences. Word coverage uses a common training-only lexicon
for every candidate. It is a spelling diagnostic, not a fluency score.

The old repetition diagnostic only detected triple repetitions with periods up
to 12 characters. It missed repeated sentences and speaker turns. That cap is
removed, with a regression test, and the saved baseline and candidate texts are
rescored. The generated text itself is unchanged. Reports now also expose the
longest repeated substring so a flag can be inspected: `III` in a speaker's
name is three copies of `I`, but it is not a generation loop.

Loss differences use a paired bootstrap over adjacent 128-token corpus blocks,
with 5,000 replicates and seed 20261021. These intervals describe the fixed
evaluation sample; they do not establish general language ability. Selection
uses validation and complete samples before any new reserved-test scoring.

## Trials

The eight-block model has width 128, four heads, feed-forward width 512,
context 128 and 1,734,400 parameters. Training it for 20,000 updates on only the
original small corpus overfit: training loss continued to fall while validation
stalled. Its selected trainer validation loss was 2.9015, compared with 2.8963
for the baseline on those same targets. The selected weights were from step
12,000. That checkpoint was not promoted as an inference model.

Adding the prepared supplement helped both four- and eight-block models. Both
use 8,000 additional AdamW updates, batch size 32, seed 2631, dropout 0.15, matrix
weight decay 0.1, and a learning rate that warms up for 200 updates before
decaying from 0.0003 to 0.00003. The optimizer starts fresh. Existing weights are
resumed; checkpoint metadata preserves their training history.

The four-block model has 941,312 parameters and starts from the previous
expanded-feed-forward experiment. Its shared validation token loss falls from
the baseline's 2.833217 to 2.682467, and passage nats per character from 1.280377
to 1.216728. That improves prediction, but its greedy continuations still loop
and sampled sentences still have grammar errors.

A smaller four-block model with feed-forward width 256 was also distilled from
that candidate. It gives only a 1.27% shared-token-loss reduction and a 0.31%
passage-score reduction, with lower token accuracy. Its complete samples still
have broken grammar. It was not selected as a quality fix.

## Selected candidate

The eight-block model trained on the broader corpus is selected for the optional
checkpoint at `models/shakespeare_transformer/checkpoint.json`, SHA256
`28e378b741faa39fbd74f42d683ee99a91efc29f15b71b6742c04a32ddc22971`.
It improves the fixed validation measurements:

| Measurement | Installed subword baseline | Selected eight-block model |
| --- | ---: | ---: |
| Token loss, same 8,192 targets | 2.833217 | 2.576525 |
| Next-token accuracy | 37.60% | 40.26% |
| Passage nats per character | 1.280377 | 1.160392 |
| Sampled cases with raw triple-repeat flags | 2 / 24 | 0 / 24 |
| Greedy cases with raw triple-repeat flags | 3 / 8 | 4 / 8 |
| Sampled words absent from common training lexicon | 14 / 850 | 7 / 857 |

Token loss falls 9.06%, and passage nats per character falls 9.37%. The paired
token-loss-change interval is [-0.291427, -0.223974]. The baseline's two sampled
flags were both `III` in a character's name. They do not establish repetition in
the baseline or a reduction in repetition from the new weights. The baseline's
greedy flags were also single-character repeats; the candidate has four genuine
sentence loops in greedy decoding, including repeated `It is a most pretty times
of a clock.` after `ROMEO:`. Greedy sentence repetition has therefore worsened in
this fixed set. This is one reason the candidate remains optional.

Sampled output still has grammatical and semantic errors. One fixed continuation
begins `My lord, I have something to say,` followed by `And then I know how I would
laugh against myself.` Another includes the invented word `pilgrimence`.
The complete before/after samples are retained in the model's `evidence/`
directory. This is a better predictive model, not a fluent assistant or a finished
solution to sentence quality. The existing faster default is preserved.

After selecting the candidate, the reserved test split was evaluated once for
this comparison. It uses another 8,192 shared token targets and 16 shared text
passages. No parameters or sampling choices were tuned using these results.

| Reserved-test measurement | Baseline | Selected model |
| --- | ---: | ---: |
| Token loss | 3.144733 | 2.726549 |
| Next-token accuracy | 33.52% | 38.57% |
| Passage nats per character | 1.454787 | 1.294597 |

The test token-loss reduction is 13.30%; the paired change interval is
[-0.471820, -0.369300]. Passage nats per character falls 11.01%.

## Reproduce the training recipe

Use the optional PyTorch environment. `--device mps` selects the Mac's GPU;
`--device cpu` uses the CPU. Neither mode is Snackwell execution. The downloaded
Gutenberg source must match the source hash above to reproduce this exact corpus.
Output directories must be new so previous checkpoints are not overwritten.

```sh
mkdir -p build/quality-reproduction
curl -L https://www.gutenberg.org/ebooks/100.txt.utf-8 \
  -o build/quality-reproduction/gutenberg-100.txt

build/model-venv/bin/python scripts/train_reference_model.py \
  --download --bpe-vocabulary 512 --width 128 --context 128 \
  --layers 8 --heads 4 --ffn-multiplier 4 --initialization gpt \
  --steps 20000 --batch-size 32 --seed 2631 \
  --dropout 0.2 --weight-decay 0.1 \
  --learning-rate 0.001 --min-learning-rate 0.0001 --warmup-steps 500 \
  --eval-every 1000 --evaluation-targets 2048 --device mps \
  --out build/quality-reproduction/deep8

python3 scripts/prepare_shakespeare.py \
  --base build/data/tinyshakespeare.txt \
  --gutenberg build/quality-reproduction/gutenberg-100.txt \
  --out build/quality-reproduction/supplement.txt

build/model-venv/bin/python scripts/train_reference_model.py \
  --resume build/quality-reproduction/deep8/checkpoint.json \
  --extra-training-data build/quality-reproduction/supplement.txt \
  --width 128 --context 128 --layers 8 --heads 4 --ffn-multiplier 4 \
  --steps 8000 --batch-size 32 --seed 2631 --dropout 0.15 --weight-decay 0.1 \
  --learning-rate 0.0003 --min-learning-rate 0.00003 --warmup-steps 200 \
  --eval-every 1000 --evaluation-targets 2048 --device mps \
  --out build/quality-reproduction/dialogue8
```

The installed checkpoint preserves the exact run. A retraining run can differ
with PyTorch, device kernels or runtime versions, even with the same seed.
The recorded run used PyTorch 2.14.1.

Reproduce the held-out passage comparison with the ordinary evaluation tool:

```sh
build/model-venv/bin/python scripts/compare_text_models.py \
  --checkpoints models/subword_transformer/checkpoint.json \
    models/shakespeare_transformer/checkpoint.json \
  --data build/data/tinyshakespeare.txt --split test --spans 16 --seed 2741 \
  --out build/quality-reproduction/passage-test.json
```

## RTL verification

The selected checkpoint passed the complete audit with RTL fingerprint
`592ce28d04d34a417c725311f338113b644d6b895cfe25296d951a7cec546c63`.

- Full 128-token forward inference passed the independent scalar reference and
  PyTorch float64 comparison. Maximum absolute probability error against PyTorch
  was 2.30e-6, within the unchanged tolerance.
- The 96-token cache rebuild and subsequent 97-token cached prediction were
  bit-identical to full RTL recomputation.
- A full 128-token training step checked all 1,734,400 parameter gradients in
  134 tensors against independent float64 autograd. Maximum absolute gradient
  error was 7.82e-7; existing gradient tolerances remained rtol 0.002, atol 2e-5.
- All 1,734,400 updated FP32 words matched independent multiplication followed
  by subtraction using the RTL gradients. The learning rate was 0.0001.
- Checkpoint save/reload preserved all weights and the tokenizer exactly.
  Inference from the reloaded updated checkpoint passed.
- Independent same-batch loss fell from 2.416223 to 2.414904 after that RTL
  update. This is a correctness/learning check on one batch, not an additional
  held-out language-quality result.

That training step executed 1,296,881,759 simulated cycles in 205,425 launches.
Its wall time overlapped other checks and is not an isolated throughput result.
The separate four-block audit also passed all 941,312 gradients and updates;
its evidence is retained alongside the selected model's results.

The 36-method focused suite passed with no skips. Coverage includes eight-block
gradients, cache/rebuild equivalence, supplemental-data split preservation,
checkpoint behavior, corpus extraction and long repetition detection. A follow-up
diagnostic test makes the Roman-numeral pattern explicit. No new full
RTL-bench-suite run is claimed; the RTL implementation itself was unchanged.
All four repetition-diagnostic tests passed after that change, bringing coverage
to 37 distinct passing methods. Lint and formatting passed for all 12 added or
changed Python files.

## RTL timing and terminal check

Both models generated 32 subword tokens from `My lord, I have something to say`,
using seed 92617660 and temperature 0.7. Runs were sequential, after the training,
evaluation and simulation audits had stopped. Compilation was excluded; numerical
verification remained enabled. These are host simulation timings, not FPGA timings.

| Measurement | Baseline | Selected model |
| --- | ---: | ---: |
| First token / prompt processing | 7.59 s | 22.26 s |
| Cached decode, mean of next 31 tokens | 2.55 s/token | 6.28 s/token |
| Total for 32 new tokens | 86.70 s | 216.79 s |
| Generated characters | 78 | 71 |
| Simulated cycles | 106,285,823 | 279,083,120 |
| Kernel launches | 32,258 | 79,354 |

Both complete RTL continuations matched their respective CPU previews exactly.
No context rebuild occurred in either timing run. This is one same-prompt
comparison, not a broad throughput study. The new model costs about 2.46 times
as much wall time per cached token. It is offered as an optional likelihood
improvement, not a speed improvement or a fluent-text replacement.

The ordinary interactive terminal command was also checked with the installed
checkpoint, the same prompt/seed/temperature, and one new token. It generated
the expected comma, reported `RTL/reference match`, and exited normally on
`/quit`. Its log is retained in `evidence/terminal-smoke.log`.

## Audit commands

The model includes `evidence/audit.py`, the experiment's audit driver with paths
adjusted for its saved location. It uses the normal simulator and runtime;
reference activations or gradients are never supplied as execution results.
The optional PyTorch environment supplies a second, independent autograd check.

```sh
build/model-venv/bin/python models/shakespeare_transformer/evidence/audit.py \
  --checkpoint models/shakespeare_transformer/checkpoint.json \
  --out build/quality-reproduction/rtl-window --mode window

build/model-venv/bin/python models/shakespeare_transformer/evidence/audit.py \
  --checkpoint models/shakespeare_transformer/checkpoint.json \
  --out build/quality-reproduction/rtl-training --mode training

build/model-venv/bin/python models/shakespeare_transformer/evidence/audit.py \
  --checkpoint models/shakespeare_transformer/checkpoint.json \
  --out build/quality-reproduction/rtl-timing --mode benchmark
```

`window` checks the full 128-token context, a cache rebuild at 96 tokens and
the subsequent cached prediction at 97 tokens. `training` checks a complete
128-token SGD step, every parameter gradient against float64 PyTorch autograd,
every FP32 update against separate multiplication/subtraction, exact checkpoint
reload and inference after reload. The audit's updated weights are saved only
under its output directory; the installed quality checkpoint is unchanged.

`benchmark` runs the baseline and candidate sequentially with the same prompt,
seed, temperature and 32 generated tokens. It checks the entire generated text
against the CPU preview and records per-token timings, cycles and launches.
Run this after other training and simulation jobs finish to avoid contention.
All three modes retain the existing numerical tolerances.

Raw experiment files, training recipes, full samples and RTL launch artifacts
are retained locally under `build/quality-repair-20261026/`. Arithmetic tolerances,
the ISA, shared-memory capacity and RTL implementation remain unchanged.

## October 27: longer-context character model rejected

A separate trial trained a four-block, four-head character transformer with width
128, feed-forward width 512 and context 128. It has 826,433 parameters. This
tests a longer character context than the default's 32 characters while using
fewer blocks than the eight-block BPE model. It uses the same train-only supplement
and unchanged TinyShakespeare splits. Host MPS training ran 20,000 updates with
batch size 32, seed 2719, dropout 0.15, matrix weight decay 0.1, GPT initialization,
500 warmup steps and cosine learning-rate decay from 0.001 to 0.0001.

Validation selected step 18,000. On the same 16 passages and 6,533 characters used
above, the default character model scores 1.372564 nats/character and this candidate
scores 1.274935, a 7.11% reduction. The earlier eight-block BPE model remains
better on that measure at 1.160392. No new reserved-test evaluation was performed.

All eight prompts were inspected with greedy decoding and three sampled seeds
(92617660, 1729 and 1730), temperature 0.7, with 192 generated characters per case.
The candidate's greedy outputs loop through slight variations of “the strength of
the state.” Sampled outputs still have broken grammar, for example
“I will not with a man court in my talk.” Exact triple-repeat counts miss many
of these near-repeated phrases; the actual continuations determine the decision.

The candidate was **rejected**. Its checkpoint remains in
`build/quality-repair-20261027/character4/`, with its recipe, export checks, complete
samples and validation results alongside it. It was not installed as a default
or packaged as another recommended model. No expensive RTL model audit was run
after its CPU text samples failed the quality requirement. The existing model
files are unchanged.
