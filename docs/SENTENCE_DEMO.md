# Compact sentence generation

This experiment targets a small, measurable language task: read two location
facts and generate a sentence answering a question about one object. It uses
Snackwell's existing transformer architecture and FP32 kernels. All predicted
tokens run through the SM RTL, with the existing independent kernel and
whole-model numerical checks.

Example input:

```text
The ball is in the kitchen. The book is in the garden. Where is the ball?
```

The required answer is `The ball is in the kitchen.` Changing the locations
must change the answer; changing the order of the facts must preserve it.

## Scope

The grammar is fixed. Supported objects are ball, book, cup, hat, key, toy, bag,
and box. Supported locations are kitchen, garden, bedroom, office, hallway,
bathroom, attic, and cellar. Inputs contain two distinct objects in distinct
locations, followed by `Where is the <object>?` with the capitalization and
punctuation above. The terminal rejects inputs outside this scope.

Input validation checks the format and vocabulary. It does not look up the
answer. The transformer generates every answer token using unfiltered greedy
selection in RTL. Generation stops at the first period or at the safety limit.
The host continues to transfer raw tensors between bounded shared-memory
launches; this is the existing simulation runtime, not a new FPGA memory path.

## Why this task

The user chose a compact model and a simpler text-generation target after the
pretrained candidates still produced grammar and context errors. The
[sentence screening results](measurements/pretrained-selection-20261028/sentences/decision.json)
retain those failures. The earlier paragraph screening remains in the same
measurement directory. These models were evaluated on the host and were not
installed as Snackwell defaults.

This is a bounded language demonstration. Its results do not establish story
writing, chat, unfamiliar-word handling, or reasoning outside this grammar.
Existing Shakespeare models remain separate.

## Data and training

`snackwell/sentence_task.py` constructs 3,552 two-fact training examples, 576
validation examples, and 576 reserved test examples. Splits reserve complete
object/location associations: `(object_index + location_index) % 8` equals zero
for test pairs, one for validation pairs, and anything else for training pairs.
Neither fact in a training example contains a reserved pair. Evaluation uses
a reserved query pair and a training-pair distractor, with the queried object
equally often first and second.

The tokenizer is fitted on training text only. Training supervises answer
tokens and masks the prompt and padding from the loss. The curriculum first
uses single-fact examples built only from training pairs, then mixes these with
the two-fact data. Location-token losses receive extra weight. Checkpoint
selection uses validation loss. Reserved test scores do not select weights.

Training runs in host PyTorch. It is separate from the project's verified RTL
training implementation. Export checks compare the saved model to the
independent scalar transformer reference. RTL inference subsequently verifies
each arithmetic stage and the complete forward pass.

## Evaluation

`scripts/evaluate_sentence_model.py` performs autonomous generation, feeding
the model's own emitted tokens back into the next step. It saves every answer,
including failures, and scores exact matches. It also checks all 6,272 possible
prompts in this finite grammar, including scenes where both facts contain pairs
excluded from training. Training examples are included in that exhaustive
count, so it is reported separately from reserved-test accuracy.

Its `--rtl` audit checks all eight reserved query pairs in both fact positions,
two examples with swapped locations, and a cached-versus-fresh computation.
The exhaustive host test is not described as 6,272 RTL executions.

## Selected model and measured quality

The selected model has four blocks, two attention heads, width 32, a 32-token
context, 97 vocabulary entries, and 41,569 FP32 parameters (166,276 raw weight
bytes). Its checkpoint SHA256 is
`66af3342b4a66ee4e63816f30756f3aed746b749b300ecd6f13c324be0d2423e`.
This is a separate checkpoint from the Shakespeare models.

Autonomous greedy generation on the host produced:

| Evaluation | Exact answers |
| --- | ---: |
| Validation questions | 576 / 576 |
| Reserved test questions | 576 / 576 |
| Entire supported prompt space | 6,244 / 6,272 (99.55%) |

The complete [host report](measurements/sentence-completion-20261028/host.json)
contains every generated answer. There are
[28 remaining wrong answers](measurements/sentence-completion-20261028/remaining-errors.json).
These are location-selection errors, so the demo is not error-free. For example,
it can assign the bag's location to the hat when both appear in the prompt.
The input validator does not hide these cases; they are accepted prompts in the
documented domain.

The selected checkpoint passed all 18
[RTL answer checks](measurements/sentence-completion-20261028/rtl.json). Cached
and fresh decoding produced identical final probabilities and token selection.
An additional [terminal run](measurements/sentence-completion-20261028/terminal.txt)
verified the corrected prompt:

```text
The toy is in the hallway. The ball is in the kitchen. Where is the ball?
Answer: The ball is in the kitchen.
```

That run executed 3,608,046 simulated cycles in 1,135 launches. On this Mac, with
Verilator already compiled and the training jobs finished, the seven-token
answer took 9.13 seconds including independent verification: 4.54 seconds to
the first token and about 0.76 seconds per cached token afterward. These are
simulation measurements, not FPGA timings. The
[per-token record](measurements/sentence-completion-20261028/terminal.json)
preserves the raw timings and checks.

The focused decode, vocabulary, sampling, dataset-isolation, and prompt-format
regressions passed. Ruff checks passed for `snackwell`, `tests`, and `scripts`.

The first four-block checkpoint scored 6,243 / 6,272. That exhaustive audit
motivated another training experiment using the original training data, with no
failing prompts added. Validation selected its first update; further updates did
not improve validation loss. The selected refinement corrected one exhaustive
case and introduced no new errors. Thus the full-domain results also served as
a regression audit during development; they are not an untouched estimate of
broader generalization. The reserved object/location pairs never enter gradient
training. Earlier experiments and results remain in the
[measurement directory](measurements/sentence-completion-20261028/).

## Run and reproduce

```sh
python3 scripts/run_sentence.py --interactive
```

Paste the example above, or change the objects, locations, fact order, and
question using the supported vocabulary. `/quit` exits. Each emitted token has
already passed the independent RTL checks. Run files include the checkpoint
hash, simulator fingerprint, per-token numerical checks, cycles, and latency.

The host training recipe, with fresh output directories:

```sh
build/model-venv/bin/python scripts/train_sentence_model.py \
  --out build/sentence-reproduction/base --layers 4 --curriculum \
  --steps 4000 --eval-every 250

build/model-venv/bin/python scripts/train_sentence_model.py \
  --out build/sentence-reproduction/refine \
  --resume build/sentence-reproduction/base/checkpoint.json \
  --layers 4 --curriculum --learning-rate 0.0005 --weight-decay 0.1 \
  --steps 4000 --eval-every 250
```

The refinement selects one additional update on top of the 4,000-update base.
Training uses a fresh AdamW optimizer when resuming. The measured recipes,
validation histories, checkpoint lineage, and source snapshots are retained;
metadata hashes can change when the training script changes.

Audit the installed checkpoint:

```sh
build/model-venv/bin/python scripts/evaluate_sentence_model.py \
  --checkpoint models/sentence_completion/checkpoint.json \
  --out build/sentence-evaluation/host

python3 scripts/evaluate_sentence_model.py \
  --checkpoint models/sentence_completion/checkpoint.json \
  --out build/sentence-evaluation/rtl --rtl
```

The host command reports exact-answer counts. The RTL command checks numerical
execution independently and also reports answer correctness; these are separate
claims. Numerical tolerances are unchanged.
