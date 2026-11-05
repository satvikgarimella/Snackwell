# Decoding repair, October 27

The subword input boundary is repaired, and an optional sampling filter now runs
on the SM. **Sentence quality remains unresolved.** No checkpoint, training data,
default model, temperature, or numerical tolerance was changed in this follow-up.
The filter is disabled by default.

## Prompt preservation

A prompt ending in a space can tokenize differently from the same text followed
by a word. Previously, generation always kept that standalone space token. Now,
if the last subword token can extend and has a preceding token, generation backs
up one token and restricts its replacement to tokens starting with that exact
text. This also handles partially typed words. The visible prompt is untouched;
only the replacement's new suffix is displayed. Selecting the original token
adds no visible text on that step.

The host prepares a packed vocabulary allow-list from text. A generated SM program
applies it to the final logits. Softmax and selection also run on the SM. The
full selected token enters the KV cache, and subsequent steps resume normal
decoding. Raw model logits, probabilities and greedy predictions are independently
checked before applying the constraint. Cancelled predictions retain the pending
prefix for a retry. Single-token prompts and character models are unchanged.

`--no-token-healing` reproduces the previous behavior. This is a token-boundary
correction, not a rule that inserts names or supplied answers.

## Optional probability filter

`--min-p 0.05` discards weights below `0.05 * max_probability`. The threshold is
computed in FP32 on the SM; equal weights at the threshold survive. The existing
categorical kernel normalizes by the surviving total when selecting an ID. No
sorting, CPU token selection, or reference-result substitution is used.

The option accepts 0 through 1; 0 disables it and greedy decoding ignores it.
The terminal runner, CPU diagnostic, quality evaluator and existing local browser
entry point accept the same setting. Reported `sampling_probabilities` retain the
normalized distribution before filtering. `sampling_weights` records the surviving
unnormalized weights; `selection` records the chosen token and any healed prefix.

## Quality checks

The [evidence directory](measurements/decoding-repair-20261027) contains complete
CPU continuations from 13 prompts, greedy decoding and three sampling seeds,
with at least 192 new characters per case (finishing the last token):

- `wide4-filter.json`: the earlier four-block width-256 Shakespeare candidate,
  temperature 0.7, min-p 0.05, prompt healing, no repetition penalty.
- `wide4-diverse.json`: the same model with temperature 0.85 and min-p 0.1.
- `wide4.json`: a rejected experiment additionally applying a 1.1 repetition
  penalty and an eight-token repetition ban. It broke recurring speaker names,
  including `MERCUNT` and `CLEOPABELLA`; those restrictions were not implemented
  in production.
- `character-filter.json`: the installed character default with temperature 0.7
  and min-p 0.05. Grammar errors persist, including `My name is doth the painted
  in his course forth.`

These are CPU sampling experiments, not RTL training or hardware timing results.
The probe used double-precision threshold multiplication; the production filter
uses a checked FP32 threshold. No conclusion about fluency is based on repetition
counts or a selected good opening.

With the installed optional eight-block model, seed 92617660, temperature 0.7,
min-p 0.05 and 24 predictions, the CPU comparison is:

```text
Before healing:
My name is imperious to this present particular.

PRINCESS:

After healing:
My name is bold and daughter.

PISTOL:
By my troth,
```

Neither output is a satisfactory answer to the prompt. The character default is
unaffected by the subword repair. The remaining failures cannot be described as
a fixed KV cache, a hardware arithmetic error, or a completed quality repair.

## Verification

Five new tests cover prompt text preservation, partial words, one-token prompts,
long prompts, padding, cancellation, unchanged replacement tokens, cached/fresh
RTL agreement, and CPU-reference agreement. Kernel tests compare Icarus and
Verilator bit for bit, including 512-entry masks crossing packed-word boundaries,
zero padding, threshold equality, tied maxima, and a probability immediately
below the threshold. Programs stay within 1,024 instructions and 4 KB of memory.

The 38 existing decoding, generation, terminal, KV-cache and subword tests passed,
including the existing subword gradient/update checks. Four local-browser tests
also passed after allowing their loopback-only test server. This is 47 focused
test methods, not a new full RTL regression sweep.

Two actual terminal runs completed 24 predictions each: the character default
with `ROMEO:` and the optional subword model with `My name is ` (including its
trailing space). Both matched the saved CPU prediction sequence exactly, and
every SM launch passed its independent checks. The subword RTL output is the
unsatisfactory healed example above. All 36 existing assembly examples also
remained unchanged. The [run summary](measurements/decoding-repair-20261027/summary.json)
records outputs, checkpoint hashes, cycles and bounded program/memory sizes.
These runs overlapped other checks, so their wall times are not isolated benchmarks.

To inspect the optional filter in the terminal, from the repository root:

```sh
python3 scripts/run_model.py \
  --checkpoint models/shakespeare_transformer/checkpoint.json \
  --interactive --tokens 24 --seed 92617660 --min-p 0.05
```

This explicitly selects the previously installed optional model. It is slower
than the character default and still produces language errors. It does not
download or train a model. The CPU preview remains a separately labeled command;
the RTL runner never invokes it.
