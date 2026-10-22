# Subword candidate

Four transformer blocks, four heads, width 128, context 64 tokens, vocabulary
512, 669,952 parameters. Trained on TinyShakespeare with CPU PyTorch. Inference
executes through Snackwell's SM RTL, including probability calculation and sampling.

Run from the repository root:

```sh
python3 scripts/run_model.py --checkpoint models/subword_transformer/checkpoint.json \
  --interactive --tokens 16 --seed 92617660 --temperature 0.7
```

Try `My lord, I have something to say`. A token may contain several characters;
`/tokens 16` now means 16 subword predictions. New prompts outside Shakespeare's
style are not a reliable use of this model.

This improves held-out text likelihood over the existing character default, with
similar overall text speed in the recorded run. It still produces malformed
words and sentences. The default character checkpoint is unchanged.

`validation.json` and `test.json` compare identical held-out passages.
`generation.json` retains all 24 CPU diagnostic continuations, including bad ones.
That file names the original prototype hash recorded in `checkpoint.json`'s
`export.prototype_sha256`; the portable export changes identifiers, not weights.
`benchmark.json` records two verified RTL runs; CPU generation matched both
exactly. Hardware verification is separate from language quality.
`verification.json` contains full-window checks, exact cache comparisons, and a
complete RTL training step with all 669,952 parameter gradients checked against
PyTorch. That step did not replace the inference weights.

See [the experiment report](../../docs/SUBWORD_MODEL.md) for the training recipe,
tokenizer, limitations, runtime changes and verification evidence.
