# Compact sentence model

Four transformer blocks, two heads, width 32, context 32 subword tokens, and
41,569 FP32 parameters. The 97-entry tokenizer and all weights are self-contained
in `checkpoint.json`.

This model answers a location question from two facts in a fixed format:

```text
The ball is in the kitchen. The book is in the garden. Where is the ball?
```

```sh
python3 scripts/run_sentence.py --interactive
```

Weights were trained in host PyTorch. Generation runs on the Snackwell SM RTL,
with independent numerical verification and greedy token selection. It uses the
existing kernels and host-managed tensor transfers. FPGA resource use and timing
still require synthesis.

Measured autonomous host generation: 576/576 reserved questions and 6,244/6,272
prompts across the complete supported format (99.55%). **28 wrong answers remain.**
The RTL audit passed 18/18 answer checks and matched cached versus fresh decoding.

Supported objects: ball, book, cup, hat, key, toy, bag, box. Supported locations:
kitchen, garden, bedroom, office, hallway, bathroom, attic, cellar. Use two
different objects and places, then ask about one of those objects.

Checkpoint SHA256:
`66af3342b4a66ee4e63816f30756f3aed746b749b300ecd6f13c324be0d2423e`.

See [the full report](../../docs/SENTENCE_DEMO.md) for scope, remaining failures,
training recipes, numerical checks, and reproducibility.
