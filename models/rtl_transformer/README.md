# Transformer trained on the Snackwell SM

This checkpoint contains a complete causal character transformer: one block,
two attention heads, width 16, context 16 and 65 vocabulary entries. All 4,657
parameters were initialized from seed 1729 and updated by 100 verified SGD steps
in RTL simulation on the local TinyShakespeare corpus.

| Cross-entropy on fixed evaluation windows | Initial | After 100 steps |
| --- | ---: | ---: |
| Training | 4.755490 | 3.473169 |
| Validation | 4.758931 | 3.241008 |

Each measurement uses 128 target characters. This checkpoint demonstrates working
full-model training and falling loss; 100 updates are not enough to claim fluent
generation. It is separate from the default, larger character-model checkpoint.

- `initial_checkpoint.json`: the random starting weights and SGD settings.
- `checkpoint.json`: the weights after 100 RTL updates; compatible with
  `scripts/run_model.py` and resumable with `scripts/train_transformer.py`.
- `verification.json`: losses, cycles, hashes and the additional checks performed
  on the existing 144,449-parameter transformer.

Continue training, using the same local corpus:

```sh
python3 scripts/train_transformer.py --resume models/rtl_transformer/checkpoint.json --steps 100
```

Run the checkpoint through normal RTL inference:

```sh
python3 scripts/run_model.py --checkpoint models/rtl_transformer/checkpoint.json --prompt 'ROMEO:' --tokens 4
```

The three-step check of the larger model lowered its training sample loss but
did not improve its small validation sample. Its default checkpoint was not
replaced. See [the full training walkthrough](../../docs/TRANSFORMER_TRAINING.md)
for equations, validation, artifact locations and the remaining work.
