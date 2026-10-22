# Previous small checkpoint

This is the preserved default before the two-block model upgrade: one block,
one head, context 16, width 16, feed-forward width 32, 65 characters and 4,657
parameters. It was initially CPU-trained for 6,000 updates and then continued
with a fresh optimizer for 12,000 updates, selecting continuation step 9,500.
Its checkpoint, evaluation and verification files retain their original bytes.
Their source fingerprints and measurements describe the code at that time.

To run this smaller, faster model using the current runner:

```sh
python3 scripts/run_model.py --checkpoint models/character_transformer/previous-small/checkpoint.json --interactive --tokens 4
```

Current launch counts and times differ because the runner now avoids unnecessary
final-row work. The checkpoint still has limited spelling and sentence quality.
See the [comparison](../../../docs/MODEL_UPGRADE.md).
