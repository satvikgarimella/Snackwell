# Archived width-128 experiment

Checkpoint `933a263b2d03…` has four blocks, four heads, width 128, context 64 and
555,073 parameters. It was briefly the default. It lowered held-out loss but
still produced weak sentences and more than doubled the measured generation
latency. The smaller width-64 checkpoint is restored as the default.

```sh
python3 scripts/run_model.py --checkpoint models/character_transformer/wide-context64/checkpoint.json --interactive --tokens 16
```

The evaluation, verification and quality records here are the original experiment's
evidence. Their source hashes, timings and original paths describe that historical
run, before the subsequent SIMD kernel changes. See
[the capacity report](../../../docs/MODEL_CAPACITY.md) and
[the rollback](../../../docs/INFERENCE_REPAIR.md).
