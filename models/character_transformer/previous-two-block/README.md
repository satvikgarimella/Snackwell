# Character-transformer checkpoint

The default model has two blocks, four attention heads per block, width 64,
feed-forward width 128, 32-character context, 65 vocabulary entries and
77,505 FP32 parameters. It is trained in CPU PyTorch and executed in Snackwell
RTL. Training on Snackwell is still future work.

- `checkpoint.json`: FP32 parameters, configuration and complete training history.
- `evaluation.json`: eight held-out windows and PyTorch probability references,
  tied to the checkpoint by SHA-256.
- `verification.json`: regression, RTL evaluation/generation, timing and source fingerprints.
- `held-out-comparison.json`: CPU comparison at identical test target positions.
- `generation_checks.json`: all fixed-prompt old/new CPU samples and three RTL continuations.
- `generation-optimization.json`: exact probability agreement between full and final-row execution.
- `previous-small/`: the previous 4,657-parameter default with its original evidence.
- `baseline/`: the original checkpoint before continued training and sampling work.

```sh
python3 scripts/run_model.py --interactive --tokens 4
python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
```

The new model improves held-out next-character accuracy from 38.48% to 51.12%
on 2,048 identical CPU-reference targets. This is separate from the smaller
256-position RTL evaluation. Text still contains errors and is not conversational.
Generation takes roughly 15–16 seconds per character in Verilator on the
development Mac; the larger model performs substantially more work.

Generation samples on the SM by default. Use `--seed 1731` for repeatable output
or `--decode greedy` for maximum-probability selection, which can still repeat.
Inference uses no PyTorch or NumPy. See the [model guide](../../docs/MODEL_INFERENCE.md)
and [upgrade decisions](../../docs/MODEL_UPGRADE.md) for details and reproduction.
