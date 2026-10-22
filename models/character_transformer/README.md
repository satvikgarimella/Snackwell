# Character-transformer checkpoint

The default is restored to `43f211daad7c…`: four blocks, four attention heads,
width 64, feed-forward width 128, context 32 and 65 character classes. It has
144,449 FP32 parameters trained on TinyShakespeare with CPU PyTorch. Snackwell
executes the forward pass and character selection in RTL simulation.

The width-128 checkpoint is preserved in `wide-context64/`. It reduced held-out
prediction loss but still generated poor sentences and more than doubled
simulation time. Neither model reliably generates coherent English. Numerical
agreement with a reference does not measure sentence quality.

```sh
python3 scripts/run_model.py --interactive --tokens 16 --seed 92617660
python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
```

Restart an existing interactive process to load the new kernel code. The terminal
should show width 64, context 32 and checkpoint `43f211daad7c`. Each character
appears only after the numerical checks pass. The completion status now says
`RTL/reference match` to describe what those checks establish.

The default temperature remains 0.7. `/temperature T`, `/tokens N`, `/seed N` and
`/quit` work interactively. Generation reuses K/V until the 32-character context
fills, then rebuilds the newest 24 characters at their new learned positions.
The cache is raw RTL output stored on the host, not on-chip memory.

For a clearly labeled CPU-only preview:

```sh
build/model-venv/bin/python scripts/preview_model.py --interactive
```

That preview is for inspecting text quality. It never substitutes predictions
for an RTL run. CPU training likewise does not implement training on Snackwell.

## Evidence

- `checkpoint.json` and `evaluation.json`: restored original files, unchanged.
- `verification.json`: current checkpoint/source hashes and verification results.
- `quality/repair/`: current regression, evaluation, cache, timing and sampling-trial records.
- `wide-context64/`: rejected default, evaluation and its original verification evidence.
- `previous-context32/`: original snapshot of the restored checkpoint and earlier evidence.
- Other root-level comparison/performance files are historical. Their embedded
  checkpoint hashes identify the models they tested.

See [the repair report](../../docs/INFERENCE_REPAIR.md) for the separate effects
of the rollback and kernel optimization, and [the model guide](../../docs/MODEL_INFERENCE.md)
for the execution and verification boundary. Global memory, hardware training
and FPGA implementation remain future work.
