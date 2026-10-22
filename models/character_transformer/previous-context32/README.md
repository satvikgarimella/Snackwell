# Character-transformer checkpoint

The default model has four blocks, four attention heads per block, width 64,
feed-forward width 128, 32-character context, 65 vocabulary entries and
144,449 FP32 parameters. It is trained in CPU PyTorch and executed in Snackwell
RTL. Training on Snackwell is still future work.

- `checkpoint.json`: FP32 parameters, configuration and complete training history.
- `evaluation.json`: eight held-out windows and PyTorch probability references,
  tied to the checkpoint by SHA-256.
- `verification.json`: regression, RTL evaluation/generation, timing and source fingerprints.
- `performance_verification.json`: subsequent speed changes, unchanged outputs and updated checks.
- `rtl_performance_verification.json`: FP32/tensor cycle improvements and their recorded source revision.
- `runtime_performance_verification.json`: multiwarp kernels, shared-memory broadcast and runner checks.
- `kv_cache_verification.json`: cached/uncached RTL comparisons, context-policy quality and timing.
- `decode_optimization_verification.json`: one-row kernels, independent reference caching,
  runtime comparisons and the subsequent model-quality experiments.
- `held-out-comparison.json`: CPU comparison at identical test target positions.
- `validation-comparison.json`: separate paired validation comparison used for selection.
- `generation_checks.json`: fixed-prompt CPU diagnostics, additional seeds and actual RTL evidence.
- `generation-optimization.json`: exact probability agreement between full and final-row execution.
- `previous-two-block/`: the prior 77,505-parameter model and its original evidence.
- `previous-small/`: the previous 4,657-parameter default with its original evidence.
- `baseline/`: the original checkpoint before continued training and sampling work.

```sh
python3 scripts/run_model.py --interactive --tokens 4
python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
```

For quick text exploration using the optional CPU reference, without RTL:

```sh
build/model-venv/bin/python scripts/preview_model.py --interactive
```

On the current paired 2,048-target CPU test comparison, next-character accuracy
improved from 53.52% to 55.76%. This is separate from the smaller 256-position RTL
evaluation (140 correct next characters, all numerical checks passing). The new
model improves word formation, but sentence grammar remains inconsistent.

Generation defaults to temperature 0.7 and samples on the SM. `--temperature 1`
uses the unmodified distribution; `--seed 92617660` repeats a run. Greedy decoding
remains available. The raw model probabilities are always checked separately.
A controlled speed comparison on the development Mac reduced a character from
36.24 to 20.65 seconds with the same model and probabilities. The first model
upgrade measurement was 37.5 seconds. See the [speed investigation](../../docs/INFERENCE_SPEED.md);
these are verified simulation timings, not FPGA performance. The subsequent RTL
update measured 22.44 to 11.21 seconds in a new paired run, with identical results.
See [the RTL speed investigation](../../docs/FP32_SPEED.md).
The subsequent multiwarp and runtime changes are recorded separately in
[RUNTIME_SPEED.md](../../docs/RUNTIME_SPEED.md); they preserve this checkpoint
and its numerical outputs.
The later [KV cache implementation](../../docs/KV_CACHE.md) keeps these weights
but changes generation to periodic overlapping context rebuilds. Cached steps
reuse raw K/V computed by RTL; `--no-kv-cache` restores the old sliding policy.
The [one-row decode update](../../docs/DECODE_OPTIMIZATION.md) further reduces
kernel launches and repeated verification work without changing these weights.

Inference uses no PyTorch or NumPy. The optional CPU preview does use PyTorch and
clearly labels itself. See the [model guide](../../docs/MODEL_INFERENCE.md) and
[quality investigation](../../docs/LANGUAGE_QUALITY.md) for details and evidence.
