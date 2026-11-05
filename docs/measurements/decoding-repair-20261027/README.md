# Decoding repair evidence

The JSON probe files contain all 52 cases per setting. They are CPU diagnostics, not RTL execution. `probe.py` reproduces these exploratory comparisons with the optional PyTorch environment; its threshold multiplication is double precision.

The deployed filter uses an independently checked FP32 threshold. `filters-tests.log` and `final-regression.log` cover the final kernels, prompt preservation, cancellation, and cache rebuild. `regression.log` is the wider 38-method focused run; `live-regression.log` adds four local-server tests. Test counts overlap and must not be added across repeated runs.

`installed-preview.json` records both prompt-boundary settings for the installed optional model. `rtl-character.json` and `rtl-name.json` record actual, fully verified terminal RTL runs of the installed character and optional subword models respectively. Their timing is not an isolated benchmark.

`checkpoint-hashes.json` confirms that the three installed model files were not replaced. The source change and limitations are described in [the decoding report](../../DECODING_REPAIR.md).
