# October 27 quality trials

These are validation and generation results, including failed continuations.
No candidate in this directory was promoted as the sentence-quality fix.

- `continued-experiment.json` contains the recipes, exact hashes and decisions.
- `validation-*.json` contains the same token targets and text passages for each candidate.
- `generation-*.json` retains every complete continuation, including greedy loops.
- `rtl-wide4-window.json` verifies the width-256 parent model at full context and through cache rebuild/decode.
- The three test logs cover the new runtime support and related regressions.
- `prose-manifest.json` records training-only source preparation and held-out exclusion.

Generation comparisons run on the CPU mathematical model. They do not claim
that all sample cases ran through RTL. The RTL audit and its checkpoint hash
are recorded separately. Training used host PyTorch/MPS, not Snackwell RTL.

The wider, deeper and prose candidates have 32 primary and 20 additional cases.
The continued-width-128 trial has the 32 primary cases. The character default's
additional cases are in `generation-default.json`; three prompts overlap its
primary set in `generation-default-matched.json`.

See [the report](../../QUALITY_REPAIR.md) for interpretation and remaining limits.
