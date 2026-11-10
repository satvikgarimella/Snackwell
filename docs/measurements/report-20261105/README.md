# Evidence archived for the November 5 engineering report

These are byte-for-byte copies of existing local audit records. No training,
benchmark, or regression was rerun to create this archive. The
[manifest](manifest.json) records each original path, byte count, SHA256, and
the report's implementation baseline. See the [engineering report](../../REPORT.md)
for interpretation.

## Regression validation

- [Combined summary](validation-summary.json): 39 RTL benches and 185 Python
  methods passed across the initial run and a focused retry; lint, formatting,
  assembly examples, and checkpoint validation are recorded separately.
- [Original full-suite result](initial-suite-result.json) and
  [Python log](initial-python.log): the initial invocation reported failure
  because `LiveModelTests.setUpClass` could not bind localhost in the sandbox.
  All 39 RTL benches passed and 181 Python methods completed without individual
  failures.
- A separate retry of `tests.test_live_model` with loopback access passed its
  four methods in 5.096 seconds. That observed result is preserved in the
  combined summary. A separate raw retry log was not saved.

The initial result intentionally retains `passed: false`. The combined record
must not be described as a single uninterrupted successful test invocation.

## October 21 transformer-training audit

- [Small-model learning result](transformer-learning.json): 100 full RTL SGD
  steps from random initialization, all step records, and fixed before/after
  evaluation on 512 sampled target positions per split.
- [Larger-model gradient audit](transformer-gradient-audit.json): two full RTL
  updates of the 669,952-parameter host-pretrained subword checkpoint, direct
  float64 autograd comparisons, FP32 update checks, and checkpoint reload checks.

The two runs overlapped; their wall times are not isolated speed benchmarks.
The original per-launch assembly, memory images, intermediate tensors, and
audit driver remain under local `build/training-audit-20261021/`. This archive
preserves summary evidence, not those complete run directories. The full audit
method and reproduction commands are in [TRAINING_AUDIT.md](../../TRAINING_AUDIT.md).
