# Restoring the faster default and improving decode kernels

The width-128 model was a poor default replacement. Its matched test loss was
about 4.1% lower, but its generated sentences remained weak. The original
16-character benchmark went from 35.69 seconds to 78.06 seconds. The default is
restored to the unchanged width-64 checkpoint `43f211daad7c…`; the larger
`933a263b2d03…` checkpoint and its evidence remain in
`models/character_transformer/wide-context64/`.

This rollback changes the model back to its earlier predictions. The kernel
optimization below is separate: it preserves that checkpoint's results exactly.
No new weights, word filters, canned completions, CPU-generated RTL answers or
relaxed verification tolerances were introduced.

## Kernel changes

Single-row matrix products now use the existing FP32 SIMD unit. Each lane owns
one output column and reads the common input-vector element through shared
memory's broadcast path. The kernel issues independent products into separate
registers, letting subsequent loads overlap multiplication. It sums products in
groups of four, then accumulates those group sums in the same order as TMUL.
Bias is added once after the last K chunk. Multirow products still use the
systolic array, which benefits from having multiple useful rows.

LayerNorm computes the same sequential mean and variance in all four lanes,
then distributes independent affine outputs across them. This avoids changing
the reduction's FP32 rounding. Single-row residual and bias additions also
distribute output columns across lanes. Scalar kernel emitters remain available;
the model runtime explicitly opts into the parallel output paths.

There is no new RTL, execution unit or ISA instruction. The source fingerprint
remains `75031f1eb062…`. The simulator still executes every model operation, and
every launch retains its independent numerical checks and memory guards.

## Measured result and verification

Two runs of each runtime used the same checkpoint, original prompt
`My lord, I have something to say`, seed 92617660 and temperature 0.7, in
before/after/after/before order. Training, evaluation and regression jobs had
finished. Compilation and tracing were excluded; all verification was included.

| Metric, restored width-64 checkpoint | Previous kernels | New kernels |
| --- | ---: | ---: |
| First character | 7.53 s | 7.57 s |
| Cached character | 1.131 s | 1.078 s |
| Context rebuild | 5.72 s | 5.77 s |
| All 16 characters, including two rebuilds | 33.68 s | 33.12 s |
| Simulated cycles | 32,970,230 | 31,380,631 |
| Kernel launches | 7,242 | 7,242 |

The kernels reduce cycles by 4.82% and measured cached latency by
4.75%. Overall measured wall time changes by only
1.66%; prefill and rebuild times are effectively unchanged within run variability.
This is a modest optimization, not a large interactive-speed improvement.
Most of the reduction relative to the user's 78.7-second wider-model run comes
from restoring the smaller model. Those two checkpoints produce different text.

Every output probability, sampling probability, selected character and error
metric matches across the four same-checkpoint runs. Their unedited output is:

```text
My lord, I have something to say 'twas. There's
```

All **107 Python regression tests** passed with optional PyTorch tests enabled
and no skips. The new matrix-vector tests compare actual Icarus and Verilator
execution against the tensor kernel and independent integer FP32 oracle,
including one through four warps, transposes, prior sums, bias, NaNs, infinities,
subnormals and separate-rounding cases. Guard checks cover every launch.

All **256 exported held-out positions** pass the model checks. All eight trained
cached/full comparisons match exactly at context lengths 3, 4, 5, 6, 31, 32, 24
and 25, including the overflow rebuild. No standalone RTL bench sweep was rerun;
the RTL itself is unchanged. Historical hardware test counts are separate.

The current [verification record](../models/character_transformer/verification.json)
links the source/checkpoint hashes, regression log, actual RTL evaluation,
cache comparisons, timing data and rejected sampling diagnostics.

## Language-quality result

The checkpoint's language quality is **not fixed**. Fast CPU diagnostics tested
two checkpoints, eight prompts, four fixed seeds and 96 generated characters per
case. In addition to ordinary temperature-0.7 sampling, an exploratory script
tried nucleus thresholds 0.95 and 0.9. All 192 outputs are retained. Grammar and
meaning remained weak; mean repeated-fourgram fraction increased with truncation
for both models. Those sampling changes were rejected and are not in the runtime.

The default remains temperature-0.7 categorical sampling. A model replacement
needs evidence of useful continuations across varied prompts and seeds, alongside
held-out metrics and its measured RTL cost. A small loss improvement or a good
isolated sentence does not meet that standard. The terminal now says
`RTL/reference match` instead of the ambiguous `verified` completion label.

## Reproduce

```sh
python3 scripts/run_model.py --stream \
  --prompt 'My lord, I have something to say' --tokens 16 --seed 92617660
python3 scripts/run_model.py --evaluate models/character_transformer/evaluation.json
python3 -m unittest tests.test_matvec tests.test_decode_kernels tests.test_wide_model -v
build/model-venv/bin/python -m unittest discover -s tests -v
```

Restart an older interactive process to load the changed Python kernel generator.
It should print width 64, context 32 and checkpoint `43f211daad7c`.
Training the reference checkpoint on the CPU is distinct from future hardware
training. These simulation times are not FPGA throughput measurements.
