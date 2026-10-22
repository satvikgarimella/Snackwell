# Transformer block on the Snackwell SM

The `transformer` command executes a complete single-head, pre-LayerNorm block:
two affine LayerNorms, Q/K/V projections, scaled dot-product attention, an
attention output projection, a two-layer GELU feed-forward network, and both
residual additions. All arithmetic runs as assembly on `snackwell_sm` in RTL
simulation. The default weights are deterministic, untrained test fixtures.

## Run

```sh
python3 scripts/run_kernel.py transformer
python3 scripts/run_kernel.py transformer --rows 8 --cols 4
python3 scripts/run_kernel.py transformer --rows 4 --cols 8
python3 scripts/run_kernel.py transformer --inputs transformer-inputs.json
```

`--rows` is sequence length N and `--cols` is model width D. Both default to 4.
The feed-forward hidden width H is fixed to 2D. Shapes 4x4, 8x4 and 4x8 fit
the current 4 KB shared-memory layout and are covered by RTL execution tests.
Larger shapes, including 8x8 and 16x4, exceed memory and are rejected before
launch. Every stage must also fit the 1,024-instruction ROM.

For the default fixture, the verified forward pass takes 17,215 simulated
execution cycles across 21 launches. Its maximum absolute difference from the
independent whole-block reference is approximately 8.68e-8. These numbers apply
to this fixture: FP operand values can change execution time. Cycle totals
exclude reset, host memory-image transfers, and launch overhead.

## Computation and launches

With row-major matrices and a separate bias for each projection, the block is:

```text
Z = LayerNorm1(X)
Q = Z Wq + bq; K = Z Wk + bk; V = Z Wv + bv
P = softmax(Q K^T / sqrt(D) + causal_mask)
R = X + (P V) Wo + bo
Y = R + GELU(LayerNorm2(R) W1 + b1) W2 + b2
```

GELU uses the tanh approximation. The causal mask is enabled by default: row r
can attend to rows 0 through r. Setting `"causal": false` removes that mask.

| Launches | Operations |
| --- | --- |
| 1 | First affine LayerNorm |
| 2-7 | Q, K and V matrix products, each followed by bias addition |
| 8-11 | Q K-transpose, scale/mask, softmax, probability-times-V product |
| 12-14 | Attention output projection, bias addition, first residual |
| 15 | Second affine LayerNorm |
| 16-18 | Feed-forward expansion, bias addition, GELU |
| 19-21 | Feed-forward contraction, bias addition, final residual |

Matrix products use the existing 4x4 tensor array, tiled when needed. Other
operations use the existing SIMD arithmetic and shared-memory path. Each
launch uses warp 0 with all four lanes active.

The simulation runner resets and launches a program for each stage, carrying
the actual dumped memory image into the next launch. It never substitutes
host-computed intermediate activations. The two norms reuse `norm`, the two
output projections reuse `projected`, and bias/GELU operations can run in
place. Per-stage memory guards check that every word outside that stage's
output region remains unchanged, including all weights and constants.

## Inputs

Use a JSON object with flat row-major arrays. Omitted fields use deterministic
fixture values; unknown keys are rejected.

| Fields | Shape / meaning |
| --- | --- |
| `x` | N x D input activations |
| `wq`, `wk`, `wv`, `wo` | D x D projection matrices |
| `bq`, `bk`, `bv`, `bo` | D-element projection biases |
| `w1` | D x H feed-forward expansion matrix |
| `b1` | H-element expansion bias |
| `w2` | H x D feed-forward contraction matrix |
| `b2` | D-element contraction bias |
| `gamma1`, `beta1`, `gamma2`, `beta2` | D-element affine LayerNorm parameters |
| `epsilon` | LayerNorm epsilon; default 1e-5 |
| `causal` | Boolean; default true |

Inputs and parameters are rounded to FP32 and must remain finite. Epsilon
must round to a positive normal FP32 value. Default biases and beta are zero;
gamma is one. Input and matrix fixtures use a fixed random seed. No trained
checkpoint is loaded by default.

## Verification

Each stage is checked using the existing independent kernel references and
[their numerical tolerances](INFERENCE_KERNELS.md#numerical-algorithms-and-accuracy).
After the final launch, `snackwell/reference/transformer.py` independently
reconstructs the entire block from the original input and parameters. It uses
double-precision sums, square root, exponential and tanh, without reading the
SM's intermediate activations. The final check requires:

```text
abs(actual - reference) <= 5e-5 + 5e-4 * abs(reference)
```

This is a test threshold, not a universal error bound. The SM uses separate
FP32 multiply/add operations and approximate transcendental functions;
overflowing or poorly conditioned inputs can fail verification.

The six transformer test methods exercise random nonzero weights, biases and
both affine norms; causal and unmasked attention; three input shapes; exact
residual identity with zero output branches; and isolation of the first causal
row from future inputs. A fault-injection test replaces the last residual with
a different valid stage and updates its local contract: local verification
passes, while the whole-block reference rejects the result. CLI tests also
check that a failed rerun removes a previous success report.

```sh
python3 -m unittest discover -s tests -p test_transformer.py -v
python3 scripts/run_tests.py
python3 scripts/write_kernel_examples.py --check
```

Outputs live under `build/kernels/transformer/`: `layout.json`, the initial and
final memory images, per-stage assembly, traces and profiles, and `result.json`
with stage checks, summed cycles and whole-block accuracy. A top-level success
report is written only after all checks pass.

## Scope

The complete trained character-model path now builds on these operations;
see [model inference](MODEL_INFERENCE.md) for embeddings, output head, checkpoint
loading and generation. The standalone `transformer` command remains a block
test with fixture weights.

This milestone verifies one transformer block's forward computation. A complete
model's input/output layers and checkpoint integration are handled by the
separate model runner. Multiple attention heads, KV caching, backward passes,
weight updates, global memory, and a persistent hardware launch/data-transfer
runtime are separate work. No FPGA clock speed or hardware execution has been
measured by these simulation tests.
