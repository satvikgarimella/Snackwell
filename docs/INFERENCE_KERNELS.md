# Small inference kernels on Snackwell

The library now includes tiled matmul, bias/residual addition, ReLU, tanh-GELU,
row sum/max, affine LayerNorm, stable row softmax, linear, and single-head
scaled dot-product attention, plus a complete pre-LayerNorm transformer block.
All device computation runs as Snackwell assembly
on the real SM in RTL simulation. Python generates instruction sequences and
constant tables, loads inputs, launches programs, and verifies results.

## Run everything

```sh
python3 scripts/run_kernel.py all
```

This verifies 17 workloads: the four earlier vector/matmul programs, all twelve
library commands, and a separate 16x16 tiled matmul. It writes a summary to
`build/kernels/all/result.json` plus per-workload assembly, input/output memory,
cycle traces, profiles, and verification results. Any mismatch exits nonzero.

```sh
python3 scripts/run_kernel.py tiled_matmul --size 8
python3 scripts/run_kernel.py tiled_matmul --size 16
python3 scripts/run_kernel.py bias_add --rows 8 --cols 16
python3 scripts/run_kernel.py residual_add
python3 scripts/run_kernel.py relu
python3 scripts/run_kernel.py gelu
python3 scripts/run_kernel.py row_sum
python3 scripts/run_kernel.py row_max
python3 scripts/run_kernel.py layernorm --rows 4 --cols 16
python3 scripts/run_kernel.py softmax --rows 8 --cols 8
python3 scripts/run_kernel.py linear --size 8
python3 scripts/run_kernel.py attention --rows 8 --cols 8
python3 scripts/run_kernel.py transformer
```

Dimensions are 4, 8, or 16. `--size` controls square tiled matmul and linear
(default 8); `--rows`/`--cols` control the other new commands (default 4x4).
For attention they mean sequence length and head width; for the transformer,
sequence length and model width, with a feed-forward width of twice the model
width. Memory capacity is
validated before launch. For example, 16-token attention with width 8 fits;
16-token attention with width 16 does not fit the current 4 KB memory.
There is no padding or support for arbitrary non-multiple-of-four dimensions.
The transformer has room for 4x4, 8x4, and 4x8 inputs with its current layout;
see [the transformer guide](TRANSFORMER.md) for its parameters and limits.

## What each command does

| Command | Computation | GPU program launches |
| --- | --- | ---: |
| `tiled_matmul` | C=A B using 4x4 tensor tiles and partial-result accumulation | 1 |
| `bias_add` | Y[r,c]=X[r,c]+b[c] | 1 |
| `residual_add` | Y=X+B, elementwise | 1 |
| `relu` | Y=max(X,0), preserving NaNs and signed zeros | 1 |
| `gelu` | Standard tanh approximation to GELU | 1 |
| `row_sum` | One sequential FP32 sum per input row | 1 |
| `row_max` | One maximum per input row | 1 |
| `layernorm` | Normalize each row, then multiply by gamma and add beta | 1 |
| `softmax` | exp(X-row_max) divided by its row sum | 1 |
| `linear` | Y=A B+b | 2 |
| `attention` | softmax(Q K-transpose / sqrt(head_width) + causal_mask) V | 4 |
| `transformer` | Pre-LayerNorm block: Q/K/V projections, attention, output projection, GELU feed-forward network, two residuals | 21 |

Linear, attention and the transformer are pipelines of GPU programs. The simulation runner
passes the actual dumped memory from each stage to the next launch; Python
does not compute intermediate scores, probabilities, or output activations for
the device. Each stage is checked, with an additional independent whole-pipeline
reference for attention and the transformer.
This simulates several launches with memory preserved through host images;
it is not yet a persistent hardware runtime or DMA path.

## Supply inputs

Use `--inputs path.json`. Arrays are flat and row-major. Fields omitted from
the JSON use deterministic example values; unknown fields are rejected.

| Command | Input fields |
| --- | --- |
| `tiled_matmul` | `a`, `b`: size x size each |
| `linear` | `a`, `b`: size x size each; `bias`: size elements |
| `bias_add` | `x`: rows x cols; `b`: cols elements |
| `residual_add` | `x`, `b`: rows x cols each |
| `relu`, `gelu`, `row_sum`, `row_max`, `softmax` | `x`: rows x cols |
| `layernorm` | `x`: rows x cols; optional `gamma`, `beta`: cols; optional `epsilon` |
| `attention` | `q`, `k`, `v`: rows x cols each; optional `causal`: boolean |
| `transformer` | `x`, projection and feed-forward weights/biases, two affine norms; [full format](TRANSFORMER.md#inputs) |

LayerNorm defaults to gamma=1, beta=0, epsilon=1e-5. Epsilon must round to a
positive normal binary32 value. Attention defaults to causal masking, allowing
query row r to read key/value rows 0..r. Set `"causal": false` for unmasked
attention. It has one head, no dropout, and no KV cache.

For example, save this as `softmax-inputs.json`:

```json
{
  "x": [1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 0, -3, -1, 1, 3]
}
```

```sh
python3 scripts/run_kernel.py softmax --inputs softmax-inputs.json
```

Inputs are rounded to FP32 before launch. Numerical kernels require finite
inputs, except softmax also accepts negative infinity for masked entries, with
at least one finite value per row. ReLU additionally supports infinities and
NaNs. Python's JSON loader accepts its nonstandard `-Infinity`/`NaN` tokens for
these test cases; ordinary finite JSON numbers are sufficient for normal runs.
Intermediate overflow can invalidate a numerical workload; LayerNorm explicitly
rejects overflowing variance, and output verification rejects non-finite
approximate results. This is not a general transcendental math library with
IEEE exception flags or correctly rounded exp/divide/sqrt.

## Tiled multiplication

One full warp computes output tiles sequentially. For each 4x4 output tile,
the kernel initializes four accumulator registers to zero, then loops over the inner
dimension in blocks of four:

1. Load a 4x4 A tile and a 4x4 B tile into eight registers.
2. Execute TMUL into R0--R3, reusing A's registers after the tensor unit gathers them.
3. Add the new tile to the partial C rows held in R8--R11.
4. Advance A/B pointers and repeat until the K blocks are finished.

The completed output tile is then stored once. When the model runtime splits K
across launches, the next launch loads the previous partial tile once.

The 8x8 product executes 8 TMULs; 16x16 executes 64. Those are full GPU programs
with counted loops, not host-computed partial results. TMUL sums four products
from positive zero; FADD then adds each completed tile to the running output.
The reference reproduces that specified order, since it can differ in FP32
rounding from one uninterrupted dot product.

The 16x16 program contains 561 instructions and uses 768 shared-memory words
for A/B/C. All generated programs are checked against the 1024-instruction ROM
limit, and layouts plus constants against the 1024-word memory limit.

## Row kernels and synchronization

Most row kernels assign one row to each lane. Four lanes process four rows,
and a counted loop advances to the next group. Each lane performs its own
reduction, with no dependence on another lane's partial result. GELU instead
loops over groups of four contiguous elements to reuse its longer math routine.

The standalone examples launch only warp 0 with all four lanes active. The model
runtime also generates matrix kernels that assign disjoint output tiles to up
to four warps. Each tile's K accumulation stays in one warp; no barrier is needed
because these warps do not consume each other's outputs. Standalone row kernels
also avoid inter-warp communication and therefore require no block barriers.
This is a correct initial mapping, not a claim of optimal reduction
throughput. General cooperating multiwarp kernels remain separate work.

## Numerical algorithms and accuracy

The standard definitions follow [PyTorch GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html),
[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
and [scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
The implementation and dependency-free references live in separate modules.

- **Reciprocal:** obtain a starting estimate from FP32 bits, then do four
  residual-form Newton steps, `y = y + y*(1-x*y)`. Softmax/GELU denominators
  are positive normal values. The residual form avoids losing small corrections
  when `2-x*y` would round to exactly 1.
- **Reciprocal square root:** obtain a bit-based estimate, then perform four
  steps `y = y*(1.5-0.5*x*y*y)`. LayerNorm adds epsilon before this operation.
- **Exponential:** for nonpositive x, evaluate a degree-six Taylor polynomial
  for exp(x/32), then square five times. Values below -80, including -inf,
  become zero. Softmax subtracts its row maximum before exponentiation, so at
  least one numerator is 1. The cutoff discards terms below about 1.81e-35.
  Reducing the number of squarings limits amplified FP32 rounding near zero;
  the larger polynomial error in the far negative tail has a tiny absolute value.
- **GELU:** evaluate the tanh approximation with coefficient 0.044715 using
  the stable exponential form of tanh. Clamp the magnitude used inside that
  approximation to 10, where it has saturated, while retaining the original
  input for the output multiplication.
- **LayerNorm:** compute mean and population variance in two passes with
  sequential FP32 sums, then apply reciprocal sqrt, gamma, and beta.

Multiplication and addition round separately; no fused MAC is assumed.
Arithmetic, tiled matmul, ReLU, and reductions are checked bit for bit against
the reference, with NaNs compared by classification. Approximate kernels use
`abs(actual-reference) <= atol + rtol*abs(reference)`:

| Function | Relative tolerance | Absolute tolerance |
| --- | ---: | ---: |
| tanh-GELU | 1e-4 | 3e-6 |
| softmax | 1e-4 | 2e-6 |
| LayerNorm | 3e-5 | 3e-6 |
| complete attention | 2e-4 | 2e-5 |
| complete transformer block | 5e-4 | 5e-5 |

These are verification thresholds, not universal error bounds for arbitrary
ill-conditioned or overflowing inputs. Softmax also checks nonnegative
probabilities and a row sum within 2e-5 of one. Its reference uses host math.exp,
not the device polynomial. GELU uses host math.tanh. LayerNorm's reference
reproduces the FP32 mean/variance order and uses host sqrt for the reciprocal;
the final attention reference starts independently from original Q/K/V with
double-precision sums and exp. The whole-block transformer reference likewise
starts from the original input and parameters, uses double-precision arithmetic,
and never consumes the SM's intermediate outputs.

## Supporting ISA additions

Three register-register instructions use the existing integer writeback and
scoreboard path in the full SM:

| Instruction | Opcode | Result |
| --- | --- | --- |
| `SHR Rd, Ra, Rb` | 0x10 | Logical unsigned right shift by Rb[4:0] |
| `SLTU Rd, Ra, Rb` | 0x11 | Integer 1 if unsigned Ra < Rb, else 0 |
| `FLT Rd, Ra, Rb` | 0x12 | Integer 1 if FP32 Ra < Rb, else 0 |

FLT compares encodings, handles negative order and infinities, treats signed
zeros as equal, and returns false for either NaN. It does not raise exception
flags. Comparisons retain ordinary active-mask and RAW/WAW behavior.
The older standalone single-warp teaching core still has its earlier subset.

## Source, output, and verification

- `snackwell/kernels/inference.py`: shape-specialized assembly emitters and layout.
- `snackwell/reference/inference.py`: independent expected results and tolerances.
- `snackwell/kernels/transformer.py`: transformer layout and stage composition.
- `snackwell/reference/transformer.py`: independent whole-block reference.
- `snackwell/inference_runner.py`: RTL execution and stage-to-stage memory transfer.
- `kernels/nn/*.asm`: readable default assembly examples, with memory addresses.
- Each run's `<stage>/program.asm`: the exact assembly that was assembled.
- Each workload's `layout.json`: array addresses and immutable constants.
- Each stage's `input.hex`, `output.hex`, `trace.csv`, `profile.json`, `result.json`.
- The top-level `result.json`: only written after the entire workload passes.

Edit the emitters to change these kernels; the default example files are
generated snapshots. Refresh or check those snapshots with:

```sh
python3 scripts/write_kernel_examples.py
python3 scripts/write_kernel_examples.py --check
```

The program generator uses shapes and constants only, never input values to
precompute device outputs. Program images are identical for different input
values at the same shape/configuration. There are no HDL real-number shortcuts.

```sh
python3 -m unittest discover -s tests -p test_inference_kernels.py -v
python3 scripts/run_tests.py
```

Tests cover every command, all supported dimensions for basic kernels, tile
accumulation and counts, signed/random inputs, dense GELU values and saturation,
softmax offsets/extreme tails/masking, affine LayerNorm and different scales,
causal/unmasked attention and several shapes, the new ISA through the full SM,
memory guards, rejected inputs/capacity, and deliberately removed stores.
Transformer tests also cover nonzero parameters, both affine norms, causal
isolation, residual identity, and a deliberately broken final stage that passes
its local check but fails the independent whole-block reference.
Existing vector, matmul, memory, FP32, tensor, and control tests continue to run.

Cycle reports exclude reset, image loading, and launch overhead. Pipeline totals
sum the GPU execution cycles of their stages. All measurements remain simulation
results. Larger model weights, global memory, a general host runtime, pretrained
model integration, multiple heads, and KV-cache decoding are outside this kernel
library milestone.

The separate [trained character-model runner](MODEL_INFERENCE.md) now composes
these operations with token embeddings, an output head and sampled or greedy generation.
It loads tensor chunks through the simulation host; hardware global memory and
the persistent runtime remain future work.
