# NVFP4 tensor arithmetic

Snackwell's reduced NVFP4 mode uses signed E2M1 elements and unsigned E4M3
scales, with one scale for each 16 values along the matrix inner dimension.
The accumulator is FP32. This retains the format's group size while using
four-wide steps in the existing 4×4 array.

The public format references are NVIDIA's
[CUTLASS type definitions](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/fundamental_types.html),
[PTX 8.7 alternate formats and block-scaled matrix instructions](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#alternate-floating-point-data-formats),
and [Transformer Engine's two-level scaling description](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).
Those specify the element/scale formats and the separate tensor-wide FP32
factor. The instructions, memory layout, reduced shape, and rounding sequence
below are Snackwell's implementation contract. They are not PTX encodings or
a claim of bit-identical results or throughput on a B200.

## Instructions and storage

| Instruction | Opcode | Operation |
| --- | --- | --- |
| `TMMA_NVFP4 T0, Ra, Rb` | `0x24` | Start a fresh 4×16 times 16×4 product |
| `TMMA_NVFP4_ACC T0, Ra, Rb` | `0x25` | Add that product to an initialized tensor tile |

`T0` or `T1` must be allocated by the issuing warp. Both descriptor registers
must contain the same word in all four active lanes. Encoding and ownership
rules are the same as `TMMA`: Rd and immediate bits 13:1 are zero; immediate
bit 0 chooses the tensor slot. Commit, wait, read, release, and free use the
existing [tensor-memory protocol](TENSOR_MEMORY.md).

Each descriptor contains a shared byte base in bits 11:0 and vector stride in
bits 23:12; bits 31:24 are zero. Base and stride are aligned to four bytes.
Stride is at least 12, and all four vectors must fit within 4 KiB. Each A
vector is a matrix row; each B vector is a matrix column. A vector stores:

| Byte offset | Contents |
| --- | --- |
| 0–3 | Elements k=0..7; first element in the lowest nibble |
| 4–7 | Elements k=8..15; first element in the lowest nibble |
| 8 | The scale for all sixteen elements |
| 9–11 | Zero padding, checked by hardware |

Thus both operands together occupy 96 bytes per job, including padding, versus
512 bytes for the same 128 elements in FP32. The mathematical format uses 4.5
bits per element before tensor-wide scale overhead; this simple word-addressed
layout uses 6 bits per element. These storage numbers are not speed claims.
For larger K, software submits consecutive 16-element groups with `_ACC` and
the group's own scales, preserving the tensor accumulator across jobs.

The unit gathers and validates both complete operands before starting any
arithmetic. It retains all packed elements and scales while running four
four-wide array passes, carrying the FP32 sums between passes. The result only
becomes complete after the fourth pass. No consumer can read an intermediate
tile. Existing asynchronous-copy guards include the packed data and scale
words; after gathering, a shared staging slot can be released and reused while
arithmetic continues from the captured operands.

## Arithmetic and conversion

E2M1 has sign bit 3, exponent bits 2:1 with bias 1, and fraction bit 0. Positive
magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, and 6; both signs of zero are supported.
UE4M3 occupies bits 6:0 of its byte: exponent bits 6:3 with bias 7, fraction
bits 2:0, and a required zero top bit. Finite scales range from zero to 448,
including subnormals starting at 2^-9. Code `0x7f` is NaN and is rejected by
this finite-scale instruction subset. Bytes with bit 7 set are also rejected.

Each PE forms the exact product `(a_e2m1 * scale_a) * (b_e2m1 * scale_b)` using
small integer significands and exponent arithmetic. Each scaled significand
fits in eight bits, so the product path uses an 8×8 coefficient multiplication
instead of launching the FP32 multiplier. All nonzero finite products fit
exactly in normal FP32. Each product then enters the existing FP32 adder in
increasing k order, with round-to-nearest, ties-to-even after each addition.
The FP32 initial accumulator retains the existing signed-zero, subnormal,
infinity, and canonical-NaN handling. The multiplication and addition are
separate; NVIDIA's internal reduction order is not reproduced.

The tensor-wide dequantization factors stay in software. After all K groups,
the kernel computes `alpha = round32(global_scale_A * global_scale_B)` with
`FMUL`, then `round32(tile * alpha)`. An initial accumulator for `_ACC` is in
the units before this epilogue. A bias in final output units is added afterward.

The host quantizer accepts finite FP32 inputs and rounds elements and block
scales to nearest with ties to even, saturating at the largest finite code.
It selects the tensor factor from the tensor maximum divided by `448 * 6`,
rounded to FP32 and clamped to the smallest positive FP32 value. An all-zero
tensor uses factor 1 and zero block scales. Each block chooses its scale from
its own maximum and the stored tensor factor, then encodes elements using that
stored block scale. Underflow to a zero block scale produces signed-zero
elements. NaN/infinite host inputs are rejected. This is a deterministic
inference quantizer; it does not implement stochastic rounding or a quantized
training recipe.

## Errors and visibility

Invalid descriptors, partial warps, nonuniform registers, and reserved fields
retain the existing tensor error behavior, with no memory or arithmetic side
effects. Invalid scale bytes or padding are detected during the operand gather:
the job completes as failed, sets `TSTATUS` bit 6 on its producer, and leaves
the accumulator storage unchanged. No array pass runs. The job still uses a
completion ticket: consumers wait, receive bit 6, and release their leases.
Reads of a failed tile return an error, and `_ACC` cannot consume it. A fresh
overwrite job or explicit `TST` initialization can recover after release.

`TSTATUS` is sticky until reset and is not a completion fence. As with the
earlier tensor interface, software checks it explicitly; global/copy device
fault flags do not substitute for checking tensor status. Reset discards all
in-flight tensor work and leases.

## Validation scope

The final regression passed **46 RTL testbenches and 227 Python test methods,
with no skips**, including existing inference/training tests. The
[saved validation record](measurements/nvfp4-20261109/README.md) includes source
hashes, both simulator checks, raw traces, quantization measurements, and an
earlier simulator-launch failure with its follow-up checks.

Run the packed matrix demo and the optional checkpoint projection from the
repository root:

```sh
python3 scripts/run_nvfp4_demo.py
python3 scripts/run_nvfp4_demo.py \
  --checkpoint models/character_transformer/checkpoint.json --prompt 'ROMEO:'
```

The default demo computes A[4,32] @ B[32,8]: two scale groups along K and two
output tiles. It compares serial copying with double buffering using the same
quantized inputs and endpoint delay of two additional cycles.

| Experiment | Serial cycles | Double-buffered cycles | Relative L2 quantization error |
| --- | ---: | ---: | ---: |
| Deterministic 4×32 by 32×8 matrix | 2,350 | 1,656 | 13.65% |
| Checkpoint query projection, 4×64 by 64×8 | 4,380 | 2,798 | 9.18% |

The matrix schedule saves 694 cycles (29.5%); the projection saves 1,582 cycles
(36.1%). These compare schedules for the same NVFP4 kernel, not NVFP4 against
FP32 throughput. Traces count 1,024 and 2,048 narrow MACs respectively, with zero
launches of the tensor array's FP32 multipliers. SIMD `FMUL` still applies the
tensor-wide factors. Every output word matches the quantized arithmetic oracle;
the error column measures the separate loss from quantizing original FP32
inputs. It does not describe a hardware mismatch.

The projection uses checkpoint SHA-256
`43f211daad7c733d69c34f1d2588dace14de4504a46cf785ec64e6722053622e` and prompt
`ROMEO:`. It computes eight query channels for four token positions in the
first block, before adding bias. Embedding and normalization prepare inputs
on the host; the entire matrix product, tensor epilogue and output stores run
on the RTL. Its weight tensor scale uses all first-block query weights, while
the test itself uses eight columns. This is one projection slice, not a
measurement of language quality or a complete migrated forward pass.

Artifacts are in `build/kernels/nvfp4/{matrix,projection}/`: original/quantized
inputs, both assembly schedules, complete shared/global images, cycle traces,
and comparison reports. `--latency N` changes the external endpoint delay.

Verification covers every element/scale encoding, conversion midpoints,
matrix products with independent FP32 references, nonzero accumulators,
changing scale groups, packed strides and orientation, failed jobs, and
shared-buffer reuse. The complete selected data path uses global-memory
copies, shared staging, persistent tensor accumulation, completion tickets,
the FP32 scale epilogue, and global output stores.

Existing FP32 kernels and checkpoint inference remain available with their
original arithmetic. The [model integration](NVFP4_INFERENCE.md) adds an opt-in
native NVFP4 path for linear layers and measures model quantization error
separately from exact RTL agreement. Physical synthesis, fit, timing, and model
migration onto the global-memory interface remain separate milestones.
