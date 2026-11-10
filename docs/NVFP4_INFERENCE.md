# Native NVFP4 model inference

Updated November 9, 2026.

`--precision nvfp4` runs Q/K/V, attention-output, feed-forward, and vocabulary
linear projections through `TMMA_NVFP4` / `TMMA_NVFP4_ACC`. The checkpoint is
unchanged. FP32 remains the default. Attention QK/PV, embeddings, LayerNorm,
GELU, softmax, residuals, bias and token selection still execute in FP32 on the
SM. This is mixed-precision inference, not an entirely four-bit transformer.

```sh
python3 scripts/run_sentence.py --precision nvfp4 --interactive
python3 scripts/run_model.py --precision nvfp4 --prompt 'ROMEO:' --tokens 4 --decode greedy
python3 scripts/compare_nvfp4_model.py --cases 8
```

The sentence demo accepts, for example:

```text
The ball is in the kitchen. The book is in the garden. Where is the ball?
```

Both modes produced ` The ball is in the kitchen.` in the recorded comparison.
The new option is also available in the model runner's interactive and streaming
terminal modes. Its existing browser and exported PyTorch evaluation modes
remain FP32-only; use the paired comparison command for NVFP4 quality checks.

## Execution and quantization

Policy ID: `nvfp4-linear-row-v1`.

- Host quantization uses finite E2M1 values and UE4M3 scales over sixteen K
  elements. K is zero-padded to a multiple of sixteen.
- Each complete weight matrix has one FP32 factor. Each activation row is
  quantized independently with its own FP32 factor. Row-local activation
  factors preserve causality and make scales independent of prompt batching,
  padding, output tiling, and whether a row came from prefill or cached decode.
  This is Snackwell's declared quantization recipe, not a claim to reproduce
  Transformer Engine's complete NVFP4 recipe.
- The host packs values; it does not perform the executed matrix product.
  Shared-memory descriptors feed the native narrow multiplier path. Tensor
  memory retains the FP32 sum across groups of sixteen.
- Launches handle four rows, up to sixteen columns and up to 128 K elements.
  A one-row decode pads three unused rows. Larger K reductions export raw
  unscaled accumulators and restore them with `TST` in the next launch. The
  final launch alone multiplies the two FP32 factors, scales the sum, and adds
  bias using actual SIMD instructions.
- Q/K/V remain separate projections in NVFP4 mode. Combining their weights
  would choose a different tensor factor and change the quantized model.
- KV state contains raw FP32 K/V outputs. Its identity includes the checkpoint,
  prefix, RTL build, and versioned precision policy. Mixing FP32 and NVFP4
  cache histories is rejected; a failed run does not commit cache updates.

The measurements below use the original host-to-shared transport. An optional
[`--memory global` path](GLOBAL_MODEL.md) now routes every inference kernel
through external memory, using asynchronous shared copies for NVFP4 operands.
Both reset the SM between launches. The checkpoint and KV cache still occupy
FP32 host storage; neither option establishes FPGA capacity, fit, timing or
on-board inference.

## Correctness versus quality

Each native launch is checked against an independent integer binary32 oracle,
including packed operand decoding, accumulation order, tensor factors, bias,
memory guards and tensor status. Icarus and Verilator produce identical results
and cycle counts in the model fixture. Traces confirm the native opcodes are
issued in linear launches, without legacy `TMUL` instructions.

An independent numerical transformer starts from checkpoint weights and token
IDs. It implements the documented FP32 approximation/reduction order so small
LayerNorm or GELU rounding differences cannot change a later quantization code.
Every block, final normalization, logits, and output probabilities must match
that model **bit for bit**. The reference never supplies executed activations.
The original FP32 checks and tolerances are unchanged.

Quality evaluation compares the two modes on identical input prefixes. Sentence
targets supply the next input token only in the teacher-forced comparison;
separate greedy generation lets each mode use its own predicted tokens and
scores the final answer. A hardware verification pass does not imply answer
correctness. The FP32 baseline also uses its original four-term reduction tree,
whereas NVFP4 carries one sequential accumulator; both contracts are checked.

## Initial measurements

Same RTL source fingerprint for both modes:
`e96b19fc9a94aa79c246795f7697132776a59f876764f8c6df3943634421a5c0`.
Complete reports are in [the measurement folder](measurements/nvfp4-inference-20261109/README.md).

| Measurement | Compact sentence model | Character model |
| --- | ---: | ---: |
| Cases | 8 reserved test questions, seed 20261109 | `ROMEO:` and `My lord,`, 4 new characters each |
| Paired next-token checks | 56 | 8 |
| FP32 / NVFP4 top-1 agreement | 56/56 | 6/8 |
| Exact autonomous answers | FP32 8/8; NVFP4 8/8 | No answer-quality target |
| Mean probability total variation | 0.00206 | 0.19116 |
| Mean relative L2 logit difference | 11.40% | 20.54% |
| FP32 / NVFP4 simulated cycles | 28,968,765 / 27,453,019 | 8,173,230 / 7,036,781 |
| FP32 / NVFP4 verified wall seconds | 15.54 / 23.60 | 4.55 / 7.53 |

The sentence model's mean target-token negative log likelihood increased from
0.000164 to 0.002316 despite retaining these eight answers. That is a measurable
loss of confidence. Eight questions do not establish accuracy across the full
reserved split or the supported sentence format.

Character continuations changed from `\nThe` to `\nMar` after `ROMEO:`, and from
` and` to `\nAnd` after `My lord,`. These short runs establish numerical drift
and successful execution, not acceptable language quality.

Cycle totals cover identical prefixes and exclude host work. Wall times include
host quantization, transfers, both verification layers, and other system load;
they are not isolated throughput benchmarks. The sentence run saved 5.2% of
simulated cycles but took longer end to end. No NVFP4 speedup on physical hardware
is claimed.

## Remaining work

Broader answer-quality evaluation and quantization calibration come before
considering an NVFP4 default. Global-memory transfers and
[persistent device allocation](RESIDENT_MEMORY.md) are implemented; direct pointer
access has removed the resident runtime's gather/scatter launches. Decode currently leaves three tensor rows
unused; scheduling and host overhead deserve profiling before optimization.
Training remains on its existing FP32 path. FPGA synthesis and timing checks
are still required for the updated RTL.
