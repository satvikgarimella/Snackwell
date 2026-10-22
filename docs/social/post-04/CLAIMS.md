# Claims checked for build log 04

The implementations below were inspected and the tests rerun before these
visuals were made. Counts and cycles come from the captured run, not estimates.

The opening image summarizes three implemented workload families. Its vector
example uses four captured outputs, its tiled grid depicts an 8×8 matrix split
into 4×4 tiles, and its lower-triangular attention grid is a causal mask. The
panels are independent examples and do not imply a fixed hardware pipeline.

| Claim | Implementation | Verification |
| --- | --- | --- |
| Shared-memory accesses to the same bank finish over multiple cycles | [`shared_memory.sv`](../../../rtl/memory/shared_memory.sv): captured request, pending lanes, one lane per bank per service cycle; [`multiwarp_execution_core.sv`](../../../rtl/sm/multiwarp_execution_core.sv): completion and writeback | [`tb_shared_memory.sv`](../../../tb/tb_shared_memory.sv), [`tb_snackwell_sm_memory.sv`](../../../tb/tb_snackwell_sm_memory.sv) |
| Readable assembly becomes instructions fetched by the SM | [`assemble.py`](../../../scripts/assemble.py), [`instruction_memory.sv`](../../../rtl/frontend/instruction_memory.sv), [`tb_snackwell_kernel.sv`](../../../tb/tb_snackwell_kernel.sv) | [`test_assembler.py`](../../../tests/test_assembler.py), [`test_kernels.py`](../../../tests/test_kernels.py) |
| Vector addition/multiplication and both tensor/SIMD 4×4 matmul run as kernels | [`kernels`](../../../kernels), [`run_kernel.py`](../../../scripts/run_kernel.py) | [`test_kernels.py`](../../../tests/test_kernels.py), [`test_matmul_kernels.py`](../../../tests/test_matmul_kernels.py), [workload results](code/evidence/workload-results.json) |
| Larger matrices use repeated 4×4 TMUL instructions | `tiled_matmul` in [`inference_kernels.py`](../../../snackwell/kernels/inference.py): load tiles, TMUL, FADD partial output, store, advance pointers | [`test_inference_kernels.py`](../../../tests/test_inference_kernels.py): identity/ones/random tests; [8×8 result](code/evidence/tiled_matmul-result.json) records 8 TMULs; [16×16 result](code/evidence/tiled_matmul_16x16-result.json) records 64 |
| LayerNorm, GELU, ReLU, softmax, reductions, bias/residual addition and linear are implemented | [`inference_kernels.py`](../../../snackwell/kernels/inference.py), readable generated [`kernels/nn`](../../../kernels/nn) examples | Independent [`inference_reference.py`](../../../snackwell/reference/inference.py); [Python test log](code/evidence/python-tests.txt) includes numerical/shape/edge/guard tests |
| Single-head attention runs in four GPU program launches | `build_case('attention')` and `scale_mask` in [`inference_kernels.py`](../../../snackwell/kernels/inference.py); [`run_inference.py`](../../../snackwell/inference_runner.py) passes actual output memory to each next launch | [Attention result](code/evidence/attention-result.json): all four stages and final output checked; causal/unmasked and supported shape tests in [`test_inference_kernels.py`](../../../tests/test_inference_kernels.py) |

The four-instruction vector-add excerpt in diagram 01 is read directly from
the kernel; its address setup is deliberately omitted and labeled as such.
Diagram 03 reads the first four operands/results from the fresh simulation
result, showing warp 0 as an example. All sixteen results were checked.
Diagram 06 reads all displayed pass counts from the evidence files.
Diagram 07 summarizes nine implemented operations in six independent cards.
Its integer-valued reductions and ReLU examples illustrate those operations;
the zero-score softmax row gives four equal weights of 0.25. The linear layer
runs matmul and bias addition in two separate launches. These cards do not
depict a single connected model or a full inference run.

## What these results establish

- The real `snackwell_sm` executes these programs in RTL simulation.
- The runner initializes input memory and launches warps; no register operand
  preload substitutes for the kernel's loads.
- The device computes intermediate attention scores, probabilities and output.
  Python generates instructions/constants and independently checks results.
- Every stage's actual dumped memory is loaded into the next simulator launch.
  This is currently host-managed simulation, with a reset between launches.
- New inference kernels use one full warp. Their current mapping avoids
  inter-warp cooperation, so it does not establish barrier support.

## Limits to keep in the post

- Single-head attention starts from supplied Q/K/V. A complete transformer
  block, projections from model inputs, pretrained weights and model predictions
  have not been assembled into one complete model run.
- Shared memory is still 4 KB. Global memory, a persistent device runtime,
  inter-warp barriers, training and full FPGA resource/timing validation remain
  future work.
- Supported sizes are 4, 8 and 16 where memory permits; arbitrary shapes and
  tails are not supported. For example 16-token attention with width 16 does
  not fit this memory.
- GELU uses the tanh approximation. Exponential, reciprocal and reciprocal
  square root are implemented through instruction sequences. Their checks use
  documented tolerances, not claims of correctly rounded transcendental math.
- The memory diagram counts service cycles, not the complete load latency.
  Acceptance, response and register writeback add time.
- Tile labels A00, A01 etc. identify 4×4 submatrices. Each highlighted product
  is a tile multiplication, not one scalar product or one clock cycle.

Raw evidence: [regression](code/evidence/regression.txt),
[Python tests](code/evidence/python-tests.txt), [kernel runs](code/evidence/workloads.txt),
[command metadata](code/evidence/run-metadata.json), [source manifest](code/manifest.json).
