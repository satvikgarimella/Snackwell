# Snackwell

Snackwell is a small GPU written in SystemVerilog, with four resident warps,
four lanes per warp, integer and FP32 execution, shared memory, and a 4×4
FP32 systolic tensor array. Its custom assembly programs run through the full
SM in RTL simulation and have their output memory checked against independent
references.

The current kernel library includes vector addition/multiplication, tensor and
SIMD 4×4 matrix multiplication, tiled matrix multiplication through 16×16,
bias/residual addition, ReLU, GELU, reductions, LayerNorm, softmax, linear, and
single-head scaled dot-product attention.

## Run

Install Python 3 and Icarus Verilog (`iverilog` and `vvp`), then run from the
repository root. The RTL and kernel test tools use the Python standard library.

```sh
python3 scripts/run_tests.py
python3 scripts/run_kernel.py vector_add
python3 scripts/run_kernel.py tiled_matmul --size 16
python3 scripts/run_kernel.py attention --rows 8 --cols 8
python3 scripts/run_kernel.py all
```

The full test command runs RTL testbenches followed by the Python assembler
and kernel tests. The kernel runner saves assembly, memory images, results,
cycle traces and profiles under `build/kernels/`. It exits nonzero on a failure.

```sh
python3 scripts/assemble.py kernels/vector_add.asm -o build/vector_add.hex
python3 scripts/run_kernel.py matmul_compare
```

## Design and programs

- [FP32 execution](docs/FP32.md)
- [Tensor array and TMUL](docs/TENSOR.md)
- [Shared memory](docs/MEMORY.md)
- [First kernels and assembler](docs/KERNELS.md)
- [Tensor versus SIMD matmul](docs/MATMUL.md)
- [Inference kernel library, input formats and numerical tolerances](docs/INFERENCE_KERNELS.md)
- [Editable build-log diagrams and captured verification results](post%204/README.md)

Memory is currently 4 KB shared memory and a 1,024-instruction program ROM.
The new inference kernels support dimensions 4, 8 and 16 where memory permits.
Linear and attention use multiple launches; the simulation runner transfers
the actual memory image between launches. Arithmetic executes on the SM.

These are simulation results. A complete pretrained-model forward pass,
global memory, a persistent hardware launch/data-transfer runtime, cooperating
warp barriers, training, and full FPGA resource/timing validation remain future
work. Older standalone teaching cores retain a smaller instruction subset;
the kernel runner targets `snackwell_sm`.

The FP32 multiplier retains Jonathan P. Dawson's copyright notice
in [`multiplier.v`](rtl/execution/fp32/multiplier.v).
