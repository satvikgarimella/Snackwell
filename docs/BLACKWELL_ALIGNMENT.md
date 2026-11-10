# Snackwell: the scaled-down Blackwell target

Updated November 9, 2026. This is an architecture alignment plan.

The first [tensor-memory prototype](TENSOR_MEMORY.md) is now implemented:
two tensor tiles, shared-memory operand descriptors, overwrite/accumulate,
generation-tagged commit/wait, and consumer release. It shares the existing
array and uses an explicitly documented FP32 backend. The instruction contract
and new tests are separate from the historical results in the report. The
[global-memory interface](GLOBAL_MEMORY.md) now supplies kernel-issued loads and
stores, delayed external responses, and a global/shared/tensor/global demo.
[Asynchronous rectangular copies](ASYNC_COPY.md), protected shared regions, and
double buffering are now implemented and compared against a serial schedule.
The [finite-scale NVFP4 path](NVFP4.md) now implements E2M1 operands, UE4M3
groups of sixteen, narrow-product hardware and FP32 accumulation. A checked
kernel spans multiple scale groups and output tiles; a checkpoint query
projection measures quantization error. [Mixed-precision model inference](NVFP4_INFERENCE.md)
now executes native NVFP4 linear layers with independent full-model checks and
an initial paired quality evaluation. [Global-memory model launches](GLOBAL_MODEL.md)
now route inference kernels through the external port, with async tile feeding
for NVFP4. [Resident model memory](RESIDENT_MEMORY.md) now keeps weights,
intermediates and KV state in external RAM between launches. Generated kernels
follow physical pointers directly without layout-copy launches. The global LSU
now [broadcasts same-address loads](measurements/load-broadcast-20261109/README.md)
within an instruction, reducing repeated external traffic. Updated physical fit
and a physical memory controller remain open.

The [resident compiler optimizations](measurements/resident-compiler-20261109/README.md)
reduce lookup and store instructions and keep compiler literals in shared RAM.
They use the existing ISA and registers; model tensors still live in external RAM.

## Goal and reference

Snackwell's goal is to build a small, inspectable implementation of Blackwell's
compute architecture. Blackwell is the architectural reference for the project,
not just the inspiration for its name. The existing SM, software stack, and
verification infrastructure are the foundation for that goal.

The user confirmed **B200 / datacenter Blackwell SM100** as the reference on
November 7, 2026. RTX graphics and datacenter compute variants must not be mixed
into this specification. In particular, NVIDIA's
[CUTLASS implementation](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py)
distinguishes the datacenter `tcgen05`/TMEM path from consumer SM120.

Scaling down needs an explicit contract: which behaviors are preserved, which
capacities are reduced, and which mechanisms are deferred. Fewer lanes and
smaller tiles can be deliberate reductions. A different tensor-memory interface
or arithmetic contract must be recorded as an architectural difference.
The current custom ISA also means NVIDIA binary/CUDA compatibility is not
implemented. Matching documented behavior and reproducing proprietary circuit
internals are separate claims.

The recommendation below is a **single-SM, dense-matrix subset**, initially
modeling one cooperating thread block (`cta_group::1`). It is a concrete first
implementation target, not a complete B200 feature list. NVIDIA documents the
reference behavior; the reduced capacities, custom instructions, and hardware
organization below are Snackwell design choices.

## Baseline before the tensor-memory prototype versus target

| Area | What exists in Snackwell | Proposed next implementation |
| --- | --- | --- |
| SM execution | Four warps of four lanes, FP32/integer units, scoreboarding, divergence and reconvergence | Retain the small configuration; document its execution/control simplifications against the reference |
| Matrix instruction | `TMUL` computes a fresh FP32 `C=A×B`; inputs and outputs use vector registers | Add shared-memory operand descriptors and matrix accumulation into tensor memory; retain legacy `TMUL` |
| Tensor storage | Private PE sums and a held result tile; no instruction-addressable tensor memory | Add a small tensor-memory space with explicit access and ownership rules |
| Completion | Independent work can overlap `TMUL`; consumers wait on destination registers | Add completion barriers, visibility rules, and protection against premature buffer reuse |
| Data movement | Shared-memory loads/stores; host moves full-model data between launches | Add a device-visible memory interface, then a small asynchronous tile-copy engine and staged shared buffers |
| Warp cooperation | Disjoint matrix tiles avoid cross-warp dependencies | Add synchronization for cooperating producer/consumer warps |
| Tensor precision | FP32 multiplication and addition round separately | Add an exact supported operand format with FP32 accumulation; use NVFP4 as the proposed block-scaled target |

Blackwell's SM organization and memory hierarchy are documented in NVIDIA's
[tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
The SM100 matrix interface uses TMEM for accumulators, supports memory-sourced
operands, and coordinates asynchronous matrix work with pipeline barriers.
These are concrete behaviors to model at a smaller scale.
[SM100 matrix programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html).

Its block-scaled arithmetic associates scale factors with groups along the
matrix inner dimension. MXFP4 and NVFP4 have different scaling specifications;
storing arbitrary four-bit values would not reproduce either format.
[Blackwell data types and scaling](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).

The table's proposed mechanisms are Snackwell design work, not a claim that
existing RTL already implements NVIDIA's interfaces. Asynchronous tile movement
and warp synchronization also occur in earlier GPU architectures; their role
here is to support the selected Blackwell compute path.

## What the audit corrected

- **Memory-sourced operands are part of the change.** Adding a buffer for sums
  while continuing to gather both matrices from registers would leave the
  selected SM100 data path incomplete. Start with both operands in shared memory
  and the accumulator in tensor memory. Optional operand-A tensor-memory sourcing
  can follow later.
- **Completion and visibility belong in the first tensor interface.** A job
  being accepted, its arithmetic finishing, and its result being safe to consume
  are distinct events. The interface must define them before other warps use it.
- **A generic wait is insufficient as a specification.** In the reference,
  matrix operations use commit/barrier completion; tensor-memory loads and stores
  have a separate wait mechanism. The reduced ISA may combine operations, but
  must state which ordering guarantees each instruction provides.
  [PTX 8.7, section 9.7.16.5.2.1](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html).
- **The old arithmetic must remain explicit.** Current tiled kernels compute
  each four-term dot product from zero and then add it to a running register
  value. Carrying a PE accumulator continuously across tiles changes the
  grouping of FP32 additions and can change results. The new operation needs its
  own numerical contract; existing `TMUL` cannot silently acquire new semantics.
- **Warp control has a separate fidelity gap.** Snackwell has one PC per warp
  and a reconvergence stack. Modern NVIDIA execution tracks state per thread.
  Keeping the current control for uniform cooperating warps is a deliberate
  restriction, not merely a smaller warp size.
  [CUDA SIMT architecture](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#simt-architecture).

## Implementation order and acceptance evidence

### 1. Specify and build the reduced tensor interface

Keep the existing array and FP32 path as the regression baseline. Introduce
separate tensor-memory operations, an operand descriptor for shared memory,
and explicit overwrite/accumulate modes. Define buffer allocation, ownership,
release, valid addresses, and the behavior when capacity is unavailable.
Private PE registers alone do not satisfy this interface.

Design completion alongside storage: distinguish submission, completion
notification, result transfer, and buffer reuse. Define barrier participants
and generation/phase reuse. Start with one outstanding matrix operation and
uniform warp participation; increase concurrency only after that contract works.
These are proposed reductions, not B200 limits.

Required evidence:

- Nonzero initial accumulators, overwrite mode, and accumulation across multiple
  inner-dimension tiles match an independent reference for the declared math.
- Shared-memory descriptor addressing and tensor-memory access work through the
  full SM, assembler, and register writeback path.
- Delayed completion and backpressure cannot expose a partial result, overwrite
  live operands, or reuse a live accumulator. Tests cover the specified handling
  of invalid accesses, allocation exhaustion, and reset during a job.
- Repeated barrier phases cannot consume an old completion. Another eligible
  warp can progress while the requester waits.
- Existing `TMUL` and FP32 behavior remains unchanged. A separately named FP32
  prototype may exercise the new interface, but does not establish SM100 tensor
  arithmetic compatibility.

### 2. Supply tiles through a device memory path

Add a device-issued memory request/response interface and simulated backing
memory. Then add an asynchronous rectangular tile copy with explicit source,
destination, dimensions, and strides, followed by two shared-memory staging
buffers. This models a restricted TMA-style use case; arbitrary tensor maps,
swizzles, multicast, and a full cache hierarchy are separate extensions.
The existing host port only accepts memory access while execution is idle.

Required evidence:

- Requests survive variable response latency and backpressure without lost or
  duplicate transfers; bounds and alignment follow a defined contract.
- A consumer cannot use a partially copied tile. A producer cannot overwrite
  a shared buffer until its consumer releases it.
- A tiled kernel visibly overlaps the next copy with current matrix work.
  Record busy/stall cycles and compare with a serial version under the same
  memory latency. The testbench supplies memory responses, not tensor results.

### 3. Implement a specific Blackwell arithmetic mode

The selected narrow-precision target is **NVFP4**: signed E2M1 values with UE4M3
block scales for groups of 16 along the inner dimension, and FP32 accumulation.
MXFP4 instead uses UE8M0 scales and groups of 32. They are different contracts.
[CUTLASS type definitions](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/fundamental_types.html).

Freeze encoding, packing, scale placement, conversion, exceptional values, and
the accumulation/rounding rule before RTL. With four-wide internal steps, a
16-element scale group spans four steps; shrinking it to four would create a
different format. An FP32 development backend is useful for debugging, but must
remain labeled as such. The systolic implementation is our hardware choice;
public interfaces do not establish B200's internal PE wiring.

Required evidence: exhaustive element decoding, conversion boundaries, changing
scale groups, independent matrix references, and measured model error against
the existing FP32 path. Bit-exact agreement with Snackwell's own reference is
distinct from numerical equivalence to a B200. Quantization or a new accumulation
order may require model adjustments; keeping a checkpoint is not proof of parity.

### 4. Demonstrate the complete selected path, then check fit

Run one dense tiled matrix kernel through device memory, shared staging,
tensor-memory accumulation, completion, and output storage. Exercise at least
two scale groups and multiple output tiles. Verify both numerical output and
the ordering/overlap trace. Map a transformer operation onto the same path,
then rerun affected kernel, model, training, and RTL regressions.

Synthesize and fit the extended hardware. Existing synthesis and resource
estimates only describe the current core. Physical fit and timing need Quartus
on a permitted Windows/Linux machine and the confirmed FPGA part. A finished
baseline fit would still not establish that the extensions fit.

## Scope beyond the first implementation

Multiple SMs, CTA-pair matrix operations, distributed shared memory, sparse
execution, more tensor formats, caches, and interconnects are later fidelity
extensions. Full independent-thread scheduling is also unresolved. Choosing
the single-block, uniform-warp subset avoids requiring it for the first demo;
it does not make the existing divergence machinery equivalent to SM100.

One SM, four lanes, and small tiles are intended scale reductions. Custom ISA,
restricted scheduling, simplified tensor layouts, and a limited copy descriptor
are semantic differences that must remain visible in the report. Completing
this plan demonstrates a defined portion of Blackwell compute behavior, not
every B200 feature or NVIDIA software compatibility.

The first prototype implements step 1's reduced data path and completion/lease
protocol. It does not implement general PTX barriers.
Step 2's device-memory interface and reduced asynchronous copy engine are
implemented and tested, including shared-buffer ownership and double buffering.
The global-memory demo retains explicit register-mediated copies; a separate
async-copy demo measures overlapped tile supply. Step 3's finite NVFP4 subset
is implemented with a declared arithmetic contract and independent checks.
The NVFP4 demo exercises step 4's complete memory/tensor path and a real
checkpoint projection slice. Opt-in inference now runs model linear layers
on native NVFP4, with initial paired quality results documented separately.
Global-memory kernel launches are now implemented; persistent device allocation,
broader quality evaluation, and physical synthesis/fit remain. There is no validated same-day completion
estimate for the entire plan or FPGA deployment.

## Source anchors

- [ISA](../rtl/snackwell_pkg.sv): legacy arithmetic/shared operations, tensor-memory
  operations and completion tickets, global loads/stores/status, and rectangular
  copy/wait/release/status instructions.
- [Tensor adapter](../rtl/sm/sm_tensor_unit.sv): register gather, array launch,
  row writeback, and final commit.
- [Processing element](../rtl/execution/tensor/fp32_mac_pe.sv): the current
  separately rounded FP32 multiply/add contract.
- [Tiled matrix generator](../snackwell/kernels/inference.py): `TMUL` followed
  by `FADD` into R8-R11, determining the current reduction grouping.
- [Warp state](../rtl/sm/warp_state_table.sv): one program counter per warp.
- [SM execution](../rtl/sm/multiwarp_execution_core.sv) and
  [shared memory](../rtl/memory/shared_memory.sv): existing resource and memory
  coordination to extend.

The [engineering report](REPORT.md) records completed work and measured results.
The linked prototype documents the new RTL and kernel checks. No new physical
FPGA fit or timing result is claimed.
