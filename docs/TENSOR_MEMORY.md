# Tensor-memory prototype

Implemented November 7, 2026. This is the first reduced SM100 data-path milestone:
shared-memory operands, persistent tensor storage, and explicit completion and
consumer ownership through the full SM. The initial arithmetic is Snackwell
FP32. Later extensions add [global memory](GLOBAL_MEMORY.md),
[asynchronous tile copies](ASYNC_COPY.md), and [NVFP4 arithmetic](NVFP4.md).
This page retains the original FP32 instruction contract; NVIDIA PTX encodings
are not implemented.

## Run and inspect it

```sh
python3 scripts/run_tensor_demo.py
python3 scripts/run_tests.py tb_sm_tensor_memory_unit tb_snackwell_sm_tensor
python3 -m unittest tests.test_tensor_memory -v
```

The demo executes [this assembly program](../kernels/tensor_memory.asm) for
`A[4,8] × B[8,4]`. Two four-wide matrix operations accumulate into one tensor
tile. The host supplies the program and inputs, then checks the actual RTL
outputs with the independent integer-based FP32 reference. It also checks
status and every unmodified memory word. It never substitutes reference values
for device outputs.

Run files are written to `build/kernels/tensor_memory/`: machine code, input and
output images, register dump, `trace.csv`, `profile.json`, and `result.json`.
The trace includes tensor-memory controller state, operand-row handshakes,
register transfers, and pending ownership. The deterministic demo takes 250
simulated cycles and performs 128 MACs in Icarus Verilog. This is a small kernel
measurement, not a model inference speed claim.

## Resources and arithmetic

There are two addressable slots, `T0` and `T1`. Each holds one row-major 4×4
FP32 tile (64 bytes); both together hold 128 bytes. They are distinct from the
lane register file and from shared memory. Four row-initialization bits prevent
reading an uninitialized row or accumulating into an incomplete tile.

The new controller and legacy `TMUL` share **one physical 4×4 array**. The array
can now start from a captured accumulator tile. Its default mode still starts
at zero, preserving existing callers. One matrix operation can execute at a
time; software ownership of a completed tile can outlive its computation.

For every output element, `TMMA` starts `s = +0`; `TMMA_ACC` starts `s = T[i,j]`.
For `k = 0,1,2,3`, both execute:

```text
p = round32(A[i,k] * B[k,j])
s = round32(s + p)
T[i,j] = s after the last term
```

Multiplication and addition are separate operations, not fused FMA. A second
`TMMA_ACC` continues from the preceding result. This differs from the existing
kernel sequence that computes each `TMUL` tile from zero, then uses `FADD` to
combine the partial tiles. The tests include cancellation that distinguishes
these orders: seed `1e20`, then terms `-1e20, 3, 0, 0`, must produce `3` on the
new path. The old path is retained without changing its contract.

The exact FP32 reference is in [reference/tensor.py](../snackwell/reference/tensor.py).
This numerical contract is a development backend for the new interface, not
evidence of B200 tensor arithmetic equivalence.

## Instructions

All new operations require all four lanes active. Descriptor and ticket
registers must contain the same value in every lane. Row loads/stores transfer
four independent FP32 words, one per lane.

| Instruction | Behavior |
| --- | --- |
| `TALLOC T0` | Reserve a slot for the issuing warp; wait if another warp owns it |
| `TFREE T0` | Owner frees the slot after computation and all consumer releases |
| `TST T0, 0, R4` | Owner initializes row 0 from the four lanes of R4 |
| `TLD R4, T0, 0` | Copy row 0 into a register; consumer must have successfully waited |
| `TMMA T0, R1, R2` | Start a new product using two shared-memory descriptors; overwrite T0 |
| `TMMA_ACC T0, R1, R2` | Start a product using the existing complete T0 as the initial sum |
| `TCOMMIT R3, T0, 5` | Publish this generation to warps selected by mask 5 (warps 0 and 2); return its ticket in every lane of R3 |
| `TWAIT T0, R3` | Wait for that exact generation to be committed and finished, then enable this consumer's reads |
| `TRELEASE T0` | End this consumer's access after its last row transfer |
| `TSTATUS R8` | Read this warp's sticky error bits, cleared by reset |

Before the first matrix operation, the owner may also read rows it initialized
with `TST`. Reinitializing a released tile with `TST` invalidates its previous
job's visibility. `TMMA_ACC` requires all four rows initialized.

The [assembler](../snackwell/assembler.py) validates slot names, row indices,
participant masks, and register operands. Opcodes `0x13` through `0x1c` follow
the table's order. All use the existing 32-bit instruction fields. The immediate
contains the slot in bit 0; row instructions use bits 2:1 for the row, and
`TCOMMIT` uses bits 4:1 for its participant mask. Other immediate bits and unused
register fields must be zero. `TST` uses rs1; `TWAIT` uses rs1; matrix operations
use rs1/rs2; `TLD`, `TCOMMIT`, and `TSTATUS` use rd.

## Shared-memory descriptors

Each descriptor register contains:

| Bits | Meaning |
| --- | --- |
| 11:0 | Byte address of the first element |
| 23:12 | Byte stride between rows |
| 31:24 | Reserved, must be zero |

Base and stride must be multiples of four, stride must be at least 16, and the
last byte of the fourth row must be below 4096. Bounds are checked before any
memory request; invalid descriptors cannot wrap into shared memory.

The controller captures both descriptors at issue, waits for the shared-memory
port to be idle, then reads four A rows and four B rows. It locks ordinary LSU
access for that entire gather, preserving a consistent operand snapshot. Once
all eight rows are captured, ordinary shared-memory access resumes while the
array computes. Integer and SIMD FP instructions may progress during the
gather and computation, subject to their ordinary dependencies/resources.

This coarse shared-memory lock is an intentional first-version simplification.
There is no copy/computation overlap within this controller and no global-memory
fetch yet. Descriptor-based shared-memory reads must not be labeled TMA.

## Ownership and completion

Allocation belongs to a warp. Only the owner may initialize, issue matrix work,
commit, or free that slot. Other warps may consume a result if named in the
commit mask. For this prototype, the resident warps belong to one cooperating
launch; there is no general CTA scheduler or cross-SM synchronization.

Each accepted matrix job increments a per-slot 32-bit generation. Freeing and
reallocating a slot preserves that counter. Wraparound is rejected, requiring
reset before another generation; reset invalidates all software-held tickets.
Waiting for an older generation cannot publish a newer result. A future ticket
waits, including when the producer has not allocated or issued yet.

`TCOMMIT` reserves consumer leases immediately and may run while arithmetic is
still in flight. A wait succeeds only after the matching generation has both
been committed and fully written into tensor memory. It does not accept a
stale completion pulse. The owner cannot overwrite, accumulate, or free while
any named consumer still holds a lease. Every named warp must wait and release;
naming an inactive/exited warp, or waiting for work never submitted, is a program
error that can prevent progress. The kernel runner's cycle limit detects this.

A row transfer snapshots its data and writeback tag. Backpressure retains that
snapshot. Releasing after the transfer has entered the SM writeback pipeline is
safe: later tile reuse cannot change the copied data. Standard register
scoreboarding prevents an instruction from consuming it before writeback.
`EXIT` waits for this warp's allocations, consumer leases, and register results.

This is a custom per-tile commit/wait/lease protocol. It combines ordering and
ownership for the supported path; it is not a general PTX `mbarrier` implementation.

## Error behavior

An invalid command retires without memory or tile changes and records a sticky
per-warp error. Failed register-returning commands produce zero in active lanes
so the scoreboard still retires. `TSTATUS` itself requires a full active warp.

| Bit | Error |
| --- | --- |
| 0 | Invalid encoding, partial collective, or nonuniform descriptor/ticket |
| 1 | Invalid ownership or consumer access |
| 2 | Uninitialized row/tile, missing completion wait, or invalid commit state |
| 3 | Invalid descriptor alignment, bounds, stride, or reserved bits |
| 4 | Zero, stale, or invalidated completion ticket |
| 5 | Generation counter exhausted |

Resource waits are separate: occupied allocation, array/memory contention,
unfinished work, held consumer leases, or writeback backpressure retry through
the scheduler without relying on a register writeback to wake the warp.
These status bits do not automatically set the external device's transport
fault bit; programs must inspect `TSTATUS`.

## Verification and remaining work

Validation completed with **40 RTL benches and 195 distinct Python test methods
passing across the initial and follow-up runs**, with no unresolved skips.
The initial Python environment lacked PyTorch and could not bind a loopback
HTTP socket; the affected checks passed in the existing model environment with
loopback access. The [saved evidence](measurements/tensor-memory-20261107/README.md)
preserves that initial failure as well as the successful follow-ups. Ruff,
generated-example checks, and Verilator structural lint also passed.

[The unit bench](../tb/tb_sm_tensor_memory_unit.sv) exercises allocation
contention, delayed requests/responses, held writeback, initial accumulator
transport, multiple consumers, invalidated tickets, and reset during both
operand gathering and computation. Its array interface is a stub; it does not
claim to test arithmetic.

[Full-SM programs](../tests/test_tensor_memory.py) execute the real array and
cover random tiled products, nonzero initial sums, rounding order, two-warp
cooperation, generation reuse, early reads, malformed/partial commands,
legacy-array contention, independent FP progress, and operand snapshot safety.
The same demo also matches memory, registers, and cycles across Icarus,
Verilator, and the persistent simulator, and matches memory/cycles through the
public device bus with simulation initialization disabled.
The existing tensor benches continue to validate legacy `TMUL`.

Global memory, asynchronous copies/double buffering, and the finite NVFP4 mode
are now implemented in the linked extensions. Mapping full model kernels onto
this path, evaluating quantized-model quality, and updated synthesis/physical
fit remain separate work.
Existing model kernels still execute their original validated FP32 path.
