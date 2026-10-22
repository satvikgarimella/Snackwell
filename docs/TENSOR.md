# Snackwell 4x4 FP32 systolic array

The tensor unit computes one row-major 4x4 tile `C = A * B` using
16 physical processing elements (PEs). Each PE holds one C element and performs
four multiply-accumulates in increasing k order. A travels right, B travels down,
and C remains in its PE until the tile completes.

`snackwell_sm` dispatches this unit through `TMUL`, gathers matrix operands from
the warp's register file, and writes the result back through the shared SM
writeback port. The standalone array and word interface are also available.

## Modules

| Module | Responsibility |
| --- | --- |
| `fp32_mac_pe` | Scalar multiplier, scalar adder, local accumulator, retained A/B tokens |
| `systolic_feeder_4x4` | Staggered row/column inputs with valid bits |
| `systolic_array_4x4` | 16 PEs, neighbor wiring, shared step control, full-tile request/result handshake |
| `tensor_core` | Word-at-a-time tile loading and ordered result streaming |
| `sm_tensor_unit` | Warp tag, four-cycle register gather, tile loading, result-row packing, destination reservation |

RTL is under `rtl/execution/tensor/`. Each PE instantiates its own copies of
`fp32_mul_wrapper` and `fp32_addsub`, independently of the SM's SIMD FP unit.
The SM adapter is `rtl/sm/sm_tensor_unit.sv`.

## SM instruction and register layout

`TMUL Rd, Ra, Rb` has opcode `0x0F` in the existing instruction format:

| Bits | Meaning |
| --- | --- |
| 31:26 | `OP_TMUL` (`6'h0F`) |
| 25:22 | First destination register, Rd |
| 21:18 | First A register, Ra |
| 17:14 | First B register, Rb |
| 13:0 | Reserved, must be zero |

Each matrix occupies four consecutive vector registers. Registers select rows;
lanes select columns:

| Register | Lane 0 | Lane 1 | Lane 2 | Lane 3 |
| --- | --- | --- | --- | --- |
| Ra | A[0][0] | A[0][1] | A[0][2] | A[0][3] |
| Ra+1 | A[1][0] | A[1][1] | A[1][2] | A[1][3] |
| Ra+2 | A[2][0] | A[2][1] | A[2][2] | A[2][3] |
| Ra+3 | A[3][0] | A[3][1] | A[3][2] | A[3][3] |

B and C use the same layout starting at Rb and Rd. All elements contain raw
FP32 bits. Bases may be any register from R0 through R12; no alignment is
required. Input and output tiles may overlap partially or completely because
both inputs are gathered before any result write. This computes a fresh `C=A*B`;
the old destination values do not initialize the accumulators.

For example:

```text
TMUL R8, R0, R4    // A in R0..R3, B in R4..R7, C written to R8..R11
FADD R12, R8, R15  // consumer waits for the complete tensor result
EXIT              // waits for all pending writes from this warp
```

The assembler accepts this syntax and emits loadable program images; see
[KERNELS.md](KERNELS.md). A complete load/TMUL/store kernel and a measured
comparison with SIMD FP32 are described in [MATMUL.md](MATMUL.md):

```sh
python3 scripts/run_kernel.py matmul_compare
```

In a testbench, the first instruction is
`{OP_TMUL, 4'd8, 4'd0, 4'd4, 14'd0}` (`32'h3e010000`).

The issue-time active mask controls **output columns**: lane c writes C[r][c]
for each of the four rows only if mask[c] is set. Inactive destination lanes
retain their previous bits. **All lanes of both input matrices are read**,
including inactive lanes, because an active result column can depend on every
input column. Initialize complete input tiles before executing a masked TMUL.
An empty mask still executes and retires a job, with no register bits changed.
The captured output mask remains valid even if the warp branches while the
array runs. A TMUL at a reconvergence PC executes after mask restoration.

An encoding with any base above R12 or a nonzero reserved immediate acts as a
NOP: advance PC, allocate no destinations, perform no memory/register writes.
This matches the SM's existing unsupported-encoding behavior; there is no
illegal-instruction trap in this version. Register ranges never wrap at R15.

## SM dispatch, hazards, and completion

There is one tensor unit shared by all four warps, independent of the SIMD FP
unit. Before issue, every A, B, and C register is checked against both ordinary
pending writes and the tensor destination reservation. On acceptance, the SM
saves the warp ID, three bases, and output mask and reserves all four C rows.

The adapter owns both register read ports for four cycles, reading one A row
and one B row per cycle. All instruction issue stalls during these cycles so
younger writers cannot overwrite an unread source. Existing writebacks continue
to drain. Once the input snapshots are complete, the read ports are released;
independent integer, FP, shared-memory, and control instructions can execute
while the adapter loads the tensor core and the array computes. A younger
instruction may reuse a source register after gathering, provided it is not
also a reserved destination.

The adapter passes its gathered tiles directly to `systolic_array_4x4` and reads
four 128-bit result rows directly from the held output tile. This removes the
former 32 serial input transfers and 16 serial output transfers. The standalone
`tensor_core` word interface remains available and tested separately. Writeback priority is:

1. A completed SIMD FP instruction.
2. A ready tensor result row.
3. A completed shared-memory load.
4. A newly issued integer instruction.

An existing writeback can commit on the same edge that the slot receives the
next result. Producers hold their results while waiting, and integer writers retry
when a completion reserves the slot. Each tensor write carries the saved
warp, destination row, and active mask. The **entire four-register destination
remains reserved until the final row actually commits**, including while debug
writes occupy the register port. Thus ordinary instructions and subsequent
TMULs cannot observe partially written tiles or overtake a tensor destination.
The tensor reservation augments the scalar scoreboard through the issue path's
`additional_dependency` input. Every committed write wakes dependent candidates
for that warp, which recheck the remaining reservations before issuing.

`run_enable=0` pauses new SM issue; an accepted tensor job continues gathering,
computing, and draining. The adapter drives the standalone core's `enable=1`.
Debug register writes pause architectural commits, with completion data held
until that port is free. Debug preload is for unused/retired state: do not edit
live operands or pending destinations. Launch only unused or fully retired
warps; relaunching over in-flight work is not a context switch. `EXIT` waits for
ordinary and tensor writes and the warp's outstanding memory request, including
stores. Reset aborts all work, clears the register file
and reservations, and discards partial result streams and writeback tags.

## Arithmetic

Each job starts with positive zero in every accumulator. For each k from 0 to 3:

```text
product = round_binary32(A[row][k] * B[k][column])
C[row][column] = round_binary32(C[row][column] + product)
```

Both operations round to nearest, ties to even. This is non-fused arithmetic:
rounding happens after multiplication and again after addition. Signed zeros,
subnormals, infinities, and NaNs follow the existing FP arithmetic. Valid numeric
zero participates in multiplication (including `0 * infinity -> NaN`); an invalid
token is a bubble and performs no arithmetic. NaN payloads/signs are not preserved
as an architectural contract.

## Dataflow and timing

At logical step t, the feeder supplies `A[r][t-r]` to the left edge of row r and
`B[t-c][c]` to the top edge of column c. Indices outside 0..3 produce invalid
tokens. After traveling through neighbors, PE (r,c) receives matching operands
with `k = t-r-c`. It therefore does useful work at steps r+c through r+c+3.
Every PE receives four valid pairs, for 64 MACs total, across ten steps (0..9).

Each step has two phases:

1. All PEs capture their current inputs. Valid pairs multiply, then submit the
   rounded product to their adder. Bubble PEs retain both independent input tokens.
2. Once every product has reached its adder, the controller asserts `step_advance`.
   All PEs forward their saved A/B tokens together. The feeder and step index
   advance at the same edge.

The next multiplication can overlap the preceding addition. If an early product
arrives while the adder is busy, the PE holds it. If the preceding addition
finishes on the edge that accepts the next product, the new sum is forwarded
directly into the adder. Additions still occur in k order with separate rounding.
After the tenth transport step, the controller drains all pending additions
before publishing C. The shared step preserves operand alignment across the
array. Ten logical steps still take multiple clocks; no clock gating is used.

`enable=0` pauses step launch and token advancement. It does not abort arithmetic
already started. After `result_valid` is asserted, result consumption proceeds
independently of `enable`.

## `tensor_core` interface

All transfers are sampled on rising clock edges. Producers hold valid and data
until ready is high; a transaction happens only when both valid and ready are 1.

### Load and launch

- Write a word using `load_valid && load_ready`.
- `load_matrix_b=0` selects A; `load_matrix_b=1` selects B.
- `load_index = row*4 + column`, and `load_data` holds raw binary32 bits.
- Load all 16 words of each matrix. Order is arbitrary, and duplicate writes
  replace that word without making any other location valid.
- `start_ready` becomes high only after both complete tiles have been loaded.
- `start_valid && start_ready` launches one multiplication and consumes both
  tiles' loaded-valid masks. Both matrices must be loaded again for the next job.
- A load takes priority over a simultaneous start, so start cannot capture a
  partly updated tile. Hold `start_valid` to retry after the load finishes.
- Once launched, `busy` stays high through computation and result draining.
  Loads and starts are backpressured while busy. A launch may be accepted with
  `enable=0`; it executes once enable is asserted.

### Result stream

The unit asserts `result_valid` with `result_data` and `result_index`. Indices
0..15 stream in row-major order; `result_last` is high on index 15. Each
`result_valid && result_ready` handshake consumes one word. All output fields
remain stable during backpressure. The last word's handshake retires the job
and permits loading the next tile.

The protocol is equivalent to:

```text
write A[0..15] and B[0..15], observing load_ready
submit start_valid until start_ready
read C[0..15], asserting result_ready for each accepted word
```

Synchronous reset aborts computation, clears load-valid masks, drops any partial
result stream, resets the output index, and resets all scalar FP pipelines.

## Lower-level interfaces

`systolic_array_4x4` captures full `a_tile` and `b_tile` buses on
`request_valid && request_ready`. Element (r,c) is at bits
`32*(4*r+c) +: 32`, with (0,0) in the least significant word. The same layout is
used by `result_tile`, which remains stable while `result_valid` waits for
`result_ready`. Only one job can be outstanding. The visible step index and PE
completion mask are useful for tracing the wavefront.

For `fp32_mac_pe`, `step_valid && step_ready` accepts the next pair of tokens.
`step_done` is a **held level**, not a pulse: it means the product has reached the
adder (or the step was a bubble), and stays asserted until `advance` forwards the
tokens. It does not mean the addition has finished. `arithmetic_busy` stays high
through the accumulator update. Early advance is ignored. A PE cannot accept a
new step until its saved tokens have been forwarded. `clear` resets the
accumulator, tokens and both arithmetic units, including operations in flight.

## Verification

```sh
python3 scripts/run_tests.py
```

Tensor-only regression:

```sh
python3 scripts/run_tests.py tb_fp32_mac_pe tb_systolic_array_4x4 tb_tensor_core tb_snackwell_sm_tensor
```

The runner generates `build/tests/tensor_vectors.txt` from `tensor_reference.py`,
using the independent exact-integer binary32 reference. It computes each product
and sum separately, preserving the specified accumulation order and rounding.

- The PE suite checks repeated accumulation, negative values, subnormals, a case
  that differs from fused FMA, bubbles, token retention, early advance, completion
  backpressure, overlapping multiply/add, same-edge sum forwarding, and clear/reset while busy.
- The array suite checks 185 matrix cases / 2,960 output elements, including
  zero/identity/basis matrices, known non-symmetric products, random signed values,
  varied exponents, special values, and random bit patterns. It also checks the
  exact input pair at every PE, 64 valid MACs and ten advances per job, paused
  execution, input capture, held results, repeated jobs, and reset/reuse.
- The word-interface suite repeats those 185 cases through permuted tile loads
  and the result stream. It checks missing-word/duplicate-write handling,
  load/start priority, busy rejection, per-word backpressure, stale-tile rejection,
  and reset partway through result draining.
- The full-SM suite runs the 185 cases through real `TMUL` dispatch and register
  writeback, including exact and partial output aliases. An independent pending
  register model checks incoming/outgoing RAW/WAW hazards, operand snapshots,
  warp tags, masks, row order, and EXIT draining. Directed cases cover every
  input/output register dependency, unaligned and overlapping input bases,
  all 16 output masks with dense inputs, four-warp contention,
  chained tensors, integer/FP/shared-memory consumers, divergent paths and
  reconvergence, FP/tensor/integer/load completion collisions, paused issue,
  per-row debug backpressure, malformed encodings, reset during gather/compute/
  drain, and relaunch of retired warps.

Logs are in `build/tests/<testbench>/test.log`. Simulation checks do not establish
FPGA resource usage or timing closure.

## Current limits

This version supports fixed 4x4 A/B/C tiles, one job at a time, with accumulators
cleared per job. Larger hardware tiles, retained in-array accumulation across jobs, fused MAC,
mixed precision, independently advancing PEs, and pipelined throughput are future
extensions. TMUL uses register tiles; loading/storing them uses the existing
ordinary shared-memory instructions, which now serialize bank conflicts
and retain load destinations until writeback (see [MEMORY.md](MEMORY.md)).
Direct tensor DMA, global memory, memory
coalescing, and block barriers remain later work.

The software `tiled_matmul` kernel now composes multiple TMUL jobs and accumulates
their partial results using FADD and shared memory, supporting square matrices
through 16x16. See [INFERENCE_KERNELS.md](INFERENCE_KERNELS.md).
