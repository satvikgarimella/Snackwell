# A complete matrix-multiplication workload

Two assembly kernels compute the same row-major 4x4 product, `C = A * B`.
Each output is a four-term dot product. Both implementations start the sum
at positive zero and, for k=0,1,2,3, separately round the multiplication and
addition to binary32, round-to-nearest ties-to-even. Neither uses fused FMA.

```sh
python3 scripts/run_kernel.py matmul_compare
```

The command assembles both programs, runs them on the full SM, verifies every
output against `tensor_reference.matmul` (the independent integer-arithmetic
FP32 oracle), and checks that all other memory words remain unchanged.
It also compares the two implementations' results. Finite numbers, infinities,
and signed zeros must match bit for bit; NaNs must match by classification.

The simulator initializes shared memory and launches warps. It does not preload
register operands or calculate the GPU's answers. The assembly performs all
address calculation, loads, arithmetic, and stores.

## Tensor kernel

Source: [matmul_4x4.asm](../kernels/matmul_4x4.asm).

One warp handles the entire tile. All four lanes must be active while loading
inputs. Each lane obtains its column from `TID & 3`, multiplies it by four to
form a byte offset, and loads the corresponding column of every input row:

- R0..R3 contain A, one row per register.
- R4..R7 contain B, one row per register.
- `TMUL R8, R0, R4` computes C into R8..R11.
- Four `ST_SHARED` instructions store those rows at byte offsets 128..191.

For example, lane 2 loads A column 2 from bytes 8,24,40,56 and B column 2 from
72,88,104,120. After TMUL, it stores C column 2 at bytes 136,152,168,184.
The first store cannot consume R8 until all four tensor result rows commit.
`EXIT` waits until the final memory store completes.

The runner launches warp 0 only; the kernel also works on another single warp.
Launching several warps with this program would repeat the same tile and stores,
so that is not the comparison's work mapping.

## SIMD FP32 kernel

Source: [matmul_4x4_simd.asm](../kernels/matmul_4x4_simd.asm).

Four warps handle one tile. Warp r computes output row r, and lane c computes
column c. Each lane obtains its output index with `TID`, then calculates:

- A row address: `4 * (TID & 12)`.
- B column address: `64 + 4 * (TID & 3)`.
- C output address: `128 + 4 * TID`.

Four unrolled groups of load/load/FMUL/FADD compute the dot product. All lanes
in a warp read the same A element for each term. Shared memory currently serves
these as four separate accesses to one bank; it has no same-address broadcast.
The B reads and output stores use four distinct banks.

This program uses four-lane SIMD instructions. The name does not mean there is
one physical scalar FP unit or that each of the 16 outputs runs in isolation.
Only one SIMD FP job can be outstanding globally in the current SM.

## Baseline measurement

Measured with the default input matrices on 2026-10-09:

```text
A = [ 1  2  3  4 ]    B = [16 15 14 13 ]
    [ 5  6  7  8 ]        [12 11 10  9 ]
    [ 9 10 11 12 ]        [ 8  7  6  5 ]
    [13 14 15 16 ]        [ 4  3  2  1 ]

C = [ 80  70  60  50 ]
    [240 214 188 162 ]
    [400 358 316 274 ]
    [560 502 444 386 ]
```

| Measurement | TMUL | SIMD FMUL/FADD |
| --- | ---: | ---: |
| Execution cycles, address setup through EXIT | 385 | 534 |
| Warps launched | 1 | 4 |
| Static program instructions | 20 | 33 |
| Instructions issued across all warps | 20 | 132 |
| FP32 multiplications | 64 | 64 |
| FP32 additions | 64 | 64 |
| Shared-memory words read | 32 | 128 |
| Shared-memory words written | 16 | 16 |
| Memory service cycles | 12 | 84 |
| Memory requests with bank conflicts | 0 | 16 |

The ratio is 534/385 = **1.387x** for these complete kernels and inputs.
The tensor path reuses operands inside its PEs. The SIMD program reloads each
operand for each output, issuing 16 same-address A requests that serialize.
The comparison includes those differences; it does not isolate FP throughput,
establish an optimal SIMD implementation, or predict large-matrix performance.

Cycles start when issue is enabled and end when every launched warp has halted
with all work drained. Reset, host image loading, and warp launch are excluded.
These are simulation cycles. FPGA frequency and wall-clock performance remain
unmeasured. Operand values can change FP latency and therefore the cycle counts.

## Where the tensor cycles go

The tensor instruction occupies its adapter for 335 of the 385 kernel cycles:

| Phase | Cycles |
| --- | ---: |
| Gather rows from the register file | 4 |
| Load 32 operand words into the tensor core | 32 |
| Launch the array | 1 |
| Array computation | 281 |
| Stream 16 result words into four vector writes | 16 |
| Commit the final result row | 1 |

The remaining 50 cycles include address setup, shared-memory loads/stores,
issue waits outside that interval, and EXIT. Array computation consists of a
clear cycle and ten transport steps lasting 22,26,27,28,28,29,29,30,31,30 clocks.
Each step waits for every PE before moving operands. This is why the array's
ten logical steps take hundreds of clocks.

The ten steps perform 1,3,6,10,12,12,10,6,3,1 useful MACs respectively: 64 total
out of 160 possible PE/step slots, or 40%. This measures useful operand pairs
per logical step, not arithmetic throughput. The trace separately measures
the fraction of PE cycles spent in MULTIPLY, ADD_START, or ADD_WAIT: about 34.0%
of the 16 x 281 PE cycles in this example. That includes arithmetic latency;
it is not a percentage of theoretical peak FLOPs.

The largest measured cost is array computation. A subsequent hardware experiment
could reduce FP latency or permit more overlap between steps. A wider tensor
load interface would address the separate 32-cycle input-loading cost. Any such
change needs a new correctness run and a comparison using the same inputs.

## Inspect or reproduce a run

`build/kernels/matmul_compare/comparison.md` is generated from each run's actual
measurements. `result.json` contains both verified results and profiles. The
`matmul_4x4/` and `matmul_4x4_simd/` subdirectories contain:

- `program.hex`, `input.hex`, `output.hex`, and `registers.hex`.
- `simulation.log` and `result.json`.
- `trace.csv`, with one row per execution clock edge.
- `profile.json`, with counts summarized from that trace.

The shared `compile.log` is in the comparison's parent output directory.
For a single kernel, all files are in its own output directory instead.

Custom inputs use two flat JSON arrays of 16 numbers each:

```sh
python3 scripts/run_kernel.py matmul_compare --inputs inputs.json --out build/my-matmul
python3 scripts/run_kernel.py matmul_4x4
python3 scripts/run_kernel.py matmul_4x4_simd
```

## Profiling definitions

The testbench samples signals immediately before each rising edge updates the
RTL. `instruction_accepted` counts architectural instructions; reconvergence
micro-operations are separate. The trace records the selected warp, PC, opcode,
mask, shared-memory/FP state, tensor adapter state, row writeback, array step
events, and per-PE arithmetic/operand-acceptance masks.

Issue-cycle categories partition the full execution interval:

- `instruction_issue`: an architectural instruction was accepted.
- `reconvergence`: an internal reconvergence operation was accepted.
- `dependency_check`: the selected candidate hit a register dependency.
- `resource_check`: the candidate had operands but could not acquire a resource.
- `no_eligible_warp`: no candidate was available to issue.

A dependency-blocked warp may stay unavailable until a writeback wakes it.
Those later cycles can appear under `no_eligible_warp`; `dependency_check` is
not total time waiting for dependencies. These categories describe the scheduler
observation, not a complete attribution of every stall to its root cause.

FP in-flight time, shared-memory service time, and tensor phases can overlap
issue activity and one another, so do not sum unrelated counters.
`tensor_state` uses the adapter's current enum encoding:
0=IDLE, 1=GATHER, 2=LOAD, 3=START, 4=RESULT, 5=COMMIT. While state 4 waits for
array result-valid, it counts as `array_compute`; afterward it is `result_stream`.
`tensor_steps` records launch-to-advance duration, including both edge samples.

## Verification

```sh
python3 -m unittest discover -s scripts/tests -p test_matmul_kernels.py -v
python3 scripts/run_tests.py
```

Both assembly paths run zero/identity matrices, every basis position, signed
random inputs, varied exponents, arbitrary FP32 bits, subnormal/overflow/NaN
cases, separate-rounding examples, and cancellation sensitive to summation order.
Tests check memory guards, every physical warp for TMUL, selected SIMD warps
and lane masks, known operation/traffic counts, custom JSON input, and cleanup
of stale successful reports after a failed CLI rerun. Deliberately missing a
store or changing an arithmetic instruction must fail output verification.

The companion `vector_mul.asm` kernel also runs through the full SM and is
tested with default, special, random, and masked inputs. All current workloads
fit in 4 KB shared memory. The [inference kernel library](INFERENCE_KERNELS.md)
now adds 8x8 and 16x16 tiled matmul and neural-network operations; global memory
remains future work.
