# Snackwell kernels

The runner executes assembly on the actual `snackwell_sm` in RTL simulation,
checks the memory results, and records a cycle trace. The initial kernels are:

| Command argument | Operation | Work mapping |
| --- | --- | --- |
| `vector_add` | 16-element FP32 addition | Four warps, one element per lane |
| `vector_mul` | 16-element FP32 multiplication | Four warps, one element per lane |
| `matmul_4x4` | 4x4 matrix multiplication using `TMUL` | One full warp handles the tile |
| `matmul_4x4_simd` | Same matrix multiplication using `FMUL`/`FADD` | Four warps, one output row per warp |
| `matmul_compare` | Verify and profile both matrix kernels | Same inputs and output addresses |

The [inference kernel library](INFERENCE_KERNELS.md) adds tiled matmul through
16x16, bias/residual addition, ReLU/GELU, row reductions, LayerNorm, softmax,
linear, and single-head attention. Run the full workload set with:

```sh
python3 scripts/run_kernel.py all
```

```sh
python3 scripts/run_kernel.py vector_mul
python3 scripts/run_kernel.py matmul_4x4
python3 scripts/run_kernel.py matmul_compare
```

The initial kernels above use the same memory layout: A at byte 0, B at 64, and C at 128.
Each array contains 16 FP32 values. Matrices are flat arrays in row-major order.
`--inputs inputs.json` accepts `{"a": [...], "b": [...]}` with 16 numbers each;
`--out path` changes the output directory. The default is
`build/kernels/<command argument>/`. These are fixed-size kernels with no
general kernel-argument interface yet.

The default vectors use A=1..16 and B=10,20,..160. Default matrices use A=1..16
and B=16..1. See [MATMUL.md](MATMUL.md) for the matrix-kernel walkthrough,
the measured comparison, and definitions of the profiling counters.

## First kernel walkthrough: vector addition

A kernel is a program executed by the SM. Within a warp, all active lanes
execute the same instruction using their own register values. Different warps
have their own program counters and take turns using the execution units.

The first kernel computes `C[i] = A[i] + B[i]` for 16 FP32 values. Four warps
of four lanes run the same ten-instruction program. Each lane handles one pair
of inputs and writes one result. This uses the SIMD FP adder.

## Follow one element through the GPU

1. The simulation host puts A and B in shared memory and launches the warps
   at byte PC 0 with all four lanes active.
2. `TID R1` gives each lane `4 * warp_id + lane_id`. Warp 0 handles elements
   0-3, warp 1 handles 4-7, warp 2 handles 8-11, and warp 3 handles 12-15.
3. Each FP32 value occupies four bytes. Two integer additions multiply the
   index by four. `ADDI` adds the array bases to get addresses.
4. `LD_SHARED` fetches the inputs. The scoreboard keeps their destination
   registers busy until the complete load writes back.
5. `FADD` adds the two binary32 values. The following store waits until that
   result is ready.
6. `ST_SHARED` writes the answer. `EXIT` waits for the store and any other
   outstanding work before halting that warp.

For example, warp 1 / lane 2 receives index 6. It reads A[6] at byte 24 and
B[6] at byte 88, then writes C[6] at byte 152. With the default inputs, that
calculation is `7 + 70 = 77`.

Fetch and decode work exactly as they do for the existing tests: the PC selects
an instruction word, the decoder extracts its fields, and issue control waits
for operands and execution resources. The assembler supplies those words from
an editable assembly file.

## The program

The source is [kernels/vector_add.asm](../kernels/vector_add.asm):

```asm
TID       R1
ADD       R1, R1, R1
ADD       R1, R1, R1
ADDI      R2, R1, 64
ADDI      R3, R1, 128
LD_SHARED R4, [R1]
LD_SHARED R5, [R2]
FADD      R6, R4, R5
ST_SHARED [R3], R6
EXIT
```

| Array | Starting byte address | Size |
| --- | --- | --- |
| A | 0 | 16 FP32 values / 64 bytes |
| B | 64 | 16 FP32 values / 64 bytes |
| C | 128 | 16 FP32 values / 64 bytes |

No register preload is needed. `TID` initializes R1 and the program derives
everything else from it. R0 is an ordinary writable register, not a hardwired
zero register. The current program has fixed array bases and a fixed size of
16; it has no general argument ABI, length check, or grid-stride loop yet.

## Run it

From the repository root, with Python 3 and Icarus Verilog installed:

```sh
python3 scripts/run_kernel.py vector_add
```

The command assembles the source, compiles the real `snackwell_sm`, runs the
kernel, reads back shared memory, and checks every result against the existing
exact-integer FP32 reference. NaNs are compared by classification; other results
must match bit for bit. It also checks that inputs and memory outside C were
not overwritten. The simulator supplies only inputs and launch control; Python
does not calculate the SM's output in place of executing its instructions.

The default inputs are A = `[1, 2, ..., 16]` and B = `[10, 20, ..., 160]`, giving
C = `[11, 22, ..., 176]`. Files are saved under `build/kernels/vector_add/`:

- `program.hex`: assembled instruction image, padded to 1,024 words with EXIT.
- `input.hex` / `output.hex`: complete shared-memory snapshots.
- `registers.hex`: architectural registers after execution.
- `compile.log` / `simulation.log`: compiler and simulation output.
- `result.json`: verification status, input/output bits, and simulated cycles.
- `trace.csv`: observations from the SM at every execution clock edge.
- `profile.json`: instruction counts, memory service, FP/tensor activity, and issue stalls.

For other inputs, pass a JSON file containing arrays `a` and `b`, each with
16 numbers:

```sh
python3 scripts/run_kernel.py vector_add --inputs inputs.json
```

Numbers are rounded to binary32 before being loaded. The cycle count measures
execution after enabling issue, excludes initialization/launch, and depends on
FP operand latency and scheduling. It is not a hardware clock-speed measurement.

## Assembler and instruction support

```sh
python3 scripts/assemble.py kernels/vector_add.asm -o build/vector_add.hex
```

The assembler supports all currently defined SM opcodes:

| Syntax | Meaning |
| --- | --- |
| `ADD/SUB/AND/OR/XOR Rd, Ra, Rb` | Per-lane 32-bit integer/bitwise operation |
| `SHR Rd, Ra, Rb` | Logical right shift by Rb[4:0] |
| `SLTU Rd, Ra, Rb` | Unsigned integer less-than, returns 0 or 1 |
| `FLT Rd, Ra, Rb` | Ordered FP32 less-than, returns integer 0 or 1 |
| `ADDI Rd, Ra, immediate` | Add a signed 14-bit integer; result wraps at 32 bits |
| `TID Rd` | Per-lane local index: `4 * warp_id + lane_id` |
| `FMUL/FADD/FSUB Rd, Ra, Rb` | Per-lane FP32 arithmetic |
| `LD_SHARED Rd, [Ra + offset]` | Load a 32-bit word at a per-lane byte address |
| `ST_SHARED [Ra + offset], Rb` | Store a 32-bit word at a per-lane byte address |
| `TMUL Rd, Ra, Rb` | Multiply two 4x4 FP32 register tiles |
| `BEQ Ra, Rb, label` | Branch if equal in each active lane |
| `SSY label` | Set the reconvergence PC for a following divergent branch |
| `NOP` / `EXIT` | Advance without work / drain and halt the warp |

Registers are R0-R15. TMUL tile bases are R0-R12 because each tile uses four
consecutive registers. Immediates and memory offsets range from -8192 to 8191.
Memory operands accept `[Ra]`, `[Ra + offset]`, or `[Ra - offset]`. Decimal,
hexadecimal, and binary integers are supported. Opcodes/registers ignore case;
labels are case-sensitive. Comments start with `;`, `//`, or `#`.

Labels may appear alone or before an instruction. BEQ and SSY label offsets are
relative to the following instruction, in instruction words. An explicit numeric
branch argument is already such a relative word displacement. No pseudoinstructions,
expressions, macros, or data directives are supported. Bad operands, unknown
labels, duplicate labels, and oversized programs produce errors with line numbers.

ADDI uses only Ra as a source; TID uses no source registers. Both reserve Rd
and respect RAW/WAW dependencies and active masks through the existing integer
writeback path. TID values are based on physical warp/lane IDs, so masks do not
renumber them. These instructions execute in the full multiwarp SM; the older
standalone single-warp teaching core retains its earlier instruction subset.
An empty-mask BEQ now falls through, allowing an inactive warp to reach EXIT.

## Instruction loading and current limits

Instruction memory defaults to 1,024 words / 4 KB. `+PROGRAM=/path/program.hex`
loads a simulator image at initialization; the module also accepts `INIT_FILE`
and `WORDS` parameters. File-loaded images initialize unused locations to EXIT.
When no file is supplied, the original four-instruction demonstration program
is retained for existing smoke tests. Out-of-range or unaligned instruction
fetches return EXIT instead of wrapping to another instruction; there is no
instruction-access trap. Reset restarts execution state without reloading the ROM.

The simulation host loads shared memory through the testbench before execution
and launches warps using the public launch interface. `+WARP_MASK` selects which
warps to launch (default `f`); `+ACTIVE_MASKS` selects active lanes within those
warps. Unlaunched warps do not execute instructions. `matmul_4x4` launches only
warp 0 with all four lanes active, since masked input loads would leave its
register tiles incomplete. The generic Python `run_program` API accepts
`warp_mask` for tests and other programs. `+TRACE` optionally writes the CSV
observations; the Python runner enables it automatically.

There is no runtime host
transfer bus, global-memory allocation, C/CUDA compiler, or FPGA program loader
in this milestone. Inputs/outputs still live in the 4 KB shared memory.

## Tests

```sh
python3 scripts/run_tests.py
python3 scripts/run_tests.py tb_snackwell_kernel
python3 -m unittest discover -s scripts/tests -v
```

The full regression includes the assembled vector-add kernel and the Python
assembler/kernel suites. Tests cover known encodings, every opcode, branch
labels and reconvergence, malformed programs, missing program files, fetch
bounds, ADDI immediate endpoints/wraparound, TID across all warps/lanes,
load/FP RAW and WAW hazards, programs beyond the old 16-word limit, every
active mask, special FP32 values, and seeded random FP32 inputs. They compare
stored results and untouched memory against an independent arithmetic reference.

Matrix-kernel tests execute both implementations with zero/identity/basis
matrices, signed random values, varied exponents, raw FP32 bits, subnormals,
overflow, infinities, and NaNs. Directed examples distinguish separate rounding
from fused FMA and verify the order of accumulation. Tests also cover a tensor
kernel on each physical warp, SIMD lane/warp masks, profile work counts, custom
JSON inputs, and missing-store/wrong-arithmetic programs that must fail verification.
