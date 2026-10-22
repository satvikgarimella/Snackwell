# Whiteboard the inference path

Use the current 32-character, 64-feature model as the example. Work through one
box at a time; predict its input/output shape and memory movement before
opening that stage's assembly or trace.

## 1. Draw the complete computation

```text
32 character IDs
    |
token lookup + position embedding -> X [32 x 64]
    |
+-- Repeat this block twice -------------------------------+
| LayerNorm -> Q, K, V projections           [32 x 64 each] |
| Split features into four heads             [32 x 16 each] |
| Per head: Q K^T / sqrt(16) + causal mask    [32 x 32]      |
| Per head: row softmax -> multiply by V     [32 x 16]      |
| Concatenate heads, output projection + X  [32 x 64]      |
| LayerNorm -> expand -> GELU                [32 x 128]     |
| Contract + residual                        [32 x 64]      |
+---------------------------------------------------------+
    |
final LayerNorm -> vocabulary projection [32 x 65]
    |
softmax -> next-character distributions
    |
sample the last distribution to produce the next character
```

Keep the residual arrows visible on the board. Explain why the causal mask
hides future characters, which axis each softmax normalizes, and why there is
one next-character distribution for every input position.

## 2. Expand one matrix multiplication

Take Q = X Wq, with X shaped 32×64 and Wq shaped 64×64. Draw a 4×4
output tile. Its inner dimension spans sixteen 4×4 pairs of input tiles.
Each pair requires one `TMUL`; the program adds the resulting partial tile
into C using `FADD`.

There are 128 output tiles, requiring 2,048 `TMUL` operations and 131,072 scalar
multiplications. A launch covers at most eight output rows, sixteen columns
and 32 inner terms. This full projection takes 32 launches, including two K
chunks per output rectangle. Later K chunks load the actual previous C words
and continue accumulating on the SM. Trace the raw-word copies between launches.

For generation, the final block only needs the last four query rows. Earlier
blocks still produce all 32 rows, and all keys and values remain necessary.
Prove that the final prediction is unchanged before counting the saved work.

Then map one tile onto registers:

```text
R0-R3  = four rows of A, four columns across lanes
R4-R7  = four rows of B, four columns across lanes
TMUL R8, R0, R4
R8-R11 = four rows of the result
```

The tensor unit gathers operands, runs its array, and writes back rows. Explain
why a dependent instruction must wait until its required destination is ready.

## 3. Follow an instruction through the SM

Draw: warp PC -> instruction fetch -> decode -> dependency/resource checks ->
register reads -> execution unit -> writeback -> register file.

Attach the scoreboard to issue and writeback. For an FP operation, label the
saved warp ID, destination register, and active lanes. The result can return
several cycles later; these fields preserve where it belongs.

For a load/store, draw four memory banks and route addresses using bits [3:2].
Try four addresses in different banks, then four in the same bank. Predict the
number of service cycles and check the trace. The model currently launches
only warp 0; scheduling more cooperating warps is later optimization work.

## 4. Separate responsibilities

Draw two boxes: simulation host and Snackwell SM. The host owns checkpoint
files, launches, memory-image copies, character encoding/decoding, and result
verification. The SM performs embedding lookup, model arithmetic, vocabulary
masking, softmax and character selection. The host supplies a seeded random
draw; the sampling kernel performs the cumulative sum and selection on the SM. The reference's calculated answers are
used only for comparisons; they never replace device activations.

This boundary also explains what is still missing for hardware deployment:
today's testbench loads internal memory arrays and resets between launches.
A hardware memory/launch interface must eventually replace that simulation
mechanism.

## 5. Learn from one measurement

Run one prediction, inspect its `result.json`, then choose one launch and read
`program.asm`, `layout.json`, and `profile.json`. Predict which part takes the
most time, compare that prediction with the measurements, and explain the gap.
Change only one thing when optimizing, rerun correctness, and compare the
same checkpoint and input before and after.

Start with [the inference guide](MODEL_INFERENCE.md) for commands. The next
whiteboard after inference is the backward pass: draw each gradient's shape
and how it travels through the same operations before writing training kernels.
