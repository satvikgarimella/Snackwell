# Snackwell build log 04 — thread draft

Draft only. Each numbered section below is one tweet; media notes are not tweet text.

## 1. Opening

Hey guys, Snackwell has kernels now :)

Last time I said I'd improve shared memory, build an assembler and write vector and matrix programs.

Those now run in RTL simulation, along with single-head attention. Here's what changed.

**Attach:** None

Text length: 229 ASCII characters (URL shortening only reduces this).

## 2. Shared memory

Shared memory used to detect when lanes hit the same bank. It now serves those lanes one at a time until the request is complete.

Lanes using different banks can still be served together.

**Attach:** [02-shared-memory.png](previews/02-shared-memory.png)

**Editable:** [02-shared-memory.excalidraw](02-shared-memory.excalidraw)

Text length: 188 ASCII characters (URL shortening only reduces this).

## 3. Assembler

I wrote an assembler for Snackwell's instruction set, so kernels can live in readable .asm files.

It turns instructions like FADD and TMUL into machine code. The SM fetches, decodes and executes that program.

**Attach:** [01-assembly-to-sm.png](previews/01-assembly-to-sm.png)

**Editable:** [01-assembly-to-sm.excalidraw](01-assembly-to-sm.excalidraw)

Text length: 209 ASCII characters (URL shortening only reduces this).

## 4. First kernels

The first kernel adds two arrays of 16 FP32 values.

Each lane loads one value from A and B, adds them, and stores C. Four warps of four lanes cover all 16 elements.

There are also vector multiply and SIMD/tensor versions of 4x4 matmul.

**Attach:** [03-vector-kernel.png](previews/03-vector-kernel.png)

**Editable:** [03-vector-kernel.excalidraw](03-vector-kernel.excalidraw)

Text length: 237 ASCII characters (URL shortening only reduces this).

## 5. Tiled matrix multiplication

The tensor array is still 4x4. To multiply 8x8 and 16x16 matrices, the kernel loads smaller tiles, multiplies them with TMUL and adds the partial results.

An 8x8 product takes 8 TMULs; 16x16 takes 64. The animation shows which tiles are used.

**Attach:** [tiled-matmul.gif](animation/tiled-matmul.gif)

**Editable:** [04-tiled-matmul.excalidraw](04-tiled-matmul.excalidraw)

**Still alternative:** [04-tiled-matmul.png](previews/04-tiled-matmul.png)

If using the still image, replace 'The animation shows which tiles are used.' with 'The diagram shows one output tile.'

Text length: 243 ASCII characters (URL shortening only reduces this).

## 6. More kernels

I also added bias/residual addition, row sums and maxima, ReLU, GELU, LayerNorm and softmax, plus a linear layer using matmul + bias.

These all run as Snackwell assembly. The approximations used for GELU, LayerNorm and softmax are checked against references.

**Attach:** None

Text length: 259 ASCII characters (URL shortening only reduces this).

## 7. Attention

Single-head attention now runs in four launches: QK^T, scaling + a causal mask, softmax, then multiplying the weights by V.

Each stage runs on the SM. The simulation runner passes its output memory to the next launch and checks the results.

**Attach:** [05-attention-kernels.png](previews/05-attention-kernels.png)

**Editable:** [05-attention-kernels.excalidraw](05-attention-kernels.excalidraw)

Text length: 241 ASCII characters (URL shortening only reduces this).

## 8. Verification

The latest run passed 37 RTL testbenches, 36 Python tests and all 16 workload cases.

The runner checks actual output memory against independent references. Matmul is checked bit for bit; approximate functions use tolerances.

**Attach:** [06-verified-results.png](previews/06-verified-results.png)

**Editable:** [06-verified-results.excalidraw](06-verified-results.excalidraw)

Text length: 225 ASCII characters (URL shortening only reduces this).

## 9. Next steps

Next I'm connecting these kernels into a small transformer block and checking its full forward pass.

After that, global memory and a runtime for loading data and launching kernels.

Code: https://github.com/satvikgarimella/Snackwell

SEE YOU SOON

**Attach:** None

Text length: 247 ASCII characters (URL shortening only reduces this).
