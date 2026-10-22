# Snackwell build log 04 — thread draft

Draft only. Each numbered section below is one tweet; media notes are not tweet text.

## 1. Opening

Hey guys, Snackwell has its first kernels now :)

I've finished the shared memory work and assembler from the last post, and got vector ops, tiled matmul and single-head attention running in RTL simulation.

Here's what I did

**Attach:** [00-kernel-overview.png](previews/00-kernel-overview.png)

**Editable:** [00-kernel-overview.excalidraw](00-kernel-overview.excalidraw)

Text length: 225 ASCII characters (URL shortening only reduces this).

## 2. Shared memory

First I had to finish shared memory. It could already spot when lanes wanted the same bank, but it didn't handle all those accesses yet.

Now those lanes take turns until the request is done. Lanes using different banks can still go together.

**Attach:** [02-shared-memory.png](previews/02-shared-memory.png)

**Editable:** [02-shared-memory.excalidraw](02-shared-memory.excalidraw)

Text length: 242 ASCII characters (URL shortening only reduces this).

## 3. Assembler

With that working, I needed a way to write kernels (programs for the GPU) without packing instruction bits by hand.

So I made an assembler. I write FADD, TMUL, loads and stores in an .asm file, and it turns them into machine code for the SM.

**Attach:** [01-assembly-to-sm.png](previews/01-assembly-to-sm.png)

**Editable:** [01-assembly-to-sm.excalidraw](01-assembly-to-sm.excalidraw)

Text length: 242 ASCII characters (URL shortening only reduces this).

## 4. First kernels

I started with vector addition: C[i] = A[i] + B[i].

Each lane loads two values, adds them and writes the result back. Four warps with four lanes each handle all 16 elements.

I also wrote vector multiply and 4x4 matmul using both the SIMD and tensor units.

**Attach:** [03-vector-kernel.png](previews/03-vector-kernel.png)

**Editable:** [03-vector-kernel.excalidraw](03-vector-kernel.excalidraw)

Text length: 257 ASCII characters (URL shortening only reduces this).

## 5. Tiled matrix multiplication

To go beyond 4x4, I split bigger matrices into tiles. The kernel loads a pair, runs TMUL and adds that partial result to C, then moves to the next pair.

That's 8 TMULs for an 8x8 multiply, and 64 for 16x16.

**Attach:** [tiled-matmul.gif](animation/tiled-matmul.gif)

**Editable:** [04-tiled-matmul.excalidraw](04-tiled-matmul.excalidraw)

**Still alternative:** [04-tiled-matmul.png](previews/04-tiled-matmul.png)

The same tweet text works with either the animation or the still image.

Text length: 207 ASCII characters (URL shortening only reduces this).

## 6. More kernels

Once tiled matmul was working, I added bias and residual adds, row sums/maxes, ReLU, GELU, LayerNorm and softmax.

There's also a linear layer, which runs matmul and then adds the bias. All of these run as Snackwell assembly too.

**Attach:** [07-more-kernels.png](previews/07-more-kernels.png)

**Editable:** [07-more-kernels.excalidraw](07-more-kernels.excalidraw)

Text length: 229 ASCII characters (URL shortening only reduces this).

## 7. Attention

That gave me enough to put together single-head attention.

It runs QK^T, scales and masks the scores, applies softmax, then multiplies the weights by V.

These are four separate launches, with the simulation runner passing the output memory from one to the next.

**Attach:** [05-attention-kernels.png](previews/05-attention-kernels.png)

**Editable:** [05-attention-kernels.excalidraw](05-attention-kernels.excalidraw)

Text length: 263 ASCII characters (URL shortening only reduces this).

## 8. Verification

I check what the SM actually writes to memory against independent references. Matmul has to match bit for bit; approximate functions are checked within tolerances.

The latest run passed 37 RTL testbenches, 36 Python tests and all 16 workload cases.

**Attach:** [06-verified-results.png](previews/06-verified-results.png)

**Editable:** [06-verified-results.excalidraw](06-verified-results.excalidraw)

Text length: 249 ASCII characters (URL shortening only reduces this).

## 9. Next steps

Next I'm putting these kernels together into a small transformer block and checking its forward pass. Then I'll add global memory and a runtime so we can load trained weights and run inference on a small model.

Code: https://github.com/satvikgarimella/Snackwell

SEE YOU SOON

**Attach:** None

Text length: 276 ASCII characters (URL shortening only reduces this).
