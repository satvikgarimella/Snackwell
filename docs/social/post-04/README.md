# Snackwell build log 04 — kernels

Archived build-post snapshot. Its diagrams and captured results describe that
milestone. See the [current repository overview](../../../README.md) for the
implemented hardware and latest commands.

Eight editable Excalidraw diagrams in the same cream, blue, orange, purple and
green style as the tensor-core post. Every label, box and arrow is a native
Excalidraw element. There are no embedded raster images or external assets.

Open a `.excalidraw` file using **Open** in Excalidraw, or drag it onto its
canvas. The archive `snackwell-post-4-excalidraw.zip` contains all eight diagrams
and eight editable animation steps. The original files remain alongside it.

| File | Topic |
| --- | --- |
| [00-kernel-overview.excalidraw](00-kernel-overview.excalidraw) | Opening-tweet image: vector operations, tiled matmul, single-head attention |
| [01-assembly-to-sm.excalidraw](01-assembly-to-sm.excalidraw) | Assembly to machine code to SM, with a short vector-add excerpt |
| [02-shared-memory.excalidraw](02-shared-memory.excalidraw) | Four separate banks versus four lanes waiting for the same bank |
| [03-vector-kernel.excalidraw](03-vector-kernel.excalidraw) | Four lanes adding four pairs; four warps cover sixteen values |
| [04-tiled-matmul.excalidraw](04-tiled-matmul.excalidraw) | An 8×8 product built from repeated 4×4 tensor operations |
| [07-more-kernels.excalidraw](07-more-kernels.excalidraw) | Simple examples of additions, reductions, activations, LayerNorm, softmax and a linear layer |
| [05-attention-kernels.excalidraw](05-attention-kernels.excalidraw) | The four launches implementing single-head causal attention |
| [06-verified-results.excalidraw](06-verified-results.excalidraw) | Fresh pass counts and how the numerical results are checked |

The matching 1600×1000 PNGs are in [previews](previews).
[all-diagrams.png](previews/all-diagrams.png) is a contact sheet for review.
The PNGs are local previews rendered from the same native element data.
Excalidraw may substitute its bundled monospace font when opening code text.

The opening image shows three separate workload examples. Its green attention
grid depicts the allowed positions of a causal mask, not probability values.

## Tiling animation

[tiled-matmul.gif](animation/tiled-matmul.gif) highlights the eight tile pairs
used by the 8×8 kernel. The editable poses are in [animation/steps](animation/steps).
They follow the implemented order: output tile row, output tile column, then
the two input-tile pairs. Each output tile starts at zero. Blue is A, orange is
B, purple is the output tile being accumulated, and green marks completed
output tiles, including the current tile after its second product. Grey tiles
are not involved in the current operation.

This illustrates kernel order. Its playback speed does not represent clock
timing; each pictured step includes loads, TMUL, FP32 addition and stores.
Excalidraw scenes are static, so edit the individual step files to change poses.

## Verification behind the claims

See [CLAIMS.md](CLAIMS.md) for source and test references. The run captured on
2026-10-09 UTC passed **37 RTL testbenches, 36 Python test methods, and all
16 runnable workload cases**. The Python methods include additional shapes,
edge cases and full-SM simulations.

The original command output, Python test log, metadata, and selected result
JSON files are preserved in [code/evidence](code/evidence). The diagrams are
generated from those results. These are simulation measurements; the project
has not yet demonstrated a complete pretrained model, training, or FPGA timing.

Arithmetic and tiled-matmul checks reproduce the specified FP32 rounding order.
GELU, LayerNorm, softmax and final attention use the tolerances documented in
[INFERENCE_KERNELS.md](../../INFERENCE_KERNELS.md). The recorded attention
error in `code/evidence/attention-result.json` is for the default 4-token,
width-4 causal example, not a general bound. Recorded cycles exclude setup and
launch overhead. The attention total sums the four execution intervals.

## Thread draft

[thread-draft.md](thread-draft.md) contains all nine tweets with the matching
image/animation and editable file labeled beside each one. The text is also
stored in [tweets.json](tweets.json); the generator checks that each tweet is
under 280 ASCII characters and that all its media files exist. Nothing has
been posted to X.

## Rebuild

With Python 3, Pillow, and macOS Arial/Menlo fonts available, run from the repo:

```sh
python3 docs/social/post-04/code/make_visuals.py
```

The generator checks scene bounds, text overlap, containment, unique element
IDs, native editability and the captured verification results. It writes a
manifest with source hashes so the published evidence has a recorded source
snapshot. It never uploads files to a service.
