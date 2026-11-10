# Persistent model memory

Updated November 9, 2026.

`--memory resident` keeps weights, intermediate tensors and FP32 KV-cache entries
in simulated external RAM for an entire prediction. It supports the existing
FP32 and NVFP4 inference arithmetic. Shared mode remains the default; global
mode remains the reference for full-image uploads. `resident` now uses direct
device pointers; `resident-copy` preserves the earlier gather/scatter baseline.

```sh
python3 scripts/run_sentence.py --precision nvfp4 --memory resident --interactive
python3 scripts/compare_model_memory.py --resident --baseline resident-copy --cases 8 \
  --out build/inference/direct-comparison
python3 scripts/compare_model_memory.py --resident --baseline resident-unoptimized --cases 8 \
  --out build/inference/compiler-comparison
```

Example prompt:

```text
The ball is in the kitchen. The book is in the garden. Where is the ball?
```

## What stays on the device

One `Generation` owns one untraced Verilator session. The simulator preserves
external RAM across SM resets. Only explicit ranges are uploaded or read back;
no full external image is supplied for each model kernel.

- FP32 mode uploads the checkpoint weights once. NVFP4 mode uploads packed
  linear-weight tiles and their factors, plus the FP32 embedding, normalization
  and bias tensors. Unused full-precision linear matrices remain on the host
  for quantization and independent verification.
- Written outputs receive new allocations. Unchanged entries, such as unmasked
  vocabulary logits, can keep their original allocation references. Python carries the verified
  shadow value and allocation reference through slices, reshapes and joins.
  A consuming kernel reads that allocation through the SM. Its shadow is not
  uploaded in place of the device result.
- K/V entries retain those same allocations across decoded tokens. A context
  rebuild computes new entries at reset positions; unchanged weights stay
  resident. Old allocations are reclaimed once no tensor view references them.
- NVFP4 activation quantization is still on the host. It reads verified output
  shadows and uploads new packed activations. This remains real host traffic,
  counted separately from the SM's memory transactions.

The allocator uses **1 MiB of simulated external RAM**. The bottom 64 KiB remains
reserved for the older kernel window, descriptors and packed-operand area. Direct
mode does not stage tensors through that window. The remainder
holds persistent allocations, with padding around each allocation. Exhaustion
raises an error rather than silently returning to full-image uploads. The
reported peak allocation is the sum of live allocation sizes, including padding;
it excludes the fixed 64 KiB window and free holes. It is not a minimum-capacity
or placement result. The RAM is initialized to a sentinel once
per simulator process, without streaming an initialization image from Python.

This capacity is a simulation setting, not FPGA BRAM utilization or proof of
board fit. A physical external-memory controller and board integration remain.
The original global transport still enforces its 64 KiB address limit.

## How kernels access allocations

Generated kernels keep their logical indexing, with a shared-memory table that
maps each logical word to its physical external address. An uncached load executes
`LD_SHARED` to obtain the pointer, followed by `LD_GLOBAL` to read the tensor.
This handles sliced and fragmented tensors without gathering their data first.
The simulator does not translate addresses on behalf of the SM.

When several active lanes request the same physical word, the global LSU now
reads it once and broadcasts the response. Separate tensor-load instructions
still make fresh external reads. This reduces repeated scalar
traffic without retaining a cache or changing the generated arithmetic. The
[serial/broadcast comparison](measurements/load-broadcast-20261109/README.md)
measures this hardware change with identical resident kernels.

A store computes the new output allocation's address, executes `ST_GLOBAL`, then
updates the shared pointer entry. Multi-pass kernels such as softmax therefore
read their own earlier writes, while other tensor views still reference the old,
unchanged input. Valid vocabulary entries that the masking kernel never writes
retain their input references. This is important for residuals and KV state.

The lowering pass computes register liveness and known values through symbolic
branches and loops. It now applies three optimizations:

- Compiler-owned arithmetic constants and tensor descriptors get a separate
  shared literal pool when space permits. A load uses that pool only when its
  complete logical address is known on every incoming control-flow path.
  Tensor inputs, weights and indirect accesses still read external memory.
- A spare register can retain a physical pointer for subsequent loads in the
  same basic block. The compiler discards cached pointers at labels, branches,
  table writes, async-copy submissions, and changes to their base or holder.
- An otherwise unused register holds the output-address displacement for the
  whole launch. Stores then need one temporary register and three or four
  instructions, instead of two temporaries and six instructions. Kernels using
  every register retain the original store expansion.

The literal pool never replaces pointer-table entries, so indirect reads of the
same constants still work. It ends before the displacement word at byte 4092;
if it fills, remaining constants keep their original external loads. Every
shared word, including literals, remains checked. Unsupported instructions,
insufficient registers or an oversized table fail explicitly. Wide LayerNorm uses counted
loops, and wide additions use smaller row batches to stay within the existing
1,024-word program ROM. The arithmetic order and 16-register ISA are unchanged.

Native NVFP4 copies each 12-word A group and 12-word B group directly from their
separate physical addresses. Copy slots S0 and S1 protect the two shared operand
regions. After the tensor unit gathers them, both slots are released and the
next pair is fetched during arithmetic. There is no combined A/B staging copy in
external RAM. The address table starts above the shared operand buffers; copy
descriptors and its output-address constant occupy separate shared locations.

Each model stage now needs one RTL launch. There are no separate gather/scatter
launches or output copies. Necessary global-to-shared NVFP4 operand transfers
remain and are measured by the existing async-copy counters.

The C++ host only loads explicit host uploads, clocks RTL, checks guards and
returns requested readback ranges. It performs neither model arithmetic nor
device-to-device copies. `resident-copy` retains the old implementation for
equivalence and performance comparisons.

The comparison command's `resident-unoptimized` baseline disables these three
compiler optimizations while retaining direct addressing and load broadcasting.
It uses `Generation(..., optimize_resident=False)`; this option is part of the
KV-cache compatibility check. The normal resident path enables optimization.
See the [compiler measurements](measurements/resident-compiler-20261109/README.md).

## Verification and lifetime

Per-stage and whole-model numerical references are unchanged. FP32 retains its
existing tolerances; NVFP4 still checks the exact mixed-precision computation.
The new focused tests additionally cover:

- Equivalent probabilities and selections for prefill, cached decode, sampling
  and context rebuilds in both precisions.
- Persistent data surviving launch resets; no unchanged-weight uploads after
  prefill; intermediate and K/V allocation reuse.
- Allocation exhaustion/reclamation, foreign or stale handles, model changes,
  invalid ranges, latency variation and multi-batch output copies.
- Fragmented views, residual aliases, partial masking, multi-pass softmax,
  causal masks, indirect embedding indices, maximum NVFP4 job counts, K chunks,
  multiwarp FP32 products and wide row kernels.
- Complete external-memory guard checks on the simulator side, plus shared
  guards, zero error status and every accepted transaction receiving a response.
- Injected readback corruption, timeouts and cancellation. No failed forward
  commits new KV state or publishes a verified token.

The driver starts a new launch only after the prior launch halts with no pending
work. A failed resident run invalidates the session and all allocation handles.
Start a new prediction after cancellation; old resident addresses cannot be
reused in a fresh process. Older shared/global generations retain their existing
retry behavior. CLI entry points close sessions on completion and exceptions.

Full direct artifacts contain the executed `program.asm`, shared pointer table
in `input.hex`, `resident-layout.json` with physical uploads and readback/write
ranges, and logical numerical input/output images. All external words outside
declared output ranges are checked in the simulator; every returned shared word
is checked against the expected table, operands, descriptors and guards. Compact
failures expand artifacts. Transport failures also save the last host
shadow and descriptors; the shadow is diagnostic state, not a claim
to have read back hardware after a timeout. Resident mode does not support the
traced/Icarus transport; use global mode for that debugging path.

## Measurements

`result.json` records `resident` metrics per forward:

- `host_upload_bytes` / `host_readback_bytes`: external data payloads only.
- `protocol_request_bytes` / `protocol_response_bytes`: complete binary frames,
  including programs, shared-memory copy descriptors, registers and headers.
- `constant_upload_bytes`: newly retained weight/packed-weight data.
- `reads`, `writes`, `responses` and `rtl_launches`: actual SM traffic and work.
  `device_copy_words` and `copy_cycles` count only the old layout copies and are
  zero in direct mode. Native global-to-shared copy traffic is still reported
  separately in each stage's `copy_reads` and `copy_writes` counters.
- `capacity_bytes` and `peak_allocation_bytes`: configured external capacity
  and allocator high-water mark, respectively.

Independent verification still reads kernel results back to the host. Reduced
readback payloads do not mean verification was disabled or that the whole model
is autonomous on hardware. Host tokenization, quantization, launch scheduling
and numerical checks remain. The [direct-addressing measurements](measurements/direct-memory-20261109/README.md)
show 51.5% fewer sentence-model cycles than the old resident implementation,
with identical probabilities and answers. The [first resident measurements](measurements/resident-memory-20261109/README.md)
remain as historical evidence. Wall times include verification and are not FPGA throughput.
