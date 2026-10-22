# Multiwarp inference kernels and runtime speed

This records the stage before KV caching. The subsequent context-policy change,
cached decoding, quality checks and timings are in [KV_CACHE.md](KV_CACHE.md).

This pass keeps checkpoint `43f211daad7c`, its learned positions, FP32 arithmetic
order, sampling settings, and verification tolerances unchanged. It addresses
unused warps, repeated constant loads, separate bias launches, and stale RTL in
interactive runners. It does not implement KV caching or change model quality.

## Controlled measurement

Prompt: `My lord, I have something to say`; seed `92617660`; temperature `0.7`.
Both measurements use Verilator without cycle tracing on the development Mac,
with all numerical checks enabled. Compilation is excluded. No regression,
evaluation, or profiling process was running during either timed forward pass.

| First-character forward pass | Before this pass | After |
| --- | ---: | ---: |
| Wall time | 10.3903 s | 8.0139 s |
| Simulated cycles | 11,847,367 | 8,101,875 |
| Kernel launches | 1,571 | 1,406 |

This is 1.30× faster in wall time and 31.61% fewer simulated cycles. The embedding,
every block output, final normalization, logits, probabilities, sampled character,
and error metrics are identical. Raw output/register files are not expected to
match between differently partitioned/fused kernels; architectural outputs are.
The earlier FP32 update has its own paired measurements in [FP32_SPEED.md](FP32_SPEED.md).

The user's older 16-character run took 352.1 seconds. Its saved first-character
cycle count was 32,235,401, matching the pre-FP32-optimization implementation.
That run is useful historical evidence, not the baseline for the table above.

A complete interactive rerun produced the same ` against my word` continuation:
first character 7.93 s, subsequent characters averaging 8.07 s, 129,637,439 total
cycles, and 22,496 launches. All 16 characters' probabilities, sampling decisions,
and error metrics match the older run exactly. The summed verified-forward time
is approximately 129 seconds; this excludes compilation and interactive idle time.

## What changed

### Assign independent matrix tiles to multiple warps

`tiled_matmul` accepts one to four warps. A uniform branch sends each warp to its
own output tiles, with private registers and disjoint output addresses. All K
chunks for one tile still accumulate in the same order. The model runtime uses
up to four warps, limited by the number of tiles available in a launch.

While the single tensor unit handles one warp's tile, another warp can load
operands or accumulate a previous tile. This changes scheduling, not the number
of tensor units. No inter-warp producer/consumer dependency or barrier is added.

The traced final-K query-projection chunk performs 64 TMUL operations:

| Trace metric | Previous single-warp kernel | Multiwarp kernel with bias |
| --- | ---: | ---: |
| Total cycles | 9,283 | 5,865 |
| No eligible warp | 6,457 | 62 |
| Tensor adapter idle | 3,779 | 361 |
| Array compute | 4,864 | 4,864 |

The new trace includes the final bias addition previously performed in a
separate launch. Resource-wait cycles remain: the tensor unit can handle only
one tile at a time. The issue and tensor-phase counters describe different,
overlapping aspects of the same execution; do not add them together.

### Broadcast read-only shared-memory requests

A bank reads one selected word and supplies it to all matching lanes. Different
words in the same bank still serialize. Any request containing a store retains
ascending lane service order, preserving mixed read/write behavior and aliases.

In the sampled GELU kernel, conflict requests drop from 2,176 to zero. Service
cycles drop from 8,960 to 2,432, while total cycles drop from 51,738 to 49,434.
The total improvement is smaller because arithmetic and memory work overlap.
Row-strided accesses in softmax can still conflict; this is not a conflict-free
memory system. See [MEMORY.md](MEMORY.md) for the precise contract.

### Add bias before storing the final matrix tile

The last K chunk loads the bias and adds it after the complete matrix sum,
before writing C. The bias is applied exactly once, with a separate FP32 add.
This preserves the old rounding order while removing 165 bias-only launches
and their extra loads/stores from this prediction.

### Improve runner diagnostics and host parsing

Compiled model runners retain an RTL/testbench source fingerprint. Terminal and
browser sessions check it between predictions and rebuild if RTL changed; one
prediction always uses one executable. The terminal prints its RTL fingerprint
and records it with every character's result. Restart an already-running older
Python runner once to load these software changes.

The terminal separately reports first-character and average subsequent-character
latency. Both include verification; compilation is excluded. These are not a
standalone prefill timer and a cached-decode timer: each character still
recomputes the sliding window.

Host profiling also identified per-word regex matching in memory-image parsing.
Whole-image hex decoding now handles that conversion while retaining exact word
counts and rejecting unknown digits, signs, underscores, and malformed widths.
All independent arithmetic checks and memory guards remain enabled.

## Verification and evidence

The combined regression passed all 38 RTL benches and 83 Python methods. The
initial full run exposed a test scanning old launch directories after fusion
reduced their count; the test now uses a fresh directory, and its rerun passed.
Four HTTP methods could not bind localhost inside the sandbox and passed with
loopback permission. The initial failure and successful focused reruns are both
preserved; this was not one uninterrupted green full-suite invocation.

New checks cover 1/2/3/4-warp matrix partitioning, transposed inputs, existing C
accumulation, bias added after multiple K chunks, stale executable replacement,
and strict memory-image decoding. The shared-memory bench verifies 8,277
transactions, including broadcast groups, aliases, masks, stores between reads,
backpressure, and reset. Icarus/Verilator output and trace comparisons also pass.

All 256 saved held-out positions match the earlier RTL probabilities and
intermediate block outputs exactly. Held-out loss remains 1.6053, with 140/256
correct next-character predictions. All 16 sampled terminal characters and their
distributions also match. These checks verify unchanged computation; they do
not claim an improvement in the model's language quality.

Artifacts are under `build/runtime-speed/`. The permanent
[verification record](../models/character_transformer/runtime_performance_verification.json)
contains current source fingerprints, benchmark, regression, generation, and
held-out evaluation results. Historical verification files remain unchanged.

## Remaining limits

The model uses learned positions within a sliding 32-character window. Shifting
the window changes old positions and later-layer context, so retaining old K/V
unchanged would change the computation. Cached decoding needs an explicit model
and context-policy change, with separate quality and correctness evaluation.

There is still one tensor unit, one outstanding SIMD FP operation, and one
outstanding shared-memory request. Host-managed launches and the 4 KB shared
memory remain. These are simulator measurements; FPGA frequency, area, and
throughput still require synthesis and timing evidence. Training on Snackwell
and FPGA deployment are separate roadmap work.
