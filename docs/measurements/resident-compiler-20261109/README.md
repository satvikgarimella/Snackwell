# Resident compiler optimization

November 9, 2026. This pass reduces address-handling instructions and constant
traffic in resident inference kernels. It uses the existing ISA and the same
RTL/simulator source as the preceding load-broadcast measurements:
`73a30077fb39366d358e93a7fe03d36326aa2e11d3ccc3d5a697b32d0252a9bb`.

Three changes are enabled by default for `--memory resident`:

1. A separate shared-memory pool holds compiler-owned literals. Known-address
   loads read these values directly; indirect accesses retain the complete
   external pointer table. Model tensors and weights stay in external RAM.
2. Dead registers can retain physical pointers within a basic block. Changes
   to their base/holder, joins, branches and table writes invalidate reuse.
3. A globally unused register, when available, retains the output displacement.
   Stores then need three or four instructions instead of six. Kernels with
   no unused register retain the original store expansion.

Register liveness and constant propagation account for symbolic branches,
reconvergence targets and loop backedges. Unknown values never become assumed
addresses. The literal pool stops before byte 4092; constants that do not fit
continue to use external loads. Numerical operation order is unchanged.

## Reproduction

```sh
python3 scripts/compare_model_memory.py --resident \
  --baseline resident-unoptimized --cases 8 --out build/inference/compiler-comparison
python3 scripts/compare_model_memory.py --resident \
  --baseline resident-unoptimized --precision fp32 --cases 2 \
  --out build/inference/compiler-fp32-comparison
python3 scripts/compare_model_memory.py --resident \
  --baseline resident-unoptimized --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'ROMEO:' --tokens 4 --out build/inference/compiler-character-comparison
python3 scripts/compare_model_memory.py --resident \
  --baseline resident-unoptimized --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'My lord, I have something to say' --tokens 4 \
  --out build/inference/compiler-rebuild-comparison
```

Both sides run the same checkpoint and broadcast-enabled RTL, with additional
external latency of two clocks. The unoptimized compiler's sentence, FP32 and
short character results exactly reproduce the cycles and reads of the prior
[broadcast measurements](../load-broadcast-20261109/README.md).
`resident-unoptimized` is a comparison label for resident memory with
`optimize_resident=False`; it is not a different hardware memory space.

## Results

| Measurement | NVFP4 sentence | FP32 sentence | NVFP4 character | NVFP4 context rebuild |
| --- | ---: | ---: | ---: | ---: |
| Prompts | 8 reserved questions | 2 reserved questions | `ROMEO:` | Long prompt above |
| Identical probability vectors/selections | 56/56 | 14/14 | 4/4 | 4/4 |
| Correct complete answers, both compilers | 8/8 | 2/2 | No answer target | No answer target |
| Previous cycles | 57,942,042 | 20,158,635 | 6,526,615 | 25,810,716 |
| Optimized cycles | 51,668,574 | 19,058,863 | 5,850,400 | 23,253,957 |
| Cycle reduction | 10.8% | 5.5% | 10.4% | 9.9% |
| Previous external reads | 3,333,184 | 1,618,210 | 394,295 | 1,619,076 |
| Optimized external reads | 2,803,456 | 1,508,170 | 342,148 | 1,399,424 |
| Previous / optimized wall seconds | 41.04 / 38.86 | 10.50 / 10.16 | 5.99 / 4.89 | 20.81 / 18.91 |

Sentence cases use seed 20261109 and checkpoint `66af3342b4a6`; character cases
use `43f211daad7c`. The short character continuation is `\nMar`; the longer
prompt produces ` the`. These are unchanged outputs, not language-quality gains.
The longer case covers one prefill, two cached decodes and one context rebuild.

NVFP4 sentence cached decode averages **0.438 → 0.415 seconds/token**; prefill
averages **2.50 → 2.37 seconds**. Wall times include verification, host launches,
pointer-table construction and activation quantization. They are single-run
simulation measurements under ordinary system load, not FPGA throughput. The
full regression was run after these comparisons, not concurrently with them.

Across the eight NVFP4 questions, writes remain 673,152 and launches remain
10,024. Host uploads/readbacks remain 869,248 / 2,511,584 external bytes. Literal
pools occupy the already transferred shared image; no new protocol or extra
device allocation is needed. Every paired stage retains identical native MAC,
FP32 tensor multiply and async-copy counts. Optimized programs never exceed the
baseline instruction count in the measured stages.

| Operation, NVFP4 sentence | Previous cycles | Optimized cycles | Previous / optimized reads |
| --- | ---: | ---: | ---: |
| Native linear kernels | 22,955,540 | 21,038,903 | 1,342,872 / 1,253,304 |
| Softmax | 10,470,929 | 8,241,485 | 403,704 / 153,480 |
| GELU | 6,778,597 | 4,887,554 | 228,480 / 43,520 |
| LayerNorm | 3,980,944 | 3,898,024 | 200,760 / 197,120 |

Together with load broadcasting, the two passes reduce sentence cycles from
71,231,426 to 51,668,574 (**27.5%**) and reads from 5,122,960 to 2,803,456
(**45.3%**), relative to the original direct-pointer implementation.

## Verification and scope

Final full regression: **46/46 RTL benches and 269/269 Python methods passed,
with no skips**. The Python suite took 393.126 seconds. Ruff check/format passed
for 147 Python files, and `git diff --check` passed. All 238 recorded source
hashes were checked against the final files.

The initial run reached the previous 420-second overall suite deadline. That
runner discarded partial output on timeout, so its completed method count is
unknown. The runner now streams its log continuously and allows 900 seconds
for the complete suite, including cold simulator builds. Per-kernel deadlines
and numerical assertions are unchanged. The full rerun passed; the original
timeout logs are retained alongside the passing result.

All existing numerical references and full shared/external guard checks remain
enabled. New execution tests compare optimized and original lowering across
pointer reuse, base changes, in-place publication, divergent joins in all four
warps, loop backedges, full shared tables, use of all sixteen registers,
register redefinition, delayed responses, sampling, rebuilds and compiler-policy
changes while KV state is live. Injected literal corruption invalidates the
session rather than publishing a verified result.

This pass does not remove every pointer lookup: dynamic/fragmented views and
invalidated register caches keep their lookup path. Shared-memory transport
remains the existing default outside explicitly selected resident mode. Physical
memory integration, FPGA fit and timing closure remain separate milestones.

## Evidence

- `*-comparison.json`: paired cases, phases, cycles, wall times and traffic.
- `validation.json`: matching FP32 probability-byte hashes for all 78 tokens.
- `operation-profile.json`: per-operation cycles, reads and instruction counts.
- `existing-memory-tests.log`, `compiler-tests.log`: targeted execution checks.
- `regression.json`, `regression.log`, `python-tests.log`: final full regression.
- `first-regression.json`, `first-regression.log`, `first-python-tests.log`:
  the initial suite timeout before the test-runner logging/deadline fix.
- `terminal-smoke.log`: the public command with the optimized compiler.
- `sources.json`: hashes of the verified source files.
