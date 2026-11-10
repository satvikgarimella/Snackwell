# Broadcasting same-address global loads

November 9, 2026. The global LSU now combines active lanes that load the same
address within one vector instruction. One response fills all matching lanes.
Stores retain their original lane order and request count. There is no cache
between instructions, and adjacent words are not combined.

## Controlled comparison

Both builds use the same source tree, model weights, resident kernels and
additional external-memory latency of 2 clocks. The serial build defines
`SNACKWELL_SERIAL_GLOBAL_LOADS`; the default build enables `BROADCAST_LOADS`.
The serial results exactly reproduce the cycle totals and generated text from
the preceding [direct-memory measurements](../direct-memory-20261109/README.md).

RTL/simulator source SHA-256:
`73a30077fb39366d358e93a7fe03d36326aa2e11d3ccc3d5a697b32d0252a9bb`.
The comparison runner includes compile definitions in its build identity, so
serial and broadcast KV states cannot be confused. Both identities, executable
hashes and build definitions are recorded in each comparison JSON; compiler
versions and flags are in `serial-build.json` and `broadcast-build.json`.

```sh
python3 scripts/compare_global_loads.py --cases 8
python3 scripts/compare_global_loads.py \
  --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'ROMEO:' --tokens 4 --out build/inference/broadcast-character-comparison
python3 scripts/compare_global_loads.py --precision fp32 --cases 2 \
  --out build/inference/broadcast-fp32-comparison
```

| Measurement | NVFP4 sentence | NVFP4 character | FP32 sentence |
| --- | ---: | ---: | ---: |
| Cases | 8 reserved questions | `ROMEO:` | 2 reserved questions |
| Identical probability vectors and selections | 56/56 | 4/4 | 14/14 |
| Correct complete answers, both builds | 8/8 | No answer target | 2/2 |
| Serial cycles | 71,231,426 | 7,890,415 | 24,939,722 |
| Broadcast cycles | 57,942,042 | 6,526,615 | 20,158,635 |
| Cycle reduction | 18.7% | 17.3% | 19.2% |
| Serial external reads | 5,122,960 | 584,688 | 2,261,836 |
| Broadcast external reads | 3,333,184 | 394,295 | 1,618,210 |
| Read reduction | 34.9% | 32.6% | 28.5% |
| Unchanged RTL launches per build | 10,024 | 992 | 2,270 |
| Serial / broadcast wall seconds | 45.17 / 40.80 | 6.41 / 5.05 | 11.58 / 10.05 |

The sentence checkpoint is `66af3342b4a6`, with questions selected using seed
20261109. The character checkpoint is `43f211daad7c`; both builds produce `\nMar`.
This is an execution optimization, not a model-quality change.

NVFP4 sentence averages: prefill **2.76 → 2.47 seconds**, cached decode
**0.48 → 0.44 seconds/token**. These wall times include host quantization,
launch handling, numerical references and memory guards. Builds were warmed,
execution order alternated between tokens, and no other test suite ran during
the saved comparison. They remain single-run simulation timings under ordinary
system load, not FPGA throughput or an isolated statistical benchmark.

Writes, native operand-copy counts, narrow MAC counts, FP32 tensor-multiply
counts, host upload bytes and readback bytes are unchanged. The eight sentence
questions still perform 673,152 external writes, upload 869,248 external bytes,
and read back 2,511,584 bytes. All independent numerical checks and full
external-memory guard checks remain enabled.

## Where the cycles went

Aggregate NVFP4 sentence profile, across prefill and decode:

| Operation | Serial cycles | Broadcast cycles | Serial / broadcast reads |
| --- | ---: | ---: | ---: |
| NVFP4 linear kernels | 27,162,461 | 22,955,540 | 1,909,056 / 1,342,872 |
| FP32 attention matrix products | 12,748,576 | 10,479,808 | 1,310,720 / 1,003,520 |
| GELU | 10,899,846 | 6,778,597 | 783,360 / 228,480 |
| Softmax | 12,020,351 | 10,470,929 | 612,504 / 403,704 |
| LayerNorm | 5,109,968 | 3,980,944 | 351,456 / 200,760 |

Broadcast scalar constants explain much of the GELU and softmax reduction.
Distinct-address traffic still needs individual requests. Shared pointer-table
lookups still execute before resident loads; reducing those lookups remains a
separate compiler/runtime opportunity. Matrix arithmetic, copy scheduling,
model shape and numerical precision are unchanged.

## Correctness coverage

Final full regression: **46/46 RTL benches and 262/262 Python test methods
passed, with no skips**. The Python suite took 388.337 seconds. Ruff checks and
format checks passed for 146 Python files, and `git diff --check` passed. All
237 recorded source hashes still matched after verification.

The initial full run passed all RTL benches and 261 Python methods, but one
traced simulator process exited with SIGKILL (`-9`) without RTL diagnostics.
That same test passed alone in 0.466 seconds, then the complete suite passed
without further RTL or test changes. The cause of the process termination was
not established. Both the initial failure and reruns are retained as evidence.

The unit bench exercises all 16 lane masks with interleaved duplicate groups,
negative offsets, address errors mixed with valid groups, failed shared
responses, empty masks, command snapshots, held requests and writeback,
duplicate-store ordering, fresh reads on later instructions, and reset while a
group is pending. The same bench also runs with broadcasting disabled.

Full-SM checks cover randomized duplicate groups across all four warps, inactive
destinations, independent progress, dependent stalls, writeback contention,
load/store/load ordering, delayed responses and contention with async copies.
Direct-clock serial/broadcast comparisons check identical memory/registers,
exact request counts, unchanged distinct-load timing, and rejection of a build
switch during a resident prediction.

The memory contract is ordinary RAM. Side-effecting MMIO and per-lane volatile
reads are not supported by broadcast loads. This change does not establish
FPGA fit, timing closure or physical memory operation.

## Evidence

- `sentence-comparison.json`, `character-comparison.json`, `fp32-comparison.json`:
  per-token probability words, selections, phases, cycle counts and traffic.
- `operation-profile.json`: full per-operation totals for both builds.
- `focused-tests.log`, `configuration-tests.log`: targeted memory and baseline checks.
- `regression.json`, `regression.log`, `python-tests.log`: full regression results.
- `first-regression.json`, `first-regression.log`, `first-python-tests.log`,
  `traced-retry.log`: the initial process termination and isolated retry.
- `terminal-smoke.log`: the public interactive sentence command.
- `sources.json`: source hashes for the final verified implementation.
