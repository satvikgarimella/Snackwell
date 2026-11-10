# Direct resident-memory addressing

November 9, 2026. These comparisons run `resident-copy` and `resident` with the
same checkpoint, prompts, precision and additional external latency of 2 clocks.
Both execute on the same untraced Verilator build:
`cae6f6a3a75bba5ffabfa0a34bdede7e5b2c77d7dff772e54e05dcbaa3c09abf`.
This is a kernel/runtime change; there is no new hardware opcode or simulator
address translation. File hashes are in `sources.json`.

Final regression: **46/46 RTL testbenches and 258/258 Python test methods passed,
with no skips**. The Python suite took 389.305 seconds. Ruff checks and format
checks passed for 143 Python files; `git diff --check` passed.

```sh
build/model-venv/bin/python scripts/compare_model_memory.py --resident \
  --baseline resident-copy --cases 8 --out build/inference/direct-comparison
build/model-venv/bin/python scripts/compare_model_memory.py --resident \
  --baseline resident-copy --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'ROMEO:' --tokens 4 --out build/inference/direct-character-comparison
build/model-venv/bin/python scripts/compare_model_memory.py --resident \
  --baseline resident-copy --precision fp32 --cases 2 \
  --out build/inference/direct-fp32-comparison
```

## Correctness and execution

| Measurement | NVFP4 sentence | NVFP4 character | FP32 sentence |
| --- | ---: | ---: | ---: |
| Checkpoint | `66af3342b4a6` | `43f211daad7c` | `66af3342b4a6` |
| Cases | 8 reserved questions | `ROMEO:` | 2 reserved questions |
| Identical probability vectors and selected tokens | 56/56 | 4/4 | 14/14 |
| Correct complete answers, both modes | 8/8 | No answer target | 2/2 |
| Copy-based cycles | 146,836,159 | 18,400,463 | 55,675,914 |
| Direct cycles | 71,231,426 | 7,890,415 | 24,939,722 |
| Cycle reduction | 51.5% | 57.1% | 55.2% |
| Copy-based / direct RTL launches | 32,536 / 10,024 | 3,786 / 992 | 9,532 / 2,270 |
| Copy-based / direct verified wall seconds | 98.82 / 46.94 | 13.06 / 5.68 | 29.64 / 11.78 |

Sentence questions use seed 20261109. The character continuation is `\nMar` in
both modes. The arithmetic and `nvfp4-linear-row-v1` quantization policy are
unchanged; this is a transport comparison, not a language-quality improvement.

Direct mode has **zero layout-copy launches and words**. Native NVFP4's necessary
global-to-shared operand copies remain. Each stage still performs independent
numerical checks, full external guard checks in the simulator and shared-table
checks. Every decode token uploads zero unchanged weight bytes.

For the eight NVFP4 sentence questions, average verified prefill time was
**5.49 → 2.85 seconds**, and average cached decode was **1.14 → 0.50 seconds/token**.
These are simulation timings including host quantization, pointer-table creation,
numerical references and memory guards, under ordinary system load. They are not
FPGA throughput estimates or isolated benchmarks.

## Traffic and storage

The eight-question NVFP4 sentence comparison records:

| Metric | `resident-copy` | `resident` |
| --- | ---: | ---: |
| External host upload payload bytes | 6,457,964 | 869,248 |
| External readback payload bytes | 43,569,888 | 2,511,584 |
| Complete protocol request bytes | 275,706,084 | 83,603,824 |
| Complete protocol response bytes | 211,846,080 | 54,355,712 |
| SM external reads | 8,064,296 | 5,122,960 |
| SM external writes | 3,435,352 | 673,152 |
| SM external responses | 11,499,648 | 5,796,112 |
| Layout-copy words | 2,762,200 | 0 |
| Retained weight bytes across 8 independent predictions | 398,912 | 398,912 |
| Peak live allocation bytes, including padding | 110,464 | 114,592 |

Direct readbacks contain physical output words; the old path also returned the
logical kernel window and scatter results. Guard checking remains complete on
the simulator side. Protocol counts include shared address tables, descriptors,
programs, registers and frame metadata, beyond the external data payloads.

The allocator still reserves the bottom 64 KiB for the old kernel window. Add
that reservation to the live allocation total; neither number counts free holes
or establishes a minimum RAM size. Both paths use a configured 1 MiB simulated
external RAM. This does not establish FPGA fit or board operation.

Direct accesses still require shared pointer lookups, and the external port
serves one word at a time. The archived full-image global path used 67,163,967
cycles for the same NVFP4 questions; direct mode's 71,231,426 cycles are 6.1%
higher, even though it avoids repeated host tensor uploads. Further optimization
needs address-lookup reduction and improved memory service, not the return of
full-image transfers.

## Evidence files

- `sentence-comparison.json`, `character-comparison.json`, `fp32-comparison.json`:
  per-token equality, phase, cycles, wall times, payloads and allocation peaks.
- `focused-tests.log`: 16 direct/resident tests, including alias preservation,
  indirect embedding, wide kernels, multi-pass outputs, register-liveness
  rejection, maximum tensor jobs, corruption, cancellation and context rebuilds.
- `terminal-smoke.log`: the public terminal command produces the kitchen answer.
- `regression.json`, `regression.log`, `python-tests.log`: the final full suite.
- `sources.json`: the checked hardware/runtime/test source hashes.
