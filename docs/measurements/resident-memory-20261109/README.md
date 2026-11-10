# Resident external-memory inference evidence

These historical measurements describe the original gather/scatter implementation,
now selected with `--memory resident-copy`. The commands below record the flags
at capture time. Current `--memory resident` uses the separately measured
[direct-addressing implementation](../direct-memory-20261109/README.md).

November 9, 2026. Untraced Verilator, native NVFP4 linears, additional external
response latency 2. RTL/simulator source SHA-256:
`cae6f6a3a75bba5ffabfa0a34bdede7e5b2c77d7dff772e54e05dcbaa3c09abf`.
Python and hardware file hashes are recorded in `sources.json`.

Final regression: **46/46 RTL testbenches and 250/250 Python test methods
passed, with no skips**. The Python suite took 379.164 seconds. Ruff checks and
format checks passed for 140 Python files; `git diff --check` passed.

```sh
build/model-venv/bin/python scripts/compare_model_memory.py --resident --cases 8 \
  --out build/inference/resident-comparison
build/model-venv/bin/python scripts/compare_model_memory.py --resident \
  --checkpoint models/character_transformer/checkpoint.json \
  --prompt 'ROMEO:' --tokens 4 --out build/inference/resident-character-comparison
build/model-venv/bin/python scripts/run_tests.py
```

| Result | Sentence model | Character model |
| --- | ---: | ---: |
| Checkpoint | `66af3342b4a6` | `43f211daad7c` |
| Cases | 8 reserved questions, seed 20261109 | `ROMEO:` |
| Identical probability vectors and selections | 56/56 | 4/4 |
| Correct complete answers | 8/8 in both modes | No answer target |
| Global cycles | 67,163,967 | 7,552,044 |
| Resident cycles, including layout copies | 146,836,159 | 18,400,463 |
| Layout-copy cycles | 79,672,192 | 10,848,419 |
| Total resident RTL launches | 32,536 | 3,786 |
| Global / resident verified wall seconds | 61.67 / 96.68 | 7.96 / 12.27 |
| Peak live allocation bytes, excluding fixed window | 110,464 | 197,792 |

The character continuation is `\nMar` in both modes. These comparisons measure
memory-transport equivalence, not improved language quality. The arithmetic and
quantization policy are unchanged.

For the eight sentence predictions:

- Full-image global uploads and readbacks: **656,932,864 bytes each**.
- Resident external uploads: **6,457,964 bytes**, a **99.0% reduction**.
- Resident external readbacks: **43,569,888 bytes**, a **93.4% reduction**.
- Resident protocol requests: **275,706,084 bytes**; responses: **211,846,080 bytes**.
  These larger counts include program images, shared copy descriptors, registers
  and frame metadata. The payload reduction is not a reduction of all wire bytes
  by the same percentage.
- Retained weight uploads: **49,864 bytes per prediction**; **zero on every
  subsequent token**. Each independent question starts a fresh session.
- Device layout copies: **2,762,200 words**. Resident totals include
  **8,064,296 reads + 3,435,352 writes = 11,499,648 responses**.

The 64 KiB kernel window and live allocations fit inside the configured 1 MiB
simulation capacity for these cases. Allocation sums exclude free holes; they
are not a minimum-capacity estimate or an FPGA placement result.

The resident path trades host uploads for extra SM traffic. It is slower here.
Generating kernels that directly address retained tensors is the next step to
remove those copies. Wall times include host quantization, independent reference
checks, whole-memory guard scans and other system load; they are not isolated
throughput measurements or hardware timings.

Files:

- `sentence-comparison.json` and `character-comparison.json`: per-token equality,
  cycles, payloads, protocol bytes, allocation peaks and answer checks.
- `focused-tests.log`: resident transport, dataflow, ownership, corruption,
  cancellation, allocation and full-model tests.
- `terminal-smoke.log`: the public CLI generated the expected kitchen answer.
- `regression.json`, `regression.log`, `python-tests.log`: final full suite.
- `initial-integration.log`: an initial retry regression in the older shared
  mode, subsequently fixed and covered by the final full suite.
