# Model inference through external memory

Updated November 9, 2026.

`--memory global` runs every inference kernel's operands, constants and results
through the SM's external-memory request/response port. Both FP32 and NVFP4
precision modes are supported. The existing shared-memory transport remains
the default for comparisons and debugging.

```sh
python3 scripts/run_sentence.py --precision nvfp4 --memory global --interactive
python3 scripts/run_model.py --precision nvfp4 --memory global \
  --prompt 'ROMEO:' --tokens 4 --decode greedy
python3 scripts/compare_model_memory.py --cases 8
```

For the sentence demo, enter:

```text
The ball is in the kitchen. The book is in the garden. Where is the ball?
```

`--memory-latency N` selects an additional response delay of 0..1000 clocks.
The endpoint also varies latency by address and periodically backpressures
requests. Large delays can exceed the existing per-kernel watchdog; they are
not silently ignored. Terminal, streaming and single-prompt modes are supported.
Browser mode and the older exported PyTorch evaluation command retain their
original transport; the comparison command above checks the new path.

## Data path

The host places a bounded kernel image in a simulated 64 KiB external RAM and
loads instructions into program memory. Shared memory starts with guard words,
not model inputs. The SM executes all arithmetic and all transfers:

- Ordinary kernels replace their shared loads/stores with `LD_GLOBAL` and
  `ST_GLOBAL`. This includes embedding lookup, FP32 matrix products, attention,
  LayerNorm, GELU, residuals, softmax and token selection. Scalar constants also
  cross the external port.
- Native NVFP4 kernels pack each A[4,16]/B[16,4] operand pair into 24 words in
  external memory. `CPASYNC`, `CPWAIT` and `CPRELEASE` alternate two protected
  shared buffers. The next pair is fetched while the current pair executes.
- `TMMA_NVFP4` and `TMMA_NVFP4_ACC` retain the accumulation in tensor memory.
  The FP32 factors, bias, and optional prior K-chunk accumulator are loaded
  globally; the final rows are stored globally. Each copy ticket is released
  after the operands have been gathered. Tensor completion is waited before
  reading its rows or reusing its accumulator.

The kernel window occupies global bytes 0..4095. Copy descriptors begin at byte
6144; packed tile pairs begin at byte 16384. The largest current native launch
has 32 pairs. Shared staging occupies bytes 0..95 and 256..351. Other shared and
global words are guards. Original unquantized operands and the independently
computed reference never replace executed RTL outputs.

This is a staged runtime. The host still tokenizes, quantizes, assembles,
launches, reshapes actual output words, and retains checkpoint and KV state.
Each kernel resets the SM and receives a new external image. The Verilator
process is reused to avoid a process launch per kernel, but weights and
activations are **not persistent device allocations in this mode** across launches.
The full checkpoint need not fit in the simulated 64 KiB staging RAM.
The separate [`--memory resident` mode](RESIDENT_MEMORY.md) now retains device
allocations across launches and tokens, accessed directly through device pointers.

## Verification and simulator behavior

The arithmetic and `nvfp4-linear-row-v1` policy are unchanged. Existing per-kernel
and whole-model numerical checks remain active. Additional checks cover:

- External/shared guards, all external requests receiving responses, and zero
  global/copy error status at completion.
- Exactly 24 copied words per native matrix job, 256 narrow MACs per job, and
  zero FP32 tensor-multiplier launches inside a native linear kernel. FP32
  attention still uses those wide multipliers.
- Identical outputs across direct Verilator, traced Verilator and Icarus
  transports. The independently clocked host-bus bench has a different request
  phase, so its cycle count can differ while arithmetic and transfer counts agree.
- Multiwarp FP32 kernels, partial lane masks, K chunk carry, row/column padding,
  serial versus overlapping copies, delayed responses, status errors, timeout
  and simulator restart, and injected memory corruption.
- Prefill, cached decode, sampling and context rebuilds. A cache cannot switch
  shared/global mode midway through a sequence, and failed checks do not commit
  new cache state.

Traced runs use the existing SystemVerilog host-bus runner. Untraced Verilator
uses a binary protocol to the same SM and external RAM model. Neither host
implementation evaluates model arithmetic. Full artifacts contain the executed
assembly, external image, sentinel shared input, outputs, registers and metrics.
Compact runs retain stage summaries and expand artifacts on failure.

## Measurements

The [saved results](measurements/global-model-20261109/README.md) use RTL/simulation
source SHA-256 `a96505f467e2ab25d15a221bec8c16234960fecc147152926cf2e39c39953a86`
and additional memory latency 2.

| Comparison | Sentence checkpoint | Character checkpoint |
| --- | ---: | ---: |
| Prompts | 8 reserved test questions, seed 20261109 | `ROMEO:` |
| Compared generated tokens | 56 | 4 |
| Identical probabilities and selected tokens | 56/56 | 4/4 |
| Correct full answers | 8/8 in both modes | No answer target |
| Shared / global cycles | 27,453,019 / 67,163,967 | 3,477,228 / 7,552,044 |
| External reads / writes | 5,302,096 / 673,152 | 625,200 / 60,160 |
| Words copied asynchronously | 1,074,816 | 243,072 |

Both sentence runs use checkpoint `66af3342b4a6`; the character run uses
`43f211daad7c`. Character continuation is `\nMar` in both memory modes. These
results compare transports at fixed NVFP4 precision, not NVFP4 language quality
against FP32. See [the precision comparison](NVFP4_INFERENCE.md) for that.

Global mode costs more cycles because transfers now incur simulated latency.
Its endpoint services one word transaction at a time. These archived runs used
serial per-lane loads; the current LSU broadcasts one read to active lanes
requesting the same address. Adjacent words still require separate requests,
and no data is cached between instructions. Recorded sentence wall times were 23.01 s
shared and 59.92 s global, including quantization and verification. They are
simulation timings under system load, not FPGA throughput measurements.

Persistent allocation and reuse now exist in [resident mode](RESIDENT_MEMORY.md).
Direct addressing now removes the original resident layout copies.
[Load broadcasting](measurements/load-broadcast-20261109/README.md) further reduces
repeated scalar traffic. The [resident compiler](RESIDENT_MEMORY.md) now removes
some pointer lookups, retains literals in shared RAM and shortens stores.
Adjacent-address coalescing, caches and a physical memory controller remain separate work.
No FPGA fit, timing closure, board execution or autonomous device inference
is established by this simulation path.
