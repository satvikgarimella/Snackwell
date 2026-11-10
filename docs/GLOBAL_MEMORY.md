# Device/global memory

Implemented November 7, 2026. This is Snackwell's reduced memory-master
interface, not an NVIDIA bus or a physical SDRAM controller.

**November 9 extension:** the [async-copy engine](ASYNC_COPY.md) now shares this
external port with ordinary loads/stores through a round-robin arbiter. It adds
rectangular copies, protected shared staging, and a measured double-buffered
kernel. The descriptions of ordinary global instructions below still apply.

Kernels can now load and store words outside the SM's 4 KiB shared memory. A
new LSU retains the warp, destination, active lanes, base addresses, offset,
and store data until completion. The master port is exposed by the execution
core, `snackwell_sm`, and `snackwell_device`.

```text
External memory <-> global LSU <-> warp registers <-> shared memory
                                          |               |
                                    integer / FP32    4x4 array
                                                          |
                                                     tensor memory
```

## Instructions and addresses

| Instruction | Opcode | Meaning |
| --- | --- | --- |
| `LD_GLOBAL Rd, [Ra +/- offset]` | `0x1d` | Load one 32-bit word per active lane |
| `ST_GLOBAL [Ra +/- offset], Rb` | `0x1e` | Store one 32-bit word per active lane |
| `GSTATUS Rd` | `0x1f` | Return the issuing warp's sticky error bits to active lanes |

The signed 14-bit offset is in **bytes**. Each lane's base is an unsigned
32-bit byte address. Addition uses a 33-bit intermediate; negative results
and results above `0xffffffff` fail instead of wrapping. Valid word addresses
are four-byte aligned, from `0` through `0xfffffffc`. Shared and global
addresses name separate spaces; global address `0x1000` is unrelated to the
host's program window at `0x1000`.

The backing endpoint decides which addresses are mapped. The simulation RAM
covers `0x00000000`–`0x0000ffff` (64 KiB), and returns an error outside that
range. This capacity is a test configuration; the RTL does not contain or
allocate that RAM. The master forwards all 32 address bits.

Inactive lanes make no requests, raise no errors, and preserve their destination
registers. An empty mask completes normally. Loads to the same address within
one instruction share one external read; the response is broadcast to all
matching active lanes. Distinct address groups are serviced in the order of
their first active lane. Duplicate stores remain separate requests in lane
order, 0 through 3; the highest active lane wins when all stores succeed.

This port targets ordinary RAM. Broadcast loads do not provide per-lane volatile
reads or support memory-mapped devices whose reads have side effects. The host
register bus is separate. There is no cross-instruction cache: a later load
reads memory again, including after a store. Adjacent addresses still require
separate requests; bursts, cache-line coalescing, atomics and byte/halfword
accesses remain unimplemented.

The `BROADCAST_LOADS` parameter defaults to enabled. Compiling with
`SNACKWELL_SERIAL_GLOBAL_LOADS` selects the original serial behavior for controlled
comparisons. `python3 scripts/compare_global_loads.py` builds both configurations
and compares the same resident model kernels. Their build identities differ so
KV state cannot silently cross configurations. See the
[measurements and reproduction commands](measurements/load-broadcast-20261109/README.md).

## Errors and completion

| `GSTATUS` bit | Meaning |
| --- | --- |
| 0 | Misaligned effective address |
| 1 | Effective address underflow/overflow |
| 2 | External endpoint returned `response_error` |

Address errors produce no external request. A failed load writes zero to all
active lanes sharing its request; a failed store skips that lane. Other address
groups continue. An endpoint
must not modify memory for a store response that reports an error. A vector
operation is not atomic: successful stores are not rolled back if another lane
fails. Alignment and range errors can both be recorded for the same lane.

Errors accumulate separately per warp until core reset. `GSTATUS` waits behind
the current global instruction, so it observes all its errors. On the device
host bus, read `0x0020` for all four statuses, three bits per warp (warp 0 in
bits 2:0). A normally terminating kernel with any global error finishes with
`fault=1, done=0`. Diagnostics remain readable until reset or the next launch.
The kernel must inspect status if it wants to handle errors before termination.

Loads and `GSTATUS` reserve their destinations in the existing scoreboard.
Dependent instructions wait for writeback; independent integer, FP32, shared,
and tensor work can run while the global operation waits. The global result
holds its captured data and metadata under writeback backpressure. Priority is
FP32, legacy tensor row, tensor-memory row, shared load, global load/status,
then integer. Stores complete only after all active lanes have responses.
`EXIT` waits for the issuing warp's pending global instruction, and device
completion waits for the SM to drain.

## External handshake

All signals use the SM clock. There is one outstanding word transaction and
one global instruction in flight for the entire SM.

| Signal | Direction from SM | Contract |
| --- | --- | --- |
| `global_request_valid` | Output | A request is available |
| `global_request_ready` | Input | Endpoint accepts it on this rising edge |
| `global_request_write` | Output | 0 = read, 1 = write |
| `global_request_address` | Output | Full 32-bit aligned byte address |
| `global_request_data` | Output | Full 32-bit store value |
| `global_response_valid` | Input | Endpoint has completed the accepted request |
| `global_response_ready` | Output | SM accepts the response on this rising edge |
| `global_response_error` | Input | Request failed |
| `global_response_data` | Input | Loaded word; ignored for stores or errors |

Acceptance requires both valid and ready. Requests stay stable until accepted;
responses must stay stable until consumed. Every accepted request, including a
store, requires exactly one response. A successful store response guarantees
that subsequent requests see the write. An immediate responder must hold its
response until the SM asserts ready; it cannot send a one-cycle pulse and drop
it while ready is low. There are no transaction IDs because only one request
is outstanding.

Global operations complete in issue order across this single SM. This does
not establish a total order with the separate shared-memory path. Cross-warp
publication needs an explicit synchronization protocol. Do not signal a shared
flag immediately after issuing a global store and assume the store finished;
waiting through `GSTATUS` first drains earlier global operations. General block
barriers and asynchronous-copy completion are separate work.

## Abort, watchdog and reset

A software abort or watchdog stops instruction issue, then drains the **entire
captured global vector**, including lanes whose requests have not yet been
accepted. Only then does the device reset the core and permit relaunch. Busy
stays high throughout draining; a new start is rejected. Already accepted stores
cannot be undone. Shared/tensor computations can be cancelled by the ensuing
core reset, so an aborted kernel's outputs must be treated as incomplete.

The external endpoint must eventually respond for this drain to finish. The
watchdog limits instruction execution cycles, not the time needed to drain a
nonresponsive memory system. Recovering from an unresponsive endpoint requires
a **coordinated system reset of both sides**. That reset must discard all pending
endpoint responses as it resets the master. Resetting only the core and later
delivering an old response violates the interface contract. Stores committed
before reset persist; unacknowledged stores may already have changed memory.

Ordinary launch/abort resets are internal to the device, and never cancel an
outstanding external request. The board integration must distribute external
system reset to both the device and its memory adapter.

## Run the full path

```sh
python3 scripts/run_global_demo.py
```

`kernels/global_tensor.asm` computes `A[4,8] @ B[8,4]` with:

- A and B at global byte addresses `0x4000` and `0x5000`.
- Ordinary global loads and shared stores to stage A/B at shared bytes 0/128.
- `TMMA` and `TMMA_ACC` using the existing array and persistent tensor tile.
- Commit/wait/release for both matrix jobs.
- Four row loads from tensor memory and global stores to `0x6000`.

The host loads the program/shared descriptors through the public host bus. A
simulation-only external RAM supplies delayed responses and periodically blocks
requests. It performs no matrix arithmetic. The reference independently computes
the FP32 result, and verification checks every shared/global word, both status
vectors, 64 word reads, 16 word writes, and two matrix jobs. The default case
takes **1,013 simulated cycles** with this memory timing. That is not a board
latency or a performance comparison against B200.

Artifacts go to `build/kernels/global_tensor/`: input/output memory images,
`program.hex`, `trace.csv`, simulator logs, and `result.json`. The trace records
actual instruction and memory handshakes. `snackwell/global_memory.py` also runs
other assembled programs against the same endpoint, including masked multiwarp
programs. Inference now has an explicit [`--memory global` mode](GLOBAL_MODEL.md)
that uses this interface for all kernel data, with asynchronous staging for
NVFP4 tiles. Shared-memory launch transport remains the default, and training
retains its existing transport.
Legacy runners without an external endpoint return an external-access error
if a program issues a global access. Use the global runner for this new path.

## Verification and remaining work

The [recorded regression](measurements/global-memory-20261107/README.md) passed
43 RTL testbenches and 204 Python methods in one full run, with no skips.

- `tb_global_memory_unit`: captured operands, masks, negative offsets, upper
  addresses, overflow, errors, all masks of interleaved broadcast groups,
  duplicate stores, held completion, and reset.
- `tb_snackwell_sm_global`: a full writeback slot blocked by debug access, correct
  warp/lane delivery, dependent FP execution, and exit draining.
- `tb_snackwell_global_memory`: host-bus launches, request and response delays,
  abort before/after request acceptance, watchdog, coordinated reset, and relaunch.
- `tests/test_global_memory.py`: random gather/scatter, complete memory guards,
  random FP32 tensor results at several latencies, error reporting, store/load
  ordering, independent warp progress, FP/shared/global contention, failed-run
  artifact handling, and Icarus/Verilator parity.
- `tests/test_global_loads.py`: serial/broadcast build comparisons, unchanged
  distinct-load timing, request reductions, and rejection of mid-prediction
  build changes. Latest full regression is saved with the
  [broadcast measurements](measurements/load-broadcast-20261109/README.md).

The subsequent async-copy engine, shared-buffer ownership, and double-buffering
milestone is documented separately above. Coalescing and a cache remain open. A physical board still needs
a memory adapter/controller, host transport, pin/timing constraints, and a new
synthesis/fit run. The older resource estimate does not include these extensions.
