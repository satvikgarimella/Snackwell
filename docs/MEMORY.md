# Shared memory in Snackwell

`LD_SHARED` and `ST_SHARED` now complete every active lane's access, including
bank conflicts. The SM captures a memory instruction once, advances its PC,
and lets the memory unit serve the captured lanes over several cycles.

## Addressing and ordering

Shared memory remains 4 KB: four banks, each holding 256 32-bit words.
The LSU adds the sign-extended 14-bit instruction immediate to each lane's
base register. Byte address bits `[3:2]` select the bank and `[11:4]` select
its row. Software should use aligned addresses from 0 through 4092; the current
hardware still ignores bits `[1:0]` and bits above bit 11, with no alignment or
bounds trap. No opcodes or instruction encodings changed.

Each bank serves its lowest-numbered pending lane on each service cycle.
Different banks work in parallel. Consequently:

- Four lanes using four different banks finish in one service cycle.
- Two, three, or four different words in one bank take two, three, or four service cycles.
- Read-only requests broadcast each selected word to all matching active lanes
  in one service cycle. Such repeated reads do not count as bank conflicts.
- Repeated stores to the same word commit in ascending lane order; the
  highest-numbered active lane supplies the final word.
- Inactive lanes neither access memory nor create conflicts.
- An empty-mask instruction still completes, with no memory changes.

There is one outstanding shared-memory request across the SM. Memory requests
are served in instruction acceptance order, including across warps. A younger
load cannot overtake an accepted store. This ordering does not provide a block
barrier: cooperating warps still need explicit synchronization in a later version.

## Request and completion interface

`rtl/memory/shared_memory.sv` exposes:

- `request_valid` / `request_ready`: acceptance of addresses, store values,
  `load_enable_mask`, and `store_enable_mask` on a rising clock edge.
- `response_valid` / `response_ready`: completion of the entire request.
  Stores produce a completion as well as loads.
- `load_data`: the complete four-lane result. Inactive/non-load lanes are zero.
  Consumers use it only with `response_valid`; it is stable under backpressure.
- `conflict_mask` / `bank_conflict`: conflicts in the offered request when
  ready, or among the remaining lanes during service; zero with no request
  or while holding a completed response.

The request is snapshotted at acceptance. Live inputs may change while the unit
works. No further request is accepted until the previous response has been
consumed. After capture, service takes one to four clocks (one for an empty
request); the response handshake and SM writeback add completion latency.

The SM issues exclusively loads or stores. At the standalone interface, a lane
with both masks set reads its old word and then writes the supplied new word
at its service edge. Any request containing a store disables broadcast, so lanes
sharing a bank still execute in ascending order, including reads around a store.

## SM integration

`multiwarp_execution_core` saves the warp, destination register, active mask,
and load/store kind alongside each accepted request. The memory unit saves
addresses and store operands, so subsequent register writes, scheduling, or
branch-mask changes cannot alter the operation.

Loads reserve their destination in the existing scoreboard. Consumers and
other writers of that register wait until the complete vector actually commits
to the register file. There are no partial per-lane register writes during replay.
The saved active mask preserves inactive destination lanes.

Writeback priority is completed FP, a ready tensor row, a completed shared load,
then a newly issued integer result. Completed loads wait when higher-priority
results or a held writeback occupy the port. Independent arithmetic and control
instructions can issue during memory service; later memory instructions wait
for the shared-memory unit to become ready.

`EXIT` waits for the selected warp's pending register writes and memory request,
including stores. `run_enable=0` pauses new issue while accepted loads/stores
continue. Debug register writes backpressure architectural writeback; they do
not pause memory service or cancel stores. Debug preloads and warp relaunches
remain restricted to unused or fully retired architectural state.

Reset clears the memory banks, aborts replay/held responses, and clears the SM's
saved completion tag and scoreboard. A partially serviced store is not rolled
back independently: the existing reset behavior clears all shared memory.

## Verification

```sh
python3 scripts/run_tests.py tb_shared_memory tb_multiwarp_memory_core tb_snackwell_sm_memory
python3 scripts/run_tests.py
```

`tb_shared_memory` compares against a flat reference memory across all 256
four-lane bank mappings and all 16 active masks, for stores and loads. It also
checks service latency, changed inputs after acceptance, busy-request rejection,
response backpressure, same-word accesses, mixed load/store masks, empty masks,
and reset during service and while a response is held.

`tb_snackwell_sm_memory` checks architectural load values, masks, destinations,
completion counts, dependencies and final memory using its own reference model.
It covers four warps, every active mask, signed offsets, captured operands,
load RAW/WAW dependencies, independent arithmetic, cross-warp ordering, aliased
address/destination registers, divergence/reconvergence, store draining on EXIT,
pause/debug backpressure, reset/reuse, and FP/tensor/load writeback contention.
A complete program runs conflicting loads -> TMUL -> conflicting stores -> load
-> FADD. The previous memory smoke test now waits for actual completion instead
of assuming a single-cycle load.

This implementation is verified in RTL simulation. Global memory, barriers,
coalescing, multiple outstanding requests, and FPGA
resource/timing validation remain separate work.
