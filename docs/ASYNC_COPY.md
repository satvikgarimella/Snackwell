# Asynchronous tile copies and double buffering

Implemented November 9, 2026. The SM can copy a rectangular block from global
memory to shared memory while executing other instructions, including matrix
arithmetic. Two staging slots track protected regions in the existing 4 KiB
shared memory. They are ownership metadata, not extra RAM.

The copy engine is a reduced global-to-shared data mover for Snackwell's
Blackwell architecture project. It uses a custom descriptor and ISA. It does
not implement NVIDIA tensor-map encodings, arbitrary tensor ranks, swizzles,
multicast, or PTX compatibility.

## Instruction contract

All four lanes of the issuing warp must participate. Instructions use the
ordinary scoreboard for returned tickets and status values.

| Instruction | Opcode | Behavior |
| --- | --- | --- |
| `CPASYNC Rt, S0, Rdesc, mask` | `0x20` | Reserve a destination region, start a copy, return its generation ticket |
| `CPWAIT S0, Rt` | `0x21` | Wait for that exact generation and acquire read access |
| `CPRELEASE S0` | `0x22` | Release this warp's consumer lease |
| `CPSTATUS Rd` | `0x23` | Read this warp's sticky copy errors |

`S0` and `S1` are the two slots. `mask` names participating consumer warps;
bits 0–3 name warps 0–3. It must be nonzero. The producer may also be a consumer
but need not be. Every named consumer must be launched and eventually wait and
release. A warp cannot `EXIT` while it holds a consumer lease or owns an active
copy. Missing participants/releases can therefore cause a watchdog timeout.

The descriptor is one **128-bit vector register**, with a different field in
each lane. For example, load four consecutive shared words into R6 using lane
addresses, then issue `CPASYNC R8, S0, R6, 1`.

| Register lane | Descriptor field |
| --- | --- |
| 0 | Unsigned 32-bit global source byte address |
| 1 | Unsigned 32-bit source row stride in bytes |
| 2 | Rows in bits 7:0, columns in bits 15:8; bits 31:16 must be zero |
| 3 | Shared byte base in bits 11:0, destination row stride in bits 23:12; bits 31:24 must be zero |

Rows and columns are each 1–16. Each element is a raw 32-bit word. Bases and
strides must be four-byte aligned, and strides must be at least `4 * columns`.
The last source byte must fit in the 32-bit address space, and the last
destination byte must be below 4096. End-address calculations check overflow
before any request. The external endpoint separately validates whether source
addresses are mapped. Descriptors and consumer masks are captured at issue;
later changes to registers do not alter the copy.

The destination's entire bounding interval is protected, including row padding.
It cannot overlap another live slot's interval. An otherwise valid `CPASYNC`
waits if its slot is still leased, the other slot overlaps, the engine is busy,
or an earlier shared operation/tensor operand gather has not drained. Invalid
descriptors retire with a zero ticket and an error; they never reserve a region
or send a request.

Each slot has a 32-bit generation counter. Reuse increments it; releasing the
last consumer does not reset it. Exhaustion is rejected instead of wrapping.
Tickets are replicated across all four lanes; `CPWAIT` requires a uniform
ticket. Zero, an older generation, or a ticket for a released generation is an
error. A future generation waits, which allows a consumer warp to arrive before
the producer. Reset invalidates all software tickets and clears generations.

Encoding: `CPASYNC` puts the ticket register in Rd, descriptor register in Rs1,
slot in immediate bit 0, and participant mask in immediate bits 4:1. Rs2 and
remaining immediate bits are reserved zero. `CPWAIT` uses Rs1 for its ticket
and immediate bit 0 for its slot, with other fields zero. `CPRELEASE` uses only
immediate bit 0. `CPSTATUS` uses only Rd. Invalid reserved fields report an error.

## Visibility and safe reuse

```text
CPASYNC reserves the region and returns a ticket
    -> external reads and shared writes run independently
    -> last shared write is acknowledged
    -> CPWAIT grants that consumer read access
    -> shared loads or TMMA gather the operands
    -> CPRELEASE drops that consumer's lease
    -> after the last release, the slot/region may be reused
```

Ordinary shared loads and tensor matrix operations that touch a live region wait until
the issuing warp successfully waits for the copy. Shared stores into the live
interval wait until the region is released by every consumer, including when
the writer is the producer. The guards respect legacy shared-memory address
aliasing, so higher address bits or low unaligned bits cannot bypass them.
Unrelated regions remain usable. Invalid tensor instructions still report their
normal tensor error instead of becoming stuck behind a copy guard.
For [NVFP4 operations](NVFP4.md), the protected operand interval includes all
packed values and scale words, using the same descriptor validation as the
tensor controller.

`CPRELEASE` waits for previously accepted ordinary shared operations and the
current tensor operand gather to finish. The existing tensor unit captures A
and B before starting arithmetic, so shared staging can then be released and
reused while that matrix job computes. This ownership is separate from the
tensor accumulator's `TCOMMIT`/`TWAIT`/`TRELEASE` protocol.

A nonparticipant cannot acquire a live region through `CPWAIT`. After the last
consumer releases, the region becomes ordinary shared memory again; its bytes
persist, but there is no longer an ownership guarantee. Programs must not
mistake that persistence for a continuing lease.

## Transfers, arbitration and errors

One copy rectangle runs at a time. The engine reads and writes in row-major
order, buffering one word between the external response and shared write. It
waits for each shared write acknowledgement before advancing. This first
implementation does not use bursts or coalesce transfers.

The copy engine and normal global LSU share one outstanding external word
transaction. A round-robin arbiter retains the selected client through request
backpressure and response acceptance. Shared memory retains its existing bank
controller; tensor gathering has priority over a copy write, and copy writes
have priority over new ordinary shared requests. Existing responses always go
to their captured owner.

The host must keep source data stable for the duration of a copy. Capturing a
descriptor does not snapshot global memory. Concurrent ordinary stores into a
copy's source create a data race. `GSTATUS` drains ordinary global instructions,
not asynchronous copies; use the copy's ticket. `CPSTATUS` reads error state and
is not itself a completion fence.

| `CPSTATUS` bit | Meaning |
| --- | --- |
| 0 | Invalid encoding, partial warp, or nonuniform wait ticket |
| 1 | Warp is not a named/unreleased consumer |
| 2 | Invalid source/destination descriptor |
| 3 | Zero, stale, or released ticket |
| 4 | External memory read failed |
| 5 | Generation counter exhausted |
| 6 | Release attempted before wait |

An external read error ends that copy after consuming the failed response. Any
earlier shared writes remain; later words are not written. The region stays
protected and cannot be read as a completed tile. Consumers can `CPWAIT`, inspect
status, and `CPRELEASE` to recover; waiting propagates the failure to each
consumer's status. They must branch around computation using a failed tile.
The producer records the error when the read fails.

Read host register `0x0024` for all four statuses, eight bits per warp with warp
0 in bits 7:0. Any copy error makes normal device termination report fault
instead of done. Status is sticky until core reset. Tensor status remains
separate and must still be checked through `TSTATUS`.

Software abort/watchdog stop new instructions, drain the captured copy rectangle
(or its failed-read termination) and any normal global instruction, then reset
execution and leases. Shared write acknowledgements are included in draining.
Busy remains set until this finishes; relaunch is rejected meanwhile. Draining
requires the external endpoint to respond. System reset must reset both sides
of the external link together; it cancels pending protocol state but does not
roll back writes already made. See [global memory](GLOBAL_MEMORY.md).

## Double-buffered matrix demo

```sh
python3 scripts/run_async_demo.py
```

`snackwell/kernels/async_tensor.py` emits both schedules for the same
`A[4,12] @ B[12,4]` product. Each copy stages two 4×4 operand tiles as one 8×4
rectangle. Slots alternate between shared bases 0 and 256, reusing S0 for the
third matrix job. Three matrix jobs accumulate into T0 with the existing
separately rounded FP32 multiply/add contract.

The serial version completes each matrix job before submitting the next copy.
The overlapped version submits the next copy before the current matrix job,
waits/releases the proper generations, and lets copying continue during
arithmetic. Both execute the same arithmetic, copy the same data, and use the
same configured external latency/backpressure pattern. This compares scheduling
with the same copy engine; it is not a comparison against an FPGA or B200.

Default measured result, with two additional endpoint-delay cycles:

| Schedule | Simulated cycles | Copy transfers during PE arithmetic |
| --- | ---: | ---: |
| Serial | 1,582 | 0 |
| Double buffered | 1,412 | 25 |

That is **170 fewer cycles (10.7%)**. Copy work and PE arithmetic are active
together for 139 cycles. Actual read/write handshakes establish that overlap;
an asserted busy flag alone is not the evidence. Different shapes, latencies,
and contention can give different improvements.

Both versions verify all 16 output words against an independent exact FP32
reference, every shared/global guard word, status, ticket generations, three
matrix jobs, 96 external reads, 16 output stores, and 96 shared copy writes.
Artifacts are in `build/kernels/async_tensor/{serial,overlapped}/`, including
readable `program.asm`, memory images, and cycle traces. The parent
`result.json` records the comparison. `--latency N` changes the external delay.

## Verification and scheduler correction

The asynchronous-copy milestone's November 9 regression passed **45 RTL testbenches and 215 Python test
methods, with no skips**, including the existing inference and training checks.
The [saved evidence](measurements/async-copy-20261109/README.md) includes logs,
both demo traces, source hashes, and a fresh-build Verilator parity check.

`tb_async_copy_unit` checks descriptor/operand capture, held requests/results,
future/stale/exhausted generations, multiple consumers, access guards, failed
reads, and reset. `tb_global_memory_arbiter` checks held grants, response routing,
and round-robin progress. The public device-bus bench also aborts copies before
and after request acceptance, exercises watchdog and coordinated reset, and
checks that all shared writes drain before relaunch.

`tests/test_async_copy.py` checks rectangular strides up to 16×16, random matrix
products and schedules at multiple latencies, region guards through the full
SM, multiple consumer warps, buffer reuse during matrix computation, malformed
instructions/descriptors, failed copies, normal-LSU contention, and Verilator
parity. It checks complete memory images, not just selected outputs.

The two-consumer test exposed an existing scheduler starvation case: when two
warps alternated resource stalls, the round-robin pointer stayed fixed and could
prevent a third eligible warp from releasing their resource. The scheduler now
advances on an explicit resource skip as well as accepted issue. Skipping does
not execute an instruction or change its PC/scoreboard. The cooperative copy
test retains this case as a regression.

Existing inference/training kernels still use the earlier launch/data path.
The later [NVFP4 extension](NVFP4.md) also uses this copy path. Remapping full
model workloads, quantized-model quality, further optimization, synthesis/fit
of the extended hardware, and a physical FPGA memory/host interface remain open.
