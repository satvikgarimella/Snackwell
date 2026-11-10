"""Persistent external allocations and explicit RTL gather/scatter kernels.

DeviceWord is a verified host shadow plus an allocation reference. Python list
slices/reshapes preserve these references. Consumers copy from the device address,
never upload the shadow value. Host quantization produces ordinary words and must
upload them. All arithmetic and device copies execute through the SM's real port.
"""

import json
import weakref
from functools import lru_cache

from snackwell.assembler import assemble
from snackwell.numeric import SENTINEL
from snackwell.simulator import write_hex

CAPACITY = 262144  # 1 MiB simulated external RAM, not FPGA block RAM.
WINDOW = 16384


class Allocation:
    def __init__(self, owner, identity, start, count):
        self.owner = weakref.ref(owner)
        self.identity, self.start, self.count = identity, start, count
        weakref.finalize(self, owner.release, identity)


class DeviceWord(int):
    def __new__(cls, value, allocation, offset):
        word = int.__new__(cls, value)
        word.allocation, word.offset = allocation, offset
        return word


@lru_cache(maxsize=1)
def copy_program():
    # One active lane copies arbitrary spans, including odd sizes. Descriptors
    # contain source byte address, destination byte address and word count.
    return assemble("""
        XOR R15, R15, R15
        XOR R10, R10, R10
        LD_SHARED R11, [R15 + 4092]
    span:
        LD_SHARED R0, [R10]
        LD_SHARED R1, [R10 + 4]
        LD_SHARED R2, [R10 + 8]
    word:
        LD_GLOBAL R3, [R0]
        ST_GLOBAL [R1], R3
        ADDI R0, R0, 4
        ADDI R1, R1, 4
        ADDI R2, R2, -1
        BEQ R2, R15, next
        BEQ R15, R15, word
    next:
        ADDI R10, R10, 12
        ADDI R11, R11, -1
        BEQ R11, R15, done
        BEQ R15, R15, span
    done:
        EXIT
    """)


class ResidentMemory:
    def __init__(self, session):
        if session.trace or session.direct_executable is None:
            raise ValueError("resident memory requires untraced direct-clock Verilator")
        self.session = session
        self.alive = True
        self.identity = object()
        self.binding = None
        self.shadow = [SENTINEL] * CAPACITY
        self.free = [(WINDOW, CAPACITY - WINDOW)]
        self.live, self.constants = {}, {}
        self.pending_uploads = []
        self.serial = self.peak_words = self.allocated_words = 0
        self.uploaded_constant_bytes = 0
        self.totals = {}

    def invalidate(self):
        self.alive = False
        self.constants.clear()
        self.pending_uploads.clear()
        self.live.clear()

    def bind(self, signature, precision):
        binding = (signature, precision, self.session.source_sha256)
        if not self.alive or self.binding not in (None, binding):
            raise ValueError(
                "resident memory belongs to a different model, precision or RTL session"
            )
        self.binding = binding

    def allocate(self, count):
        if not self.alive:
            raise ValueError("resident session is closed")
        if type(count) is not int or count < 1:
            raise ValueError("allocation must contain at least one word")
        # Four guard words on each side, plus alignment padding. The transport
        # checks every word outside declared destinations after every launch.
        size = ((count + 3) & ~3) + 8
        for i, (start, available) in enumerate(self.free):
            if available < size:
                continue
            self.free[i : i + 1] = [(start + size, available - size)] if available > size else []
            self.serial += 1
            self.live[self.serial] = (start, size)
            self.allocated_words += size
            self.peak_words = max(self.peak_words, self.allocated_words)
            return Allocation(self, self.serial, start + 4, count)
        raise MemoryError("resident external RAM exhausted (1 MiB); no implicit host fallback")

    def release(self, identity):
        block = self.live.pop(identity, None)
        if block is None:
            return
        self.allocated_words -= block[1]
        self.free.append(block)
        merged = []
        for start, count in sorted(self.free):
            if merged and sum(merged[-1]) == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + count)
            else:
                merged.append((start, count))
        self.free = merged

    def words(self, allocation, values):
        return [DeviceWord(w, allocation, i) for i, w in enumerate(values)]

    def constant(self, key, values):
        """Upload one immutable weight tensor/packed tile once per prediction."""
        raw = tuple(map(int, values))
        if key in self.constants:
            existing = self.constants[key]
            if tuple(existing) != raw:
                raise ValueError("resident constant changed without a new model binding")
            return existing
        allocation = self.allocate(len(raw))
        result = self.words(allocation, raw)
        self.pending_uploads.append((allocation.start, list(raw)))
        self.shadow[allocation.start : allocation.start + len(raw)] = raw
        self.constants[key] = result
        self.uploaded_constant_bytes += len(raw) * 4
        return result

    def address(self, word):
        allocation = word.allocation
        if (
            not self.alive
            or allocation.owner() is not self
            or allocation.identity not in self.live
            or not 0 <= word.offset < allocation.count
        ):
            raise ValueError("stale or foreign resident word")
        address = allocation.start + word.offset
        if self.shadow[address] != int(word):
            raise ValueError("resident shadow does not match allocation")
        return address

    def execute(self, program, shared, uploads, readback, writable, **options):
        if not self.alive:
            raise ValueError("resident session is closed")
        result = self.session.run_resident_words(
            program, shared, uploads, readback, writable, **options
        )
        for start, data in uploads:
            self.shadow[start : start + len(data)] = data
        offset = 0
        for start, count in readback:
            self.shadow[start : start + count] = result[2][offset : offset + count]
            offset += count
        for key, value in result[3].items():
            self.totals[key] = self.totals.get(key, 0) + value
        self.totals["rtl_launches"] = self.totals.get("rtl_launches", 0) + 1
        return result

    def copy(self, spans, uploads, *, latency, readback=()):
        metrics = []
        expected_readback = None
        if readback:
            destinations = {t + i: self.shadow[s + i] for s, t, n in spans for i in range(n)}
            expected_readback = [
                destinations.get(i, self.shadow[i])
                for start, size in readback
                for i in range(start, start + size)
            ]
        # A batch also bounds the cycle limit at high external-memory latency.
        batches, batch, count = [], [], 0
        for source, target, size in spans:
            for offset in range(0, size, 128):
                part = min(128, size - offset)
                if len(batch) == 256 or count + part > 256:
                    batches.append(batch)
                    batch, count = [], 0
                batch.append((source + offset, target + offset, part))
                count += part
        if batch:
            batches.append(batch)
        for batch in batches:
            shared = [SENTINEL] * 1024
            shared[: 3 * len(batch)] = [w for s, t, n in batch for w in (s * 4, t * 4, n)]
            shared[-1] = len(batch)
            last = batch is batches[-1]
            shared_after, _, data, metric = self.execute(
                copy_program(),
                shared,
                uploads,
                readback if last else [],
                [(t, n) for _, t, n in batch],
                active_masks=1,
                latency=latency,
                max_cycles=1000000,
            )
            if (
                shared_after != shared
                or metric["reads"] != sum(n for _, _, n in batch)
                or metric["writes"] != metric["reads"]
            ):
                raise ValueError("resident copy changed its descriptors or transfer count")
            if last and readback and data != expected_readback:
                raise ValueError("resident RTL scatter copy mismatch")
            uploads = []
            for source, target, size in batch:
                # Copy oracle updates a host shadow only; RTL moved the data.
                wanted = self.shadow[source : source + size]
                self.shadow[target : target + size] = wanted
            metrics.append(metric)
        return metrics

    def run(self, program, external, output_span, directory, *, active_masks, warp_mask, latency):
        """Gather existing allocations, execute a model kernel, retain its output."""
        uploads = self.pending_uploads
        self.pending_uploads = []
        spans = []
        for i, word in enumerate(external):
            if isinstance(word, DeviceWord):
                source = self.address(word)
                if (
                    spans
                    and spans[-1][0] + spans[-1][2] == source
                    and spans[-1][1] + spans[-1][2] == i
                ):
                    s, t, n = spans[-1]
                    spans[-1] = (s, t, n + 1)
                else:
                    spans.append((source, i, 1))
            elif self.shadow[i] != word:
                if uploads and uploads[-1][0] + len(uploads[-1][1]) == i:
                    uploads[-1][1].append(int(word))
                else:
                    uploads.append((i, [int(word)]))
        start, count = output_span
        allocation = self.allocate(count)
        totals_before = dict(self.totals)
        try:
            copies = self.copy(spans, uploads, latency=latency) if spans else []
            shared, registers, after, metrics = self.execute(
                program,
                [SENTINEL] * 1024,
                [] if spans else uploads,
                [(0, 1024)],
                [(start, count)],
                active_masks=active_masks,
                warp_mask=warp_mask,
                latency=latency,
                max_cycles=1000000,
            )
            values = after[start : start + count]
            copies.extend(
                self.copy(
                    [(start, allocation.start, count)],
                    [],
                    latency=latency,
                    readback=[(allocation.start, count)],
                )
            )
            if self.shadow[allocation.start : allocation.start + count] != values:
                raise ValueError("resident RTL output copy mismatch")
        except BaseException as error:
            directory.mkdir(parents=True, exist_ok=True)
            write_hex(directory / "resident-shadow.hex", self.shadow)
            (directory / "resident-failure.json").write_text(
                json.dumps(
                    dict(
                        error=str(error),
                        gather_spans=spans,
                        output_span=output_span,
                        uploads=uploads,
                        capacity_words=CAPACITY,
                    ),
                    indent=2,
                )
                + "\n"
            )
            self.session.close()
            raise
        transfer = dict(
            metrics,
            transport="resident_external_memory",
            latency=latency,
            hardware_tested=False,
            guard_words_checked=CAPACITY - count,
        )
        total = {k: v - totals_before.get(k, 0) for k, v in self.totals.items()}
        transfer.update(
            total_cycles=total["cycles"],
            copy_cycles=sum(m["cycles"] for m in copies),
            copy_launches=len(copies),
            device_copy_words=sum(m["writes"] for m in copies),
            total=total,
            capacity_bytes=CAPACITY * 4,
            peak_allocation_bytes=self.peak_words * 4,
        )
        return shared, registers, after, transfer, self.words(allocation, values)


def resident_memory(session):
    if session.process is not None and session.process_mode != "+RESIDENT":
        session.close()
    if session.resident is None:
        session.resident = ResidentMemory(session)
    return session.resident
