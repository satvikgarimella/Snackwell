"""Plan and verify a model launch that accesses retained tensors in place."""

import json

from snackwell.kernels.resident import OUTPUT_DELTA, literal_loads, lower, native_program
from snackwell.numeric import SENTINEL
from snackwell.resident_memory import CAPACITY, DeviceWord
from snackwell.simulator import write_hex


def spans(addresses):
    """Coalesce ordered word addresses without changing their logical order."""
    result = []
    for address in addresses:
        if result and sum(result[-1]) == address:
            result[-1] = (result[-1][0], result[-1][1] + 1)
        else:
            result.append((address, 1))
    return result


class DirectLaunch:
    def __init__(self, device, layout, source, operation, params, output_span, *, optimize=True):
        self.device, self.layout = device, layout
        self.optimized = optimize
        self.output_start, self.count = output_span
        self.host = device.allocate(1024)
        self.output = device.allocate(self.count)

        def address(index):
            value = layout.memory[index]
            return (
                device.address(value) if isinstance(value, DeviceWord) else self.host.start + index
            )

        self.shared = [SENTINEL] * 1024
        table_base = 0
        if operation == "nvfp4_matmul":
            source, table_base, descriptor_start, descriptors = native_program(
                layout, params, address
            )
            self.shared[descriptor_start : descriptor_start + len(descriptors)] = descriptors
        else:
            source = source.replace("LD_SHARED", "LD_GLOBAL").replace("ST_SHARED", "ST_GLOBAL")
        self.table_word = table_base // 4
        if layout.cursor + self.table_word > OUTPUT_DELTA // 4:
            raise ValueError("direct pointer table exceeds shared RAM")
        self.addresses = [address(i) for i in range(layout.cursor)]
        self.shared[self.table_word : self.table_word + layout.cursor] = [
            p * 4 for p in self.addresses
        ]
        self.shared[-1] = ((self.output.start - self.output_start) * 4) & 0xFFFFFFFF
        # Keep the complete pointer table for indirect accesses. Compiler-owned
        # literals get a separate read-only pool; tensor/weight data stay external.
        pool_start = self.table_word + layout.cursor
        if operation == "nvfp4_matmul":
            pool_start = descriptor_start + len(descriptors)
        candidates = literal_loads(source, frozenset(layout.constants.values())) if optimize else {}
        self.literals = {}
        for logical in sorted(candidates, key=lambda a: (-candidates[a], a)):
            if pool_start >= OUTPUT_DELTA // 4:
                break
            self.literals[logical] = pool_start * 4
            self.shared[pool_start] = layout.memory[logical // 4]
            pool_start += 1
        self.expected_shared = list(self.shared)
        self.written = [
            operation != "vocabulary_mask" or i % params["cols"] >= params["vocabulary"]
            for i in range(self.count)
        ]
        self.result_addresses = []
        for i, written in enumerate(self.written):
            logical = self.output_start + i
            pointer = self.output.start + i if written else self.addresses[logical]
            self.result_addresses.append(pointer)
            self.expected_shared[self.table_word + logical] = pointer * 4
        if operation == "nvfp4_matmul":
            groups = params["k"] // 16
            a = layout.regions["a"][0] + (groups - 1) * 12
            b = layout.regions["b"][0] + (params["n"] // 4 * groups - 1) * 12
            self.expected_shared[:24] = list(
                map(int, layout.memory[a : a + 12] + layout.memory[b : b + 12])
            )
        self.readback = spans(self.result_addresses)
        self.writable = spans(
            self.output.start + i for i, written in enumerate(self.written) if written
        )
        self.source = lower(
            source, table_base, literals=tuple(self.literals.items()), optimize=optimize
        )
        self.uploads = []
        for i, value in enumerate(layout.memory[: layout.cursor]):
            if isinstance(value, DeviceWord):
                continue
            pointer = self.host.start + i
            if device.shadow[pointer] != value:
                if self.uploads and self.uploads[-1][0] + len(self.uploads[-1][1]) == pointer:
                    self.uploads[-1][1].append(int(value))
                else:
                    self.uploads.append((pointer, [int(value)]))

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        write_hex(directory / "input.hex", self.shared)
        write_hex(directory / "logical-input.hex", list(map(int, self.layout.memory)))
        (directory / "resident-layout.json").write_text(
            json.dumps(
                dict(
                    input_word_addresses=self.addresses,
                    output_word_addresses=self.result_addresses,
                    readback_ranges=self.readback,
                    writable_ranges=self.writable,
                    uploads=self.uploads,
                    table_byte_address=self.table_word * 4,
                    literal_byte_addresses=self.literals,
                    optimized=self.optimized,
                ),
                indent=2,
            )
            + "\n"
        )

    def run(self, program, directory, *, active_masks, warp_mask, latency):
        device = self.device
        # Include queued immutable weights in the same explicit upload request.
        self.uploads = device.pending_uploads + self.uploads
        device.pending_uploads = []
        try:
            shared, registers, values, metrics = device.execute(
                program,
                self.shared,
                self.uploads,
                self.readback,
                self.writable,
                active_masks=active_masks,
                warp_mask=warp_mask,
                latency=latency,
                max_cycles=1000000,
            )
            if shared != self.expected_shared:
                raise ValueError("direct kernel changed a pointer, descriptor or shared guard")
        except BaseException:
            self.save(directory)
            write_hex(directory / "resident-shadow.hex", device.shadow)
            device.session.close()
            raise
        # An immutable host shadow lets existing independent numerical checks
        # inspect outputs. These values came from the physical readback above.
        after = list(map(int, self.layout.memory))
        after[self.output_start : self.output_start + self.count] = values
        output = []
        for i, value in enumerate(values):
            if self.written[i]:
                allocation, offset = self.output, i
            else:
                old = self.layout.memory[self.output_start + i]
                allocation, offset = (
                    (old.allocation, old.offset)
                    if isinstance(old, DeviceWord)
                    else (self.host, self.output_start + i)
                )
            output.append(DeviceWord(value, allocation, offset))
        total = dict(metrics, rtl_launches=1)
        transfer = dict(
            metrics,
            transport="resident_external_memory",
            addressing="shared_pointer_table",
            optimized=self.optimized,
            literal_words=len(self.literals),
            latency=latency,
            hardware_tested=False,
            guard_words_checked=CAPACITY - sum(n for _, n in self.writable),
            total_cycles=metrics["cycles"],
            copy_cycles=0,
            copy_launches=0,
            device_copy_words=0,
            total=total,
            capacity_bytes=CAPACITY * 4,
            peak_allocation_bytes=device.peak_words * 4,
        )
        return shared, registers, after, transfer, output
