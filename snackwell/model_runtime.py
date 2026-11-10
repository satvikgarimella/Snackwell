"""Simulation launch orchestration for models larger than shared memory.

Host code copies/reshapes raw words and optionally quantizes linear inputs.
Model arithmetic, token lookup and selection run on the SM. Shared mode loads
internal memory directly. Global mode stages inputs externally and executes
ordinary loads/stores plus asynchronous shared copies for native NVFP4 tiles.
"""

import json
from functools import lru_cache
from pathlib import Path

from snackwell.assembler import assemble, write_image
from snackwell.decoding import validate_min_p, validate_temperature
from snackwell.kernels.inference import (
    Layout,
    Stage,
    elementwise,
    layernorm,
    loop_layernorm,
    scale_mask,
    tiled_matmul,
)
from snackwell.kernels.matvec import simd_matvec
from snackwell.kernels.model import (
    argmax,
    categorical,
    embedding,
    loop_softmax,
    min_p_filter,
    token_mask,
    vocabulary_mask,
)
from snackwell.reference.inference import expected_stage, verify_stage
from snackwell.reference.native import tensor_product
from snackwell.simulator import (
    ModelSimulator,
    SimulatorSession,
    compile_runner,
    run_program,
    write_hex,
)


@lru_cache(maxsize=128)
def assembled_program(source):
    """Repeated tensor shapes share code; input words are never cached."""
    return tuple(assemble(source))


def rectangle(data, stride, row, rows, col, columns):
    """Copy a rectangular slice of a flat row-major tensor without converting its words."""
    return [data[(row + r) * stride + col + c] for r in range(rows) for c in range(columns)]


def insert_rectangle(target, stride, row, rows, col, columns, data):
    for r in range(rows):
        target[(row + r) * stride + col : (row + r) * stride + col + columns] = data[
            r * columns : (r + 1) * columns
        ]


class ModelRuntime:
    """Split tensors into launches that fit shared memory; verify every launch."""

    def __init__(
        self,
        output,
        executable=None,
        on_progress=None,
        *,
        memory="shared",
        memory_latency=2,
        optimize_resident=True,
    ):
        if memory not in ("shared", "global", "resident", "resident-copy"):
            raise ValueError("memory must be shared, global, resident or resident-copy")
        if type(memory_latency) is not int or not 0 <= memory_latency <= 1000:
            raise ValueError("memory latency must be an integer from 0 to 1000")
        if type(optimize_resident) is not bool:
            raise ValueError("optimize_resident must be a boolean")
        self.memory, self.memory_latency = memory, memory_latency
        self.optimize_resident = optimize_resident
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.executable = executable or compile_runner(self.output)
        self.reports = []
        self.on_progress = on_progress
        self.compact = (
            isinstance(self.executable, SimulatorSession)
            and self.executable.compact
            and not self.executable.trace
        )
        self.simulation = dict(
            backend=self.executable.backend
            if isinstance(self.executable, (ModelSimulator, SimulatorSession))
            else "icarus",
            trace=self.executable.trace
            if isinstance(self.executable, (ModelSimulator, SimulatorSession))
            else True,
        )
        self.device = None
        if memory.startswith("resident"):
            from snackwell.resident_memory import resident_memory

            if not isinstance(self.executable, SimulatorSession):
                raise ValueError("resident memory requires an explicit Verilator session")
            self.device = resident_memory(self.executable)
        if memory != "shared":
            self.simulation.update(memory=memory, memory_latency=memory_latency)
        if memory == "resident":
            self.simulation["resident_optimized"] = optimize_resident
        if self.compact:
            self.simulation["artifacts"] = "summary"

    def launch(
        self,
        name,
        layout,
        source,
        operation,
        inputs,
        output,
        *,
        warp_mask=1,
        active_masks=0xFFFF,
        **params,
    ):
        if self.on_progress:
            self.on_progress(dict(type="kernel_start", stage=name, launch=len(self.reports) + 1))
        external = wanted_shared = direct = None
        if self.memory == "resident":
            from snackwell.direct_memory import DirectLaunch

            direct = DirectLaunch(
                self.device,
                layout,
                source,
                operation,
                params,
                layout.regions[output],
                optimize=self.optimize_resident,
            )
            source, wanted_shared = direct.source, direct.expected_shared
        elif self.memory != "shared":
            from snackwell.kernels.global_model import prepare

            source, external, wanted_shared = prepare(layout, source, operation, params)
        stage = Stage(name, source, operation, tuple(inputs), output, params)
        directory = self.output / f"{len(self.reports):03d}_{name}"
        words = list(assembled_program(source))

        def save_inputs():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "program.asm").write_text(source)
            (directory / "layout.json").write_text(json.dumps(layout.regions, indent=2) + "\n")
            if direct is not None:
                direct.save(directory)
            if external is not None:
                from snackwell.numeric import SENTINEL

                write_hex(directory / "global-input.hex", external)
                write_hex(directory / "input.hex", [SENTINEL] * 1024)

        if not self.compact:
            save_inputs()
        if directory.exists():
            (directory / "result.json").unlink(missing_ok=True)
            (directory / "failure.json").unlink(missing_ok=True)
            if self.memory != "shared":
                for filename in (
                    "output.hex",
                    "global-output.hex",
                    "logical-output.hex",
                    "shared-output.hex",
                    "registers.hex",
                    "trace.csv",
                    "metrics.txt",
                    "profile.json",
                ):
                    (directory / filename).unlink(missing_ok=True)
        after = global_after = shared_after = None
        transfer = resident_output = None
        try:
            expected = expected_stage(
                stage, list(map(int, layout.memory)), layout, matrix_product=tensor_product
            )
            if self.memory != "shared":
                from snackwell.global_memory import run_model_global

                if direct is not None:
                    shared_after, registers, global_after, transfer, resident_output = direct.run(
                        words,
                        directory,
                        warp_mask=warp_mask,
                        active_masks=active_masks,
                        latency=self.memory_latency,
                    )
                elif self.device is not None:
                    shared_after, registers, global_after, transfer, resident_output = (
                        self.device.run(
                            words,
                            external,
                            layout.regions[output],
                            directory,
                            warp_mask=warp_mask,
                            active_masks=active_masks,
                            latency=self.memory_latency,
                        )
                    )
                else:
                    shared_after, registers, global_after, transfer = run_model_global(
                        self.executable,
                        words,
                        external,
                        directory,
                        warp_mask=warp_mask,
                        active_masks=active_masks,
                        latency=self.memory_latency,
                    )
                cycles = transfer.get("total_cycles", transfer["cycles"])
                after = global_after[:1024]
                if transfer["status"] or transfer["copy_status"]:
                    raise ValueError("global-memory or asynchronous-copy status reported an error")
                if shared_after != wanted_shared or (
                    self.device is None and global_after[1024:] != external[1024:]
                ):
                    raise ValueError("global transport changed a shared or external memory guard")
                if operation == "nvfp4_matmul":
                    jobs = params["n"] // 4 * (params["k"] // 16)
                    if transfer["copy_reads"] != jobs * 24 or transfer["copy_writes"] != jobs * 24:
                        raise ValueError("incorrect NVFP4 operand transfer count")
                    if "nvfp4_macs" in transfer and (
                        transfer["nvfp4_macs"] != jobs * 256
                        or transfer["fp32_tensor_multiplies"] != 0
                    ):
                        raise ValueError("incorrect native NVFP4 arithmetic path")
                elif transfer["copy_reads"] or transfer["copy_writes"]:
                    raise ValueError("unexpected async copies in a global scalar/register kernel")
                if not self.compact:
                    if direct is not None:
                        direct.save(directory)
                    write_image(directory / "program.hex", words)
                    write_hex(directory / "output.hex", after)
                    write_hex(
                        directory / ("logical-output.hex" if direct else "global-output.hex"),
                        global_after,
                    )
                    write_hex(directory / "shared-output.hex", shared_after)
                    write_hex(directory / "registers.hex", registers)
            else:
                after, registers, cycles = run_program(
                    self.executable,
                    words,
                    layout.memory,
                    directory,
                    warp_mask=warp_mask,
                    active_masks=active_masks,
                    max_cycles=200000,
                )
            if operation == "nvfp4_matmul" and any(registers[lane * 16 + 14] for lane in range(4)):
                raise ValueError("NVFP4 tensor status reported an execution error")
            accuracy = verify_stage(stage, layout.memory, after, layout, prepared=expected)
        except BaseException as error:
            if self.device is not None:
                self.executable.close()
            if self.compact:
                save_inputs()
                write_image(directory / "program.hex", words)
                if self.memory == "shared":
                    write_hex(directory / "input.hex", layout.memory)
                if after is not None:
                    write_hex(directory / "output.hex", after)
                    write_hex(directory / "registers.hex", registers)
                if global_after is not None:
                    write_hex(
                        directory / ("logical-output.hex" if direct else "global-output.hex"),
                        global_after,
                    )
                    write_hex(directory / "shared-output.hex", shared_after)
                (directory / "failure.json").write_text(
                    json.dumps(
                        dict(
                            verified=False,
                            error=str(error),
                            memory=self.memory,
                            memory_latency=self.memory_latency,
                            warp_mask=warp_mask,
                            active_masks=active_masks,
                            max_cycles=1000000 if self.device is not None else 200000,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            raise
        report = dict(
            stage=name,
            operation=operation,
            cycles=cycles,
            verified=True,
            instructions=len(words),
            allocated_words=layout.cursor,
            warp_mask=warp_mask,
            active_masks=active_masks,
            accuracy=accuracy,
        )
        if transfer is not None:
            report["memory"] = transfer
        self.reports.append(report)
        if not self.compact:
            (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if self.on_progress:
            self.on_progress(dict(type="kernel_complete", launch=len(self.reports), **report))
        start, count = layout.regions[output]
        return resident_output if resident_output is not None else after[start : start + count]

    def matmul(self, name, a, b, m, n, k, transpose_b=False, bias=None):
        """Compute an m-by-n output in chunks, reducing across k inputs per element."""
        if (
            type(m) is not int
            or (m != 1 and (m < 4 or m % 4))
            or any(type(x) is not int or x < 4 or x % 4 for x in (n, k))
        ):
            raise ValueError(
                "model matmul requires one row or multiples of 4; n and k multiples of 4"
            )
        if len(a) != m * k or len(b) != k * n:
            raise ValueError("matmul tensor length mismatch")
        if bias is not None and len(bias) != n:
            raise ValueError("matmul bias length mismatch")
        out = [None] * (m * n)
        # Carry C between K chunks as raw words. Both kernels accumulate
        # four-term FP32 sums in the same order as an unsplit tensor product.
        for row in range(0, m, 8):
            row_count = min(8, m - row)
            column_chunk = 28 if m == 1 else 16
            for col in range(0, n, column_chunk):
                columns = min(column_chunk, n - col)
                result, start = None, 0
                while start < k:
                    inner = min(32, 1 << ((k - start).bit_length() - 1))
                    left = rectangle(a, k, row, row_count, start, inner)
                    right = (
                        rectangle(b, k, col, columns, start, inner)
                        if transpose_b
                        else rectangle(b, n, start, inner, col, columns)
                    )
                    layout = Layout()
                    layout.allocate("a", len(left), left)
                    layout.allocate("b", len(right), right)
                    layout.allocate("out", row_count * columns, result)
                    accumulate = start != 0
                    add_bias = bias is not None and start + inner == k
                    inputs = ("a", "b", "out") if accumulate else ("a", "b")
                    if add_bias:
                        layout.allocate("bias", columns, bias[col : col + columns])
                        inputs += ("bias",)
                    warps = min(4, ((row_count + 3) // 4) * (columns // 4))
                    options = dict(
                        transpose_b=transpose_b,
                        accumulate=accumulate,
                        warps=warps,
                        bias="bias" if add_bias else None,
                    )
                    source = (
                        simd_matvec(layout, columns, inner, **options)
                        if row_count == 1
                        else tiled_matmul(layout, row_count, columns, inner, **options)
                    )
                    result = self.launch(
                        name,
                        layout,
                        source,
                        "matmul",
                        inputs,
                        "out",
                        warp_mask=(1 << warps) - 1,
                        m=row_count,
                        n=columns,
                        k=inner,
                        transpose_b=transpose_b,
                        accumulate=accumulate,
                        bias=add_bias,
                    )
                    start += inner
                insert_rectangle(out, n, row, row_count, col, columns, result)
        return out

    def nvfp4_linear(self, name, a, b, m, n, k, bias=None):
        """Quantize on the host, execute native packed products and the epilogue on RTL."""
        from snackwell.kernels.nvfp4_model import program
        from snackwell.nvfp4_inference import activation_rows, tile, weight_columns
        from snackwell.reference.nvfp4 import QuantizedVectors

        if any(type(x) is not int or x < 1 for x in (m, n, k)) or n % 4:
            raise ValueError("NVFP4 linear requires positive dimensions and columns divisible by 4")
        if len(a) != m * k or len(b) != k * n or (bias is not None and len(bias) != n):
            raise ValueError("NVFP4 linear tensor length mismatch")
        padded = (k + 15) // 16 * 16
        activations = activation_rows(a, m, k)
        # The quantizer's process-wide LRU must not retain device allocation handles.
        weights = weight_columns(tuple(map(int, b)), n, k)
        weight_scale = [weights.tensor_scale]
        if self.device is not None:
            weight_scale = self.device.constant(("nvfp4_scale", weights.tensor_scale), weight_scale)
        out = [None] * (m * n)
        for row in range(0, m, 4):
            count = min(4, m - row)
            rows = activations[row : row + count]
            scales = [v.tensor_scale for v in rows] + [0x3F800000] * (4 - count)
            left = QuantizedVectors(
                [v.codes[0] for v in rows] + [[0] * padded for _ in range(4 - count)],
                [v.scales[0] for v in rows] + [[0] * (padded // 16) for _ in range(4 - count)],
                0x3F800000,
            )
            for col in range(0, n, 16):
                columns = min(16, n - col)
                result = None
                for start in range(0, padded, 128):
                    inner = min(128, padded - start)
                    aa, bb = (
                        tile(left, 0, 4, start, inner),
                        tile(weights, col, columns, start, inner),
                    )
                    packed_a = [w for g in range(inner // 16) for w in aa.group_words(0, g)]
                    packed_b = [
                        w
                        for c in range(0, columns, 4)
                        for g in range(inner // 16)
                        for w in bb.group_words(c, g)
                    ]
                    if self.device is not None:
                        packed_b = self.device.constant(("packed_nvfp4", tuple(packed_b)), packed_b)
                    layout = Layout()
                    layout.allocate("a", len(packed_a), packed_a)
                    layout.allocate("b", len(packed_b), packed_b)
                    layout.allocate("scale_a", 4, scales)
                    layout.allocate("scale_b", 1, weight_scale)
                    inputs = ["a", "b", "scale_a", "scale_b"]
                    if result is not None:
                        layout.allocate("initial", 4 * columns, result)
                        inputs.append("initial")
                    final = start + inner == padded
                    add_bias = final and bias is not None
                    if add_bias:
                        layout.allocate("bias", columns, bias[col : col + columns])
                        inputs.append("bias")
                    layout.allocate("out", 4 * columns)
                    options = dict(accumulate=start != 0, final=final, bias=add_bias)
                    result = self.launch(
                        name,
                        layout,
                        program(layout, columns, inner, **options),
                        "nvfp4_matmul",
                        inputs,
                        "out",
                        n=columns,
                        k=inner,
                        **options,
                    )
                insert_rectangle(out, n, row, count, col, columns, result[: count * columns])
        return out

    def rows(
        self,
        name,
        kind,
        x,
        rows,
        cols,
        other=None,
        gamma=None,
        beta=None,
        epsilon=1e-5,
        vocabulary=None,
        draws=None,
        temperature=1.0,
        min_p=0.0,
    ):
        classifier = kind in ("softmax", "vocabulary_mask", "argmax", "categorical", "min_p")
        max_cols = 1024 if kind == "gelu" else 512 if classifier else 256
        if (
            (rows != 1 and (rows < 4 or rows % 4))
            or cols < 4
            or cols > max_cols
            or cols & (cols - 1)
        ):
            raise ValueError(
                "row kernels require one row or multiples of 4 and power-of-two columns "
                f"4..{max_cols}"
            )
        if len(x) != rows * cols:
            raise ValueError("row tensor length mismatch")
        if kind == "gelu" and cols > 128:
            # GELU has no cross-feature reduction: reshape contiguous words
            # into bounded launches, leaving space for its constants.
            chunk = 128 if rows == 1 else 512
            return [
                word
                for start in range(0, len(x), chunk)
                for word in self.rows(name, kind, x[start : start + chunk], chunk // 128, 128)
            ]
        out = []
        row_chunk = (
            1
            if cols > 128
            or (self.memory == "resident" and cols >= 128 and kind in ("bias_add", "residual_add"))
            else 4
        )
        for row in range(0, rows, row_chunk):
            layout = Layout()
            count = min(row_chunk, rows - row)
            block = x[row * cols : (row + count) * cols]
            layout.allocate("x", len(block), block)
            ins, output, params = ["x"], "x", dict(rows=count, cols=cols)
            if kind in ("bias_add", "residual_add"):
                b = other if kind == "bias_add" else other[row * cols : (row + count) * cols]
                if len(b) != (cols if kind == "bias_add" else count * cols):
                    raise ValueError("addition tensor length mismatch")
                layout.allocate("b", len(b), b)
                ins.append("b")
                source = elementwise(
                    layout, kind, count, cols, "x", "x", parallel_columns=count == 1
                )
            elif kind == "gelu":
                source = elementwise(layout, kind, count, cols, "x", "x")
            elif kind == "layernorm":
                if gamma is None or beta is None or len(gamma) != cols or len(beta) != cols:
                    raise ValueError("normalization parameter length mismatch")
                layout.allocate("gamma", cols, gamma)
                layout.allocate("beta", cols, beta)
                ins += ["gamma", "beta"]
                params["epsilon"] = epsilon
                normalize = loop_layernorm if self.memory == "resident" and cols > 16 else layernorm
                source = normalize(
                    layout,
                    count,
                    cols,
                    epsilon,
                    "x",
                    "x",
                    parallel_affine=count == 1 and not (self.memory == "resident" and cols > 128),
                )
            elif kind == "softmax":
                params["temperature"] = validate_temperature(temperature)
                source = loop_softmax(layout, count, cols, temperature)
            elif kind == "min_p":
                params["min_p"] = validate_min_p(min_p)
                source = min_p_filter(layout, count, cols, min_p)
            elif kind == "vocabulary_mask":
                if type(vocabulary) is not int or not 1 <= vocabulary <= cols:
                    raise ValueError("invalid vocabulary size")
                params["vocabulary"] = vocabulary
                source = vocabulary_mask(layout, count, cols, vocabulary)
            elif kind == "argmax":
                layout.allocate("out", count)
                output = "out"
                source = argmax(layout, count, cols)
            elif kind == "categorical":
                if draws is None or len(draws) != rows:
                    raise ValueError("sampling requires one random draw per row")
                layout.allocate("draws", count, draws[row : row + count])
                layout.allocate("out", count)
                ins.append("draws")
                output = "out"
                source = categorical(layout, count, cols)
            else:
                raise ValueError(kind)
            out.extend(
                self.launch(
                    name,
                    layout,
                    source,
                    kind,
                    ins,
                    output,
                    active_masks=1
                    if count == 1
                    and (
                        kind not in ("gelu", "layernorm", "bias_add", "residual_add")
                        or (self.memory == "resident" and kind == "layernorm" and cols > 128)
                    )
                    else 0xFFFF,
                    **params,
                )
            )
        return out

    def restrict_tokens(self, name, logits, allowed):
        cols = len(logits)
        if not 4 <= cols <= 512 or cols & (cols - 1) or len(allowed) != cols:
            raise ValueError("token restriction requires 4..512 power-of-two logits and a mask")
        if any(type(a) is not bool for a in allowed) or not any(allowed):
            raise ValueError("token restriction requires a nonempty boolean allow-list")
        packed = [
            sum(int(flag) << bit for bit, flag in enumerate(allowed[start : start + 32]))
            for start in range(0, cols, 32)
        ]
        layout = Layout()
        layout.allocate("x", cols, logits)
        layout.allocate("allowed", len(packed), packed)
        return self.launch(
            name,
            layout,
            token_mask(layout, cols),
            "token_mask",
            ("x", "allowed"),
            "x",
            cols=cols,
            active_masks=1,
        )

    def lookup(self, ids, table, position, width, vocabulary):
        rows = len(ids)
        if (rows != 1 and (rows < 4 or rows > 128 or rows % 4)) or width not in (
            4,
            8,
            16,
            32,
            64,
            128,
            256,
        ):
            raise ValueError("unsupported embedding shape")
        if any(type(i) is not int or not 0 <= i < vocabulary for i in ids):
            raise ValueError("token ID outside vocabulary")
        if type(vocabulary) is not int or not 1 <= vocabulary <= 512:
            raise ValueError("embedding vocabulary must contain 1..512 entries")
        if len(table) != vocabulary * width or len(position) != rows * width:
            raise ValueError("embedding tensor length mismatch")
        out = [None] * (rows * width)
        # Bound both dimensions: a 128-entry table and 32 context rows use
        # 800 words. Longer contexts reuse that table in another row launch.
        feature_chunk = 8 if rows == 1 and vocabulary <= 120 and width >= 8 else 4
        if vocabulary > 128:
            feature_chunk = 2 if vocabulary <= 256 else 1
        for col in range(0, width, feature_chunk):
            columns = min(feature_chunk, width - col)
            data = rectangle(table, width, 0, vocabulary, col, columns)
            for row in range(0, rows, 32):
                count = min(32, rows - row)
                layout = Layout()
                layout.allocate("ids", count, ids[row : row + count])
                layout.allocate("table", len(data), data)
                layout.allocate(
                    "position",
                    count * columns,
                    rectangle(position, width, row, count, col, columns),
                )
                layout.allocate("out", count * columns)
                result = self.launch(
                    "embedding",
                    layout,
                    embedding(layout, count, columns),
                    "embedding",
                    ("ids", "table", "position"),
                    "out",
                    rows=count,
                    cols=columns,
                    active_masks=(1 << columns) - 1 if count == 1 and columns < 4 else 0xFFFF,
                )
                insert_rectangle(out, width, row, count, col, columns, result)
        return out

    def attention_scale(self, scores, rows, width, columns=None, row_offset=0):
        columns = rows if columns is None else columns
        if (rows != 1 and (rows < 4 or rows % 4)) or len(scores) != rows * columns:
            raise ValueError("invalid attention score shape")
        out = []
        # A 32x32 score matrix fills memory before constants. Chunk query rows
        # and retain their absolute position for the causal mask.
        for row in range(0, rows, 4):
            layout = Layout()
            count = min(4, rows - row)
            layout.allocate(
                "scores", count * columns, scores[row * columns : (row + count) * columns]
            )
            out.extend(
                self.launch(
                    "attention_scale_mask",
                    layout,
                    scale_mask(
                        layout, count, width, True, columns=columns, row_offset=row_offset + row
                    ),
                    "scale_mask",
                    ("scores",),
                    "scores",
                    rows=count,
                    cols=columns,
                    width=width,
                    causal=True,
                    row_offset=row_offset + row,
                    active_masks=1 if count == 1 else 0xFFFF,
                )
            )
        return out
