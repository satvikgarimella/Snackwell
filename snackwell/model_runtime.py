"""Simulation launch orchestration for models larger than shared memory.

Host code copies raw words and reshapes/slices tensors. Every model arithmetic
operation, token lookup and greedy selection runs on the SM. This is not DMA
or a hardware global-memory implementation.
"""

import json
from functools import lru_cache
from pathlib import Path

from snackwell.assembler import assemble
from snackwell.decoding import validate_temperature
from snackwell.kernels.inference import (
    Layout,
    Stage,
    elementwise,
    layernorm,
    scale_mask,
    tiled_matmul,
)
from snackwell.kernels.matvec import simd_matvec
from snackwell.kernels.model import argmax, categorical, embedding, loop_softmax, vocabulary_mask
from snackwell.reference.inference import expected_stage, verify_stage
from snackwell.reference.native import tensor_product
from snackwell.simulator import ModelSimulator, SimulatorSession, compile_runner, run_program


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

    def __init__(self, output, executable=None, on_progress=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.executable = executable or compile_runner(self.output)
        self.reports = []
        self.on_progress = on_progress
        self.simulation = dict(
            backend=self.executable.backend
            if isinstance(self.executable, (ModelSimulator, SimulatorSession))
            else "icarus",
            trace=self.executable.trace
            if isinstance(self.executable, (ModelSimulator, SimulatorSession))
            else True,
        )

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
        stage = Stage(name, source, operation, tuple(inputs), output, params)
        directory = self.output / f"{len(self.reports):03d}_{name}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").unlink(missing_ok=True)
        words = list(assembled_program(source))
        (directory / "program.asm").write_text(source)
        (directory / "layout.json").write_text(json.dumps(layout.regions, indent=2) + "\n")
        expected = expected_stage(stage, layout.memory, layout, matrix_product=tensor_product)
        after, _, cycles = run_program(
            self.executable,
            words,
            layout.memory,
            directory,
            warp_mask=warp_mask,
            active_masks=active_masks,
            max_cycles=200000,
        )
        accuracy = verify_stage(stage, layout.memory, after, layout, prepared=expected)
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
        self.reports.append(report)
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if self.on_progress:
            self.on_progress(dict(type="kernel_complete", launch=len(self.reports), **report))
        start, count = layout.regions[output]
        return after[start : start + count]

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
    ):
        classifier = kind in ("softmax", "vocabulary_mask", "argmax", "categorical")
        max_cols = 512 if classifier or kind == "gelu" else 128
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
        row_chunk = 1 if cols > 128 else 4
        for row in range(0, rows, row_chunk):
            layout = Layout()
            count = min(row_chunk, rows - row)
            block = x[row * cols : (row + count) * cols]
            layout.allocate("x", len(block), block)
            ins, output, params = ["x"], "x", dict(rows=count, cols=cols)
            if kind in ("bias_add", "residual_add"):
                b = other if kind == "bias_add" else other[row * cols : (row + 4) * cols]
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
                if len(gamma) != cols or len(beta) != cols:
                    raise ValueError("normalization parameter length mismatch")
                layout.allocate("gamma", cols, gamma)
                layout.allocate("beta", cols, beta)
                ins += ["gamma", "beta"]
                params["epsilon"] = epsilon
                source = layernorm(
                    layout, count, cols, epsilon, "x", "x", parallel_affine=count == 1
                )
            elif kind == "softmax":
                params["temperature"] = validate_temperature(temperature)
                source = loop_softmax(layout, count, cols, temperature)
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
                    if count == 1 and kind not in ("gelu", "layernorm", "bias_add", "residual_add")
                    else 0xFFFF,
                    **params,
                )
            )
        return out

    def lookup(self, ids, table, position, width, vocabulary):
        rows = len(ids)
        if (rows != 1 and (rows < 4 or rows > 128 or rows % 4)) or width not in (
            4,
            8,
            16,
            32,
            64,
            128,
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
