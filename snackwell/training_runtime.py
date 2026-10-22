"""Bounded RTL launches for feed-forward loss, backward propagation and SGD."""

from snackwell.kernels import training as kernels
from snackwell.kernels.inference import Layout
from snackwell.model_runtime import ModelRuntime, rectangle


class TrainingRuntime(ModelRuntime):
    def vector(self, name, kind, x, b, scale=1.0):
        if not x or len(x) != len(b) or len(x) % 4:
            raise ValueError("training vectors need matching nonempty, multiple-of-four lengths")
        out = []
        for start in range(0, len(x), 256):
            left, right = x[start : start + 256], b[start : start + 256]
            layout = Layout()
            layout.allocate("x", len(left), left)
            layout.allocate("b", len(right), right)
            layout.allocate("out", len(left))
            out.extend(
                self.launch(
                    name,
                    layout,
                    kernels.vector(layout, kind, len(left), scale),
                    kind,
                    ("x", "b"),
                    "out",
                    scale=scale,
                )
            )
        return out

    def loss(self, y, target):
        if not y or len(y) != len(target):
            raise ValueError("loss requires matching nonempty prediction and target tensors")
        total = [0]
        for start in range(0, len(y), 256):
            left, right = y[start : start + 256], target[start : start + 256]
            scale = 1 / (2 * len(y)) if start + len(left) == len(y) else 1
            layout = Layout()
            layout.allocate("x", len(left), left)
            layout.allocate("b", len(right), right)
            layout.allocate("initial", 1, total)
            layout.allocate("out", 1)
            total = self.launch(
                "loss",
                layout,
                kernels.squared_loss(layout, len(left), scale),
                "squared_loss",
                ("x", "b", "initial"),
                "out",
                active_masks=1,
                scale=scale,
            )
        return total

    def column_sum(self, name, x, rows, cols):
        if rows < 1 or cols < 4 or cols % 4 or len(x) != rows * cols:
            raise ValueError("invalid column reduction shape")
        out = []
        for col in range(0, cols, 16):
            columns = min(16, cols - col)
            total = [0] * columns
            for row in range(0, rows, 32):
                count = min(32, rows - row)
                block = rectangle(x, cols, row, count, col, columns)
                layout = Layout()
                layout.allocate("x", len(block), block)
                layout.allocate("initial", columns, total)
                layout.allocate("out", columns)
                total = self.launch(
                    name,
                    layout,
                    kernels.column_sum(layout, count, columns),
                    "column_sum",
                    ("x", "initial"),
                    "out",
                    rows=count,
                    cols=columns,
                )
            out.extend(total)
        return out


def transpose(words, rows, cols):
    """Copy raw words into a transposed layout; perform no tensor arithmetic."""
    if len(words) != rows * cols:
        raise ValueError("transpose shape mismatch")
    return [words[r * cols + c] for c in range(cols) for r in range(rows)]
