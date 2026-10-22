"""Bounded SM launches for normalization, attention and classifier backward passes."""

from snackwell.kernels import backward
from snackwell.kernels.inference import Layout
from snackwell.numeric import bits
from snackwell.training_runtime import TrainingRuntime


class TransformerTrainingRuntime(TrainingRuntime):
    @staticmethod
    def row_shape(rows, cols, maximum=128):
        if (
            type(rows) is not int
            or rows < 1
            or type(cols) is not int
            or cols < 4
            or cols > maximum
            or cols & (cols - 1)
        ):
            raise ValueError(
                f"row kernels need positive rows and power-of-two columns in 4..{maximum}"
            )

    def arithmetic(self, name, kind, x, y=None, scale=1.0):
        if not x or (y is not None and len(x) != len(y)):
            raise ValueError("arithmetic requires matching nonempty vectors")
        pad = (-len(x)) % 4
        right = [0] * len(x) if y is None else y
        return self.vector(name, kind, x + [0] * pad, right + [0] * pad, scale)[: len(x)]

    def norm_backward(self, name, x, dy, gamma, rows, cols, epsilon):
        self.row_shape(rows, cols)
        if len(x) != rows * cols or len(dy) != len(x) or len(gamma) != cols:
            raise ValueError("LayerNorm backward shape mismatch")
        dx, dg = [], []
        chunk = 4 if cols <= 32 else 1
        for row in range(0, rows, chunk):
            count = min(chunk, rows - row)
            layout = Layout()
            layout.allocate("x", count * cols, x[row * cols : (row + count) * cols])
            layout.allocate("dy", count * cols, dy[row * cols : (row + count) * cols])
            layout.allocate("gamma", cols, gamma)
            layout.allocate("out", 2 * count * cols)
            out = self.launch(
                name,
                layout,
                backward.layernorm_backward(layout, count, cols, epsilon),
                "layernorm_backward",
                ("x", "dy", "gamma"),
                "out",
                rows=count,
                cols=cols,
                epsilon=epsilon,
                active_masks=(1 << count) - 1,
            )
            dx.extend(out[: count * cols])
            dg.extend(out[count * cols :])
        return (
            dx,
            self.column_sum(name + "_gamma", dg, rows, cols),
            self.column_sum(name + "_beta", dy, rows, cols),
        )

    def softmax_backward(self, name, p, dy, rows, cols, scale=1.0):
        self.row_shape(rows, cols)
        if len(p) != rows * cols or len(dy) != len(p):
            raise ValueError("softmax backward shape mismatch")
        out = []
        chunk = 4 if cols <= 64 else 1
        for row in range(0, rows, chunk):
            count = min(chunk, rows - row)
            layout = Layout()
            layout.allocate("p", count * cols, p[row * cols : (row + count) * cols])
            layout.allocate("dy", count * cols, dy[row * cols : (row + count) * cols])
            layout.allocate("out", count * cols)
            out.extend(
                self.launch(
                    name,
                    layout,
                    backward.softmax_backward(layout, cols, scale),
                    "softmax_backward",
                    ("p", "dy"),
                    "out",
                    rows=count,
                    cols=cols,
                    scale=scale,
                    active_masks=(1 << count) - 1,
                )
            )
        return out

    def cross_entropy(self, logits, targets, rows, cols):
        self.row_shape(rows, cols, maximum=512)
        if len(logits) != rows * cols or len(targets) != rows:
            raise ValueError("cross entropy shape mismatch")
        if any(type(t) is not int or not 0 <= t < cols for t in targets):
            raise ValueError("target outside classifier columns")
        losses = []
        chunk = 1 if cols > 128 else 4
        for row in range(0, rows, chunk):
            count = min(chunk, rows - row)
            layout = Layout()
            layout.allocate("x", count * cols, logits[row * cols : (row + count) * cols])
            layout.allocate("targets", count, targets[row : row + count])
            layout.allocate("out", count)
            losses.extend(
                self.launch(
                    "cross_entropy",
                    layout,
                    backward.cross_entropy(layout, cols),
                    "cross_entropy",
                    ("x", "targets"),
                    "out",
                    rows=count,
                    cols=cols,
                    active_masks=(1 << count) - 1,
                )
            )
        # Four columns fit the existing reduction; only the first is populated.
        padded = [w for loss in losses for w in (loss, 0, 0, 0)]
        total = self.column_sum("loss_sum", padded, rows, 4)
        return self.arithmetic("loss_mean", "scale", total, scale=1 / rows)[:1]

    def classifier_backward(self, probabilities, targets, cols):
        one_hot = [bits(int(col == target)) for target in targets for col in range(cols)]
        return self.arithmetic(
            "classifier_gradient", "mse_gradient", probabilities, one_hot, scale=1 / len(targets)
        )

    def embedding_backward(self, ids, derivative, width, vocabulary):
        """Group raw rows by ID on the host, then accumulate repeated IDs on the SM."""
        out = [0] * (vocabulary * width)
        for token in sorted(set(ids)):
            rows = [
                derivative[r * width : (r + 1) * width] for r, i in enumerate(ids) if i == token
            ]
            values = self.column_sum(
                "embedding_gradient", [w for row in rows for w in row], len(rows), width
            )
            out[token * width : (token + 1) * width] = values
        return out
