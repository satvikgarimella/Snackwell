"""Stream pretrained decoder tensors through the existing 4 KiB SM.

Only tensor movement, indexing and shape management occur here. Reductions,
nonlinearities, normalization and token selection execute as verified assembly.
"""

import math

from snackwell.kernels import pretrained as kernels
from snackwell.kernels.inference import Layout
from snackwell.model_runtime import ModelRuntime, rectangle
from snackwell.numeric import bits, value
from snackwell.reference.inference import categorical_reference, stable_softmax, verify_approx


class PretrainedRuntime(ModelRuntime):
    def reduction(self, name, x, kind="sum"):
        if not x or kind not in ("sum", "squares", "max"):
            raise ValueError("invalid reduction")
        state = [0xFF800000 if kind == "max" else 0]
        for start in range(0, len(x), 512):
            part = x[start : start + 512]
            layout = Layout()
            layout.allocate("x", len(part), part)
            layout.allocate("state", 1, state)
            state = self.launch(
                name,
                layout,
                kernels.reduce(layout, len(part), kind),
                "pretrained_" + kind,
                ("x", "state"),
                "state",
                active_masks=1,
            )
        return state[0]

    def inverse(self, name, word, kind="reciprocal", *, width=1, epsilon=0.0):
        if not math.isfinite(value(word)) or value(word) < 0:
            raise ValueError("inverse requires a finite nonnegative reduction")
        if kind == "reciprocal" and value(word) == 0:
            raise ValueError("reciprocal requires a positive value")
        if kind == "rms_inverse" and (width < 1 or epsilon <= 0):
            raise ValueError("RMS normalization requires positive width and epsilon")
        layout = Layout()
        layout.allocate("state", 1, [word])
        return self.launch(
            name,
            layout,
            kernels.inverse(layout, kind, width, epsilon),
            "pretrained_" + kind,
            ("state",),
            "state",
            active_masks=1,
            width=width,
            epsilon=epsilon,
        )[0]

    def vector(
        self,
        name,
        x,
        kind,
        *,
        other=None,
        scalar=None,
        cos=None,
        sin=None,
        maximum=None,
        temperature_inverse=None,
    ):
        vectors = {k: v for k, v in dict(other=other, cos=cos, sin=sin).items() if v is not None}
        scalars = {
            k: v
            for k, v in dict(
                scalar=scalar, maximum=maximum, temperature_inverse=temperature_inverse
            ).items()
            if v is not None
        }
        if not x or any(len(v) != len(x) for v in vectors.values()):
            raise ValueError("vector tensor length mismatch")
        inputs = {
            "add": ("x", "other"),
            "multiply": ("x", "other"),
            "scale": ("x", "scalar"),
            "norm": ("x", "scalar", "other"),
            "rope": ("x", "other", "cos", "sin"),
            "exp": ("x", "maximum", "temperature_inverse"),
            "swiglu": ("x", "other"),
        }[kind]
        if set(inputs) != {"x", *vectors, *scalars}:
            raise ValueError("missing or extra vector operands")
        out = []
        for start in range(0, len(x), 128):
            part = x[start : start + 128]
            count = (len(part) + 3) // 4 * 4
            layout = Layout()
            padding = [0xFF800000 if kind == "exp" else 0] * (count - len(part))
            layout.allocate("x", count, list(part) + padding)
            for key, data in vectors.items():
                layout.allocate(
                    key, count, list(data[start : start + len(part)]) + [0] * len(padding)
                )
            for key, word in scalars.items():
                layout.allocate(key, 1, [word])
            result = self.launch(
                name, layout, kernels.vector(layout, count, kind), "pretrained_" + kind, inputs, "x"
            )
            out.extend(result[: len(part)])
        return out

    def rmsnorm(self, name, x, gamma, epsilon):
        total = self.reduction(name + "_squares", x, "squares")
        inv = self.inverse(name + "_inverse", total, "rms_inverse", width=len(x), epsilon=epsilon)
        return self.vector(name, x, "norm", other=gamma, scalar=inv)

    def softmax(self, name, logits, temperature=1.0):
        if not math.isfinite(temperature) or not 0.1 <= temperature <= 2:
            raise ValueError("temperature must be 0.1..2")
        maximum = self.reduction(name + "_max", logits, "max")
        if not math.isfinite(value(maximum)):
            raise ValueError("softmax requires at least one finite logit")
        weights = self.vector(
            name + "_exp", logits, "exp", maximum=maximum, temperature_inverse=bits(1 / temperature)
        )
        # Bounded partial sums prevent long-vocabulary summation error from
        # growing linearly with the full vocabulary length.
        partials = [
            self.reduction(name + "_sum", weights[i : i + 512]) for i in range(0, len(weights), 512)
        ]
        total = self.reduction(name + "_total", partials)
        inv = self.inverse(name + "_inverse", total)
        result = self.vector(name + "_normalize", weights, "scale", scalar=inv)
        verify_approx(result, stable_softmax([value(z) / temperature for z in logits]), "softmax")
        if abs(math.fsum(map(value, result)) - 1) > 2e-5:
            raise ValueError("streamed softmax probabilities do not sum to one")
        return result

    def argmax(self, name, values):
        if not values or not any(math.isfinite(value(v)) for v in values):
            raise ValueError("argmax needs finite candidates")
        state = [0xFF800000, 0]
        for start in range(0, len(values), 512):
            part = values[start : start + 512]
            layout = Layout()
            layout.allocate("x", len(part), part)
            layout.allocate("state", 2, state)
            state = self.launch(
                name,
                layout,
                kernels.argmax(layout, len(part), start),
                "pretrained_argmax",
                ("x", "state"),
                "state",
                active_masks=1,
                offset=start,
            )
        return state[1]

    def sample(self, name, weights, draw):
        if not math.isfinite(draw) or not 0 <= value(bits(draw)) < 1:
            raise ValueError("sampling draw must round into [0,1)")
        total = self.reduction(name + "_sum", weights)
        if not math.isfinite(value(total)) or value(total) <= 0:
            raise ValueError("sampling requires positive finite total weight")
        threshold = self.vector(name + "_threshold", [total], "scale", scalar=bits(draw))[0]
        state = [0, 0xFFFFFFFF, 0xFFFFFFFF]
        for start in range(0, len(weights), 512):
            part = weights[start : start + 512]
            layout = Layout()
            layout.allocate("x", len(part), part)
            layout.allocate("state", 3, state)
            layout.allocate("threshold", 1, [threshold])
            state = self.launch(
                name,
                layout,
                kernels.categorical(layout, len(part), start),
                "pretrained_categorical",
                ("x", "state", "threshold"),
                "state",
                active_masks=1,
                offset=start,
            )
        selected = state[1] if state[1] != 0xFFFFFFFF else state[2]
        if selected != categorical_reference(weights, bits(draw)):
            raise ValueError("streamed categorical result differs from independent reference")
        return selected

    def embedding(self, token, table, width, vocabulary):
        if type(token) is not int or not 0 <= token < vocabulary or width < 4 or width % 4:
            raise ValueError("invalid embedding index or width")
        if len(table) != width * vocabulary:
            raise ValueError("embedding table shape mismatch")
        # The host supplies a page of raw weights. The SM resolves the token's
        # row within that page and copies its words, preserving signed zeros.
        page = token // 16 * 16
        rows = min(16, vocabulary - page)
        out, col = [], 0
        while col < width:
            columns = min(32, 1 << ((width - col).bit_length() - 1))
            layout = Layout()
            layout.allocate("ids", 1, [token - page])
            data = [int(w) for w in rectangle(table, width, page, rows, col, columns)]
            layout.allocate("table", len(data), data)
            layout.allocate("out", columns)
            out.extend(
                self.launch(
                    "embedding",
                    layout,
                    kernels.lookup(layout, columns),
                    "pretrained_lookup",
                    ("ids", "table"),
                    "out",
                    columns=columns,
                )
            )
            col += columns
        return out
