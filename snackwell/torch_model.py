"""Host reference model for training and diagnostics; never used by RTL inference.

PyTorch is optional. Inference uses the separate scalar mathematical reference.
"""

import math

import torch
import torch.nn.functional as F

from snackwell.character_model import block_prefix, parameter_shapes


class ReferenceTransformer(torch.nn.Module):
    def __init__(self, config, weights=None, *, dropout=0.0, initialization="fan_in"):
        super().__init__()
        if not 0 <= dropout < 1 or initialization not in ("fan_in", "gpt"):
            raise ValueError("invalid training dropout or initialization")
        self.config = config
        self.dropout = torch.nn.Dropout(dropout)
        n, d, v = config["context"], config["width"], len(config["vocabulary"])
        self.heads, self.layers = config.get("heads", 1), config.get("layers", 1)
        self.parameters_by_name = torch.nn.ParameterDict()
        for name, shape in parameter_shapes(
            n, d, v, self.layers, config.get("ffn_multiplier", 2)
        ).items():
            if weights is not None:
                initial = torch.tensor(weights[name]).reshape(shape)
            elif "gamma" in name:
                initial = torch.ones(shape)
            elif len(shape) == 1:
                initial = torch.zeros(shape)
            else:
                if initialization == "gpt":
                    scale = 0.02
                    if name.endswith(("wo", "w2")):
                        scale /= math.sqrt(2 * self.layers)
                else:
                    scale = 0.1 if "embedding" in name else 1 / math.sqrt(shape[0])
                initial = torch.randn(shape) * scale
            self.parameters_by_name[name] = torch.nn.Parameter(initial)
        self.register_buffer("mask", torch.ones(n, n, dtype=torch.bool).triu(1))

    def forward(self, ids):
        p, d = self.parameters_by_name, self.config["width"]
        batch, n = ids.shape
        if not 1 <= n <= self.config["context"]:
            raise ValueError("input length is outside the model context")
        head_width = d // self.heads

        def norm(x, suffix, prefix=""):
            return F.layer_norm(
                x,
                (d,),
                p[prefix + "gamma" + suffix],
                p[prefix + "beta" + suffix],
                self.config["epsilon"],
            )

        x = self.dropout(F.embedding(ids, p["token_embedding"]) + p["position_embedding"][:n])
        for layer in range(self.layers):
            prefix = block_prefix(layer, self.layers)
            z = norm(x, "1", prefix)
            q, k, v = [
                (z @ p[prefix + "w" + s] + p[prefix + "b" + s])
                .reshape(batch, n, self.heads, head_width)
                .transpose(1, 2)
                for s in ("q", "k", "v")
            ]
            scores = (q @ k.transpose(-1, -2) / math.sqrt(head_width)).masked_fill(
                self.mask[:n, :n], float("-inf")
            )
            attended = (
                (self.dropout(F.softmax(scores, dim=-1)) @ v).transpose(1, 2).reshape(batch, n, d)
            )
            residual = x + self.dropout(attended @ p[prefix + "wo"] + p[prefix + "bo"])
            hidden = F.gelu(
                norm(residual, "2", prefix) @ p[prefix + "w1"] + p[prefix + "b1"],
                approximate="tanh",
            )
            x = residual + self.dropout(hidden @ p[prefix + "w2"] + p[prefix + "b2"])
        return norm(x, "_final") @ p["lm_weight"] + p["lm_bias"]
