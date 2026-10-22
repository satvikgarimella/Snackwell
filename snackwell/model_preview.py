"""Fast CPU diagnostics for model quality. Never imported by the RTL runner."""

from snackwell.character_model import encode_prompt, reference_forward
from snackwell.numeric import bits
from snackwell.reference.inference import categorical_reference, stable_softmax


class ModelPreview:
    def __init__(self, model, backend="torch"):
        self.model = model
        self.backend = backend
        self.vocabulary = model["config"]["vocabulary"]
        if backend == "torch":
            import torch

            from snackwell.torch_model import ReferenceTransformer

            torch.set_num_threads(1)
            self.network = ReferenceTransformer(model["config"], model["weights"]).eval()
        elif backend != "scalar":
            raise ValueError("preview backend must be torch or scalar")

    def logits(self, ids):
        if self.backend == "scalar":
            return reference_forward(self.model, ids)["logits"][-len(self.vocabulary) :]
        import torch

        with torch.no_grad():
            return self.network(torch.tensor([ids]))[0, -1].tolist()

    def generate(self, prompt, tokens, decoder, *, cache=True):
        if type(tokens) is not int or not 1 <= tokens <= 256:
            raise ValueError("preview length must be 1..256 tokens")
        generated = prompt
        context = self.model["config"]["context"]
        ids = encode_prompt(self.model, prompt, pad=not cache)
        for _ in range(tokens):
            if len(ids) > context:
                ids = ids[-max(1, 3 * context // 4) :]
            logits = self.logits(ids)
            if decoder.mode == "greedy":
                selected = max(range(len(logits)), key=logits.__getitem__)
            else:
                probabilities = stable_softmax([x / decoder.temperature for x in logits])
                selected = categorical_reference(
                    [bits(p) for p in probabilities], bits(decoder.draw())
                )
            generated += self.vocabulary[selected]
            ids = ids + [selected] if cache else (ids + [selected])[-context:]
        return generated
