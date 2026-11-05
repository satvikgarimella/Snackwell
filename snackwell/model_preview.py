"""Fast CPU diagnostics for model quality. Never imported by the RTL runner."""

from snackwell.character_model import reference_forward
from snackwell.decoding import prepare_prompt
from snackwell.numeric import bits
from snackwell.reference.inference import categorical_reference, min_p_reference, stable_softmax


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

    def generate(self, prompt, tokens, decoder, *, cache=True, characters=None, heal=True):
        """Generate up to tokens, optionally stopping after a target of new characters."""
        if type(tokens) is not int or not 1 <= tokens <= 256:
            raise ValueError("preview length must be 1..256 tokens")
        if characters is not None and (
            type(characters) is not int or not 1 <= characters <= tokens
        ):
            raise ValueError("character target must be positive and no larger than the token limit")
        generated = prompt
        context = self.model["config"]["context"]
        ids, token_prefix = prepare_prompt(self.model, prompt, cache=cache, heal=heal)
        for _ in range(tokens):
            if len(ids) > context:
                ids = ids[-max(1, 3 * context // 4) :]
            logits = self.logits(ids)
            if token_prefix:
                logits = [
                    z if self.vocabulary[i].startswith(token_prefix) else -float("inf")
                    for i, z in enumerate(logits)
                ]
            if decoder.mode == "greedy":
                selected = max(range(len(logits)), key=logits.__getitem__)
            else:
                probabilities = stable_softmax([x / decoder.temperature for x in logits])
                weights = [bits(p) for p in probabilities]
                if getattr(decoder, "min_p", 0):
                    weights = min_p_reference(weights, decoder.min_p)
                selected = categorical_reference(weights, bits(decoder.draw()))
            generated += self.vocabulary[selected][len(token_prefix) :]
            token_prefix = ""
            ids = ids + [selected] if cache else (ids + [selected])[-context:]
            # Finish the emitted token, even if it crosses the character target.
            if characters is not None and len(generated) - len(prompt) >= characters:
                break
        return generated
