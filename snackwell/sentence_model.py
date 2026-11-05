"""Generate a bounded answer with the ordinary verified transformer RTL runner."""

import json
import re
from pathlib import Path

from snackwell.kv_cache import Generation, latency_report
from snackwell.sentence_task import LOCATIONS, OBJECTS
from snackwell.tokenizer import Tokenizer


def validate_prompt(prompt):
    match = re.fullmatch(
        r"The ([a-z]+) is in the ([a-z]+)\. The ([a-z]+) is in the ([a-z]+)\. Where is the ([a-z]+)\?",
        prompt,
    )
    if not match:
        raise ValueError(
            "Use two facts followed by 'Where is the <object>?' in the example format."
        )
    first, place1, second, place2, query = match.groups()
    if (
        first not in OBJECTS
        or second not in OBJECTS
        or place1 not in LOCATIONS
        or place2 not in LOCATIONS
    ):
        raise ValueError("Use objects and places from the displayed vocabulary.")
    if first == second or place1 == place2 or query not in (first, second):
        raise ValueError(
            "Use two different objects and places, then ask about one of those objects."
        )


def complete(model, prompt, output, executable, *, on_token=None, max_tokens=12):
    validate_prompt(prompt)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32:
        raise ValueError("max_tokens must be 1..32")
    tokenizer = Tokenizer.restore(model["tokenizer"])
    ids = tokenizer.encode(prompt)
    if not ids or len(ids) >= model["config"]["context"]:
        raise ValueError("prompt must fit in the model context with room for an answer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    # The question ends at a punctuation boundary, so token healing is unnecessary.
    generation = Generation(model, prompt, heal=False)
    reports, text = [], ""
    for step in range(max_tokens):
        if len(generation.ids) > model["config"]["context"]:
            raise ValueError("answer exceeds context; refusing to drop facts from the prompt")
        result = generation.next(output / f"token_{step:03d}", executable)
        reports.append(result)
        piece = result["next_character"]
        text += piece
        if on_token:
            on_token(piece)
        if "." in piece:
            break
    report = dict(
        execution="Snackwell RTL; independent kernel and whole-model verification",
        verified=True,
        complete=text.endswith("."),
        checkpoint_sha256=model.get("sha256"),
        prompt=prompt,
        answer=text,
        decoding="greedy; no token filters",
        rtl_source_sha256=getattr(executable, "source_sha256", None),
        cycles=sum(r["cycles"] for r in reports),
        launches=sum(r["launches"] for r in reports),
        latency=latency_report(reports, True, subword=True),
        tokens=[
            dict(
                token_id=r["selection"]["token_id"],
                text=r["next_character"],
                accuracy=r["accuracy"],
                wall_seconds=r["wall_seconds"],
                cycles=r["cycles"],
                launches=r["launches"],
                cache=r["cache"],
            )
            for r in reports
        ],
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
