"""CPU-only screening. Never called by Snackwell's RTL runner."""

import argparse
import hashlib
import json
import random
import re
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARTIFACTS = ROOT / "build/pretrained-selection-20261028"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate",
        choices=(
            "stories260K",
            "stories15M",
            "tinystories28m",
            "tinystories33m",
            "simplestories1m",
            "simplestories5m",
            "simplestories11m",
            "simplestories35m",
            "smollm2-135m",
            "smollm2-135m-instruct",
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=64,
        help="Stop a preliminary screen early; never implies a complete pass",
    )
    parser.add_argument("--suite", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.suite.read_text())
    output = args.output or HERE / (args.candidate + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    prompts = [
        (group, prompt) for group in ("core", "stress") for prompt in protocol[group + "_prompts"]
    ]
    if args.candidate.startswith("stories"):
        path = ARTIFACTS / (args.candidate + ".bin")
        tokenizer = ARTIFACTS / (
            "tok512.bin" if args.candidate == "stories260K" else "tokenizer.bin"
        )
        config = dict(
            zip(
                ("width", "hidden", "layers", "heads", "kv_heads", "vocabulary", "context"),
                struct.unpack("<7i", path.read_bytes()[:28]),
            )
        )
        implementation = "author's llama2.c, FP32, -O3 -ffp-contract=off; xorshift sampling"
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(1)
        path = ARTIFACTS / args.candidate
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=(path / "model.safetensors").exists(),
            )
            .float()
            .eval()
        )
        config = model.config.to_dict()
        config["total_parameters"] = sum(p.numel() for p in model.parameters())
        implementation = (
            "official Transformers "
            + model.config.model_type
            + ", FP32 CPU; Python random 24-bit uniform sampling"
        )
        # The SimpleStories authors explicitly specify EOS=1 in the model card.
        # The HF config has Llama's default EOS=2, which is actually '!'.
        eos_id = 1 if args.candidate.startswith("simplestories") else tokenizer.eos_token_id
        config["evaluation_eos_token_id"] = eos_id
    report = dict(
        kind="CPU quality evaluation; no Snackwell execution or training",
        candidate=args.candidate,
        config=config,
        implementation=implementation,
        protocol=protocol,
        protocol_sha256=hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_cases=args.max_cases,
        cases=[],
    )
    print(args.candidate, config, flush=True)
    for mode, seed in [("greedy", None)] + [("sample", s) for s in protocol["sample_seeds"]]:
        for group, prompt in prompts:
            if len(report["cases"]) >= args.max_cases:
                return
            limit = protocol["new_token_limit"]
            if args.candidate.startswith("stories"):
                result = subprocess.run(
                    [
                        str(ARTIFACTS / "probe"),
                        str(path),
                        str(tokenizer),
                        prompt,
                        "0" if mode == "greedy" else "0.7",
                        str(seed or 1),
                        str(limit),
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=120,
                )
                continuation, extra = result.stdout, json.loads(result.stderr)
            else:
                rng = random.Random(seed)
                if args.candidate.endswith("-instruct"):
                    messages = [
                        {
                            "role": "user",
                            "content": "Continue this text naturally in two or three sentences: "
                            + prompt,
                        },
                        {"role": "assistant", "content": prompt},
                    ]
                    ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, continue_final_message=True
                    )
                else:
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                emitted, past, current = [], None, ids
                decoded_prompt = tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                sentence_complete = False
                with torch.inference_mode():
                    for _ in range(limit):
                        result = model(
                            torch.tensor([current]), past_key_values=past, use_cache=True
                        )
                        logits = result.logits[0, -1]
                        if mode == "greedy":
                            token = int(logits.argmax())
                        else:
                            probabilities = (logits.double() / protocol["temperature"]).softmax(-1)
                            draw = rng.getrandbits(24) / 2**24
                            token = min(
                                int(torch.searchsorted(probabilities.cumsum(-1), draw, right=True)),
                                len(probabilities) - 1,
                            )
                        emitted.append(token)
                        if protocol.get("stop_after_sentence"):
                            partial = tokenizer.decode(
                                ids + emitted,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )[len(decoded_prompt) :]
                            sentence_complete = bool(re.search(r"[.!?]", partial))
                            if sentence_complete:
                                break
                        if token == eos_id:
                            break
                        past, current = result.past_key_values, [token]
                decoded_prompt = tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                decoded = tokenizer.decode(
                    ids + emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                if not decoded.startswith(decoded_prompt):
                    raise ValueError("tokenizer changed the decoded prompt prefix")
                continuation = decoded[len(decoded_prompt) :]
                extra = dict(
                    prompt_tokens=len(ids),
                    new_tokens=len(emitted),
                    ended=emitted[-1] == eos_id,
                    prompt_ids=ids,
                    generated_ids=emitted,
                    sentence_complete=sentence_complete,
                )
            case = dict(
                index=len(report["cases"]),
                group=group,
                prompt=prompt,
                mode=mode,
                seed=seed,
                continuation=continuation,
                **extra,
            )
            report["cases"].append(case)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(case["index"], mode, seed, repr(prompt + continuation), flush=True)


if __name__ == "__main__":
    main()
