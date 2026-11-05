"""Verify a selected candidate in RTL; never supply reference execution outputs."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from snackwell.character_model import load_checkpoint
from snackwell.decoding import Decoder
from snackwell.kv_cache import Generation, KVCache, latency_report
from snackwell.model_preview import ModelPreview
from snackwell.model_runner import forward
from snackwell.numeric import bits, value
from snackwell.reference.inference import verify_approx
from snackwell.simulator import SimulatorSession, compile_model_runner
from snackwell.training import write_json
from snackwell.transformer_training import run_step
from snackwell.transformer_training_data import Corpus
from snackwell.transformer_training_runtime import TransformerTrainingRuntime

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("window", "training", "benchmark"), required=True)
    parser.add_argument("--artifacts", choices=("summary", "full"), default="summary")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = load_checkpoint(args.checkpoint)
    simulator = compile_model_runner(args.out, "verilator", compact=args.artifacts == "summary")
    common = {
        "checkpoint_sha256": model["sha256"],
        "rtl_sha256": simulator.source_sha256,
        "python_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "snackwell").rglob("*.py"))
        },
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with SimulatorSession(simulator) as execution:
        if args.mode == "benchmark":
            rows = []
            paths = [
                ("baseline", ROOT / "models/subword_transformer/checkpoint.json"),
                ("candidate", args.checkpoint),
            ]
            for label, path in paths:
                current = load_checkpoint(path)
                prompt = "My lord, I have something to say"
                decoder = Decoder("sample", 92617660, 0.7)
                generation = Generation(current, prompt)
                tokens = []
                for index in range(32):
                    got = generation.next(
                        args.out / label / f"token_{index:03d}",
                        execution,
                        draw=decoder.draw(),
                        temperature=0.7,
                    )
                    tokens.append(
                        {
                            k: got[k]
                            for k in (
                                "next_character",
                                "wall_seconds",
                                "cycles",
                                "launches",
                                "cache",
                                "selection",
                            )
                        }
                    )
                    if (index + 1) % 8 == 0:
                        print(label, index + 1, repr(generation.text), flush=True)
                expected = ModelPreview(current).generate(
                    prompt, 32, Decoder("sample", 92617660, 0.7)
                )
                if generation.text != expected:
                    raise AssertionError("RTL continuation differs from CPU diagnostic")
                row = {
                    "label": label,
                    "checkpoint_sha256": current["sha256"],
                    "generated": generation.text,
                    "tokens": tokens,
                    "latency": latency_report(tokens, True, subword=True),
                    "verified": True,
                    "cpu_text_identical": True,
                }
                rows.append(row)
                write_json(args.out / f"{label}.json", row)
                print(label, row["latency"], flush=True)
            write_json(args.out / "result.json", {**common, "verified": True, "comparisons": rows})
            return
        import torch
        import torch.nn.functional as F

        from snackwell.torch_model import ReferenceTransformer

        torch.set_num_threads(1)
        c = model["config"]
        corpus = Corpus(
            ROOT / "build/data/tinyshakespeare.txt",
            c["context"],
            c["vocabulary"],
            model["tokenizer"],
        )
        net = ReferenceTransformer(c, model["weights"]).double().eval()
        if args.mode == "training":
            ids, targets = corpus.window("train", 2759, 23)
            loss = F.cross_entropy(
                net(torch.tensor([ids])).reshape(len(ids), -1), torch.tensor(targets)
            )
            loss.backward()
            print("Independent autograd complete; running RTL training", flush=True)
            runtime = TransformerTrainingRuntime(args.out / "step", execution)
            got, updated = run_step(runtime, model, ids, targets, 0.0001)
            checks = {}
            exact_updates = {}
            for name, parameter in net.parameters_by_name.items():
                checks[name] = verify_approx(
                    got["tensors"]["gradients"][name],
                    parameter.grad.flatten().tolist(),
                    "transformer_training",
                )
                gradient = torch.tensor(
                    [value(w) for w in got["tensors"]["gradients"][name]], dtype=torch.float32
                )
                expected = (
                    torch.tensor(model["weights"][name], dtype=torch.float32)
                    - torch.tensor(0.0001, dtype=torch.float32) * gradient
                )
                if [bits(x) for x in expected.tolist()] != [
                    bits(x) for x in updated["weights"][name]
                ]:
                    raise AssertionError(f"FP32 SGD update differs: {name}")
                exact_updates[name] = len(updated["weights"][name])
            report = {
                **common,
                "training_tokens": len(ids),
                "model_context": c["context"],
                **{k: got[k] for k in ("verified", "loss", "cycles", "launches", "seconds")},
                "torch_loss": loss.item(),
                "torch_gradients": checks,
            }
            write_json(args.out / "updated-checkpoint.json", updated)
            reloaded = load_checkpoint(args.out / "updated-checkpoint.json")
            if reloaded["weights"] != updated["weights"] or reloaded.get("tokenizer") != model.get(
                "tokenizer"
            ):
                raise AssertionError("checkpoint reload changed weights or tokenizer")
            with torch.no_grad():
                after = ReferenceTransformer(c, reloaded["weights"]).double().eval()
                report["torch_loss_after_update"] = F.cross_entropy(
                    after(torch.tensor([ids])).reshape(len(ids), -1), torch.tensor(targets)
                ).item()
            inference = forward(
                reloaded, ids[:8], args.out / "reloaded_inference", execution, last_only=True
            )
            report.update(
                exact_fp32_update_words=sum(exact_updates.values()),
                checkpoint_reload_exact=True,
                reloaded_inference_verified=inference["verified"],
            )
            write_json(args.out / "result.json", report)
            print({k: v for k, v in report.items() if k != "torch_gradients"}, flush=True)
            return
        ids, _ = corpus.window("validation", 2759, 0)
        cache = KVCache()
        print(f"Checking full {len(ids)}-token context", flush=True)
        got = forward(model, ids, args.out / "full_window", execution, kv_cache=cache)
        with torch.no_grad():
            expected = F.softmax(net(torch.tensor([ids])), dim=-1)[0].flatten().tolist()
        checks = [
            {
                "phase": "full_window",
                "verified": got["verified"],
                "torch": verify_approx(
                    [bits(p) for p in got["probabilities"]], expected, "transformer"
                ),
            }
        ]
        generation = Generation(model, "a")
        generation.ids = ids + [got["selection"]["token_id"]]
        generation.cache = cache
        generation.steps = 1
        for phase in ("rebuild", "decode"):
            print("Checking cache", phase, flush=True)
            actual = generation.next(args.out / phase, execution)
            if actual["cache"]["phase"] != phase:
                raise AssertionError("Unexpected cache phase")
            reference = forward(
                model, actual["input_ids"], args.out / (phase + "_full"), execution, last_only=True
            )
            if (
                actual["probabilities"][-len(c["vocabulary"]) :]
                != reference["probabilities"][-len(c["vocabulary"]) :]
                or actual["selection"] != reference["selection"]
            ):
                raise AssertionError("Cached/recomputed RTL outputs differ")
            checks.append(
                {"phase": phase, "bit_identical": True, "context_tokens": len(actual["input_ids"])}
            )
        write_json(args.out / "result.json", {**common, "verified": True, "checks": checks})
        print(checks, flush=True)


if __name__ == "__main__":
    main()
