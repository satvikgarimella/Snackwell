"""CPU-only sampling comparison; never a replacement for the RTL runner."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "snackwell/character_model.py").is_file())))
from snackwell.character_model import encode_prompt, load_checkpoint
from snackwell.cli.generation_quality import repetition
from snackwell.decoding import Decoder
from snackwell.model_preview import ModelPreview
from snackwell.numeric import bits
from snackwell.reference.inference import categorical_reference, stable_softmax

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--min-p', type=float, default=.05)
parser.add_argument('--penalty', type=float, default=1.1)
parser.add_argument('--ngram', type=int, default=8)
parser.add_argument('--temperature', type=float, default=.7)
args = parser.parse_args()
model = load_checkpoint(args.checkpoint)
preview = ModelPreview(model)
vocab = model['config']['vocabulary']
context = model['config']['context']
prompts = ['My lord, I have something to say', 'ROMEO:', 'JULIET:', 'To be, or not to be',
           'I pray you,', 'The king', 'What is your name?', 'Good night,',
           'The king is dead, and we must now', 'the sky today is', 'My name is ', 'I love', 'To be, or not to ']
cases = []
for mode, seed in [('greedy', None), ('sample', 92617660), ('sample', 1729), ('sample', 1730)]:
    for prompt in prompts:
        ids = encode_prompt(model, prompt, pad=False)
        prefix = ''
        if 'tokenizer' in model and len(ids) > 1:
            tail = vocab[ids[-1]]
            if any(t.startswith(tail) and len(t) > len(tail) for t in vocab):
                prefix = tail
                ids.pop()
        history = list(ids)
        generated = ''
        decoder = Decoder(mode, seed, args.temperature)
        for step in range(256):
            if len(ids) > context:
                ids = ids[-3 * context // 4:]
            logits = preview.logits(ids)
            seen = set(history[-context:])
            banned = set()
            if args.ngram and len(history) >= args.ngram:
                suffix = history[-(args.ngram - 1):]
                banned = {history[i + args.ngram - 1] for i in range(len(history) - args.ngram + 1)
                          if history[i:i + args.ngram - 1] == suffix}
            for i, z in enumerate(logits):
                if (prefix and not vocab[i].startswith(prefix)) or i in banned:
                    logits[i] = float('-inf')
                elif i in seen:
                    logits[i] = z / args.penalty if z >= 0 else z * args.penalty
            if mode == 'greedy':
                selected = max(range(len(logits)), key=logits.__getitem__)
            else:
                probs = stable_softmax([z / decoder.temperature for z in logits])
                threshold = max(probs) * args.min_p
                probs = [p if p >= threshold else 0.0 for p in probs]
                selected = categorical_reference([bits(p) for p in probs], bits(decoder.draw()))
            generated += vocab[selected][len(prefix):]
            prefix = ''
            ids.append(selected)
            history.append(selected)
            if len(generated) >= 192:
                break
        case = dict(mode=mode, seed=seed, prompt=prompt, continuation=generated, **repetition(generated))
        cases.append(case)
        print(json.dumps(case), flush=True)
        args.out.write_text(json.dumps(dict(kind='CPU diagnostic, not RTL', checkpoint_sha256=model['sha256'],
                                          temperature=args.temperature, min_p=args.min_p, penalty=args.penalty, ngram=args.ngram, cases=cases), indent=2)+'\n')
