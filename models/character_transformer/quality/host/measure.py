"""ABBA measurement of the previous host and direct-clock host, same RTL."""
import json
import statistics
import types
from dataclasses import replace
from pathlib import Path

from snackwell import model_runner, simulator
from snackwell.character_model import load_checkpoint
from snackwell.decoding import Decoder
from snackwell.kv_cache import Generation, latency_report

root = Path('build/inference-quality-v4')
current_forward = model_runner._forward
current_write_hex = simulator.write_hex
source = Path('snackwell/model_runner.py').read_text()
old_code = '''    # Encode each validated tensor in one call, retaining the same FP32 rounding.
    weights = {
        name: list(struct.unpack(f"<{len(data)}I", struct.pack(f"<{len(data)}f", *data)))
        for name, data in model["weights"].items()
    }'''
assert old_code in source
source = source.replace(old_code, '    weights = {name: [bits(a) for a in data] for name, data in model["weights"].items()}')
previous = types.ModuleType('previous_model_runner')
exec(compile(source, 'previous_model_runner.py', 'exec'), previous.__dict__)
model = load_checkpoint('models/character_transformer/checkpoint.json')
runner = simulator.compile_model_runner(root/'timing')
prompt = 'My lord, I have something to say'
runs = []
try:
    for label, repeat in [('before', 1), ('after', 1), ('after', 2), ('before', 2)]:
        if label == 'before':
            model_runner._forward = previous._forward
            simulator.write_hex = lambda path, words: path.write_text(''.join(f'{w:08x}\n' for w in words))
            executable = replace(runner, session_executable=None)
        else:
            model_runner._forward = current_forward
            simulator.write_hex = current_write_hex
            executable = runner
        generation = Generation(model, prompt)
        decoder = Decoder(seed=92617660, temperature=.7)
        reports = []
        output = root/'timing'/f'{label}-{repeat}'
        for step in range(16):
            result = generation.next(output/f'token_{step:03d}', executable, draw=decoder.draw(), temperature=.7)
            assert result['verified']
            reports.append({k: result[k] for k in (
                'wall_seconds','cycles','launches','cache','next_character','probabilities',
                'sampling_probabilities','selection','accuracy')})
            print(label, repeat, step, result['wall_seconds'], result['cache']['phase'], repr(result['next_character']), flush=True)
        row = dict(label=label, repeat=repeat, generated=generation.text, tokens=reports,
                   latency=latency_report(reports, True), cycles=sum(r['cycles'] for r in reports),
                   launches=sum(r['launches'] for r in reports),
                   wall_seconds=sum(r['wall_seconds'] for r in reports))
        (output/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        runs.append(row)
finally:
    model_runner._forward = current_forward
    simulator.write_hex = current_write_hex
for row in runs[1:]:
    assert (row['generated'],row['cycles'],row['launches']) == (runs[0]['generated'],runs[0]['cycles'],runs[0]['launches'])
    for old,new in zip(runs[0]['tokens'],row['tokens']):
        for key in ['probabilities','sampling_probabilities','selection','accuracy']:
            assert old[key] == new[key], (row['label'],key)
summary = {}
for label in ['before','after']:
    selected = [r for r in runs if r['label']==label]
    summary[label] = {key: statistics.mean(r[key] for r in selected) for key in ['wall_seconds','cycles','launches']}
    summary[label]['latency'] = {key: statistics.mean(r['latency'][key] for r in selected) for key in [
        'first_character_seconds','decode_seconds','rebuild_seconds','subsequent_character_seconds','rebuild_steps']}
record = dict(note='Same checkpoint, RTL, prompt, draws, temperature and numerical checks; ABBA; compilation excluded; no concurrent training or test runs.',
              checkpoint_sha256=model['sha256'],source_sha256=runner.source_sha256,
              prompt=prompt,seed=92617660,temperature=.7,bit_identical=True,summary=summary,runs=runs)
(root/'timing.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
