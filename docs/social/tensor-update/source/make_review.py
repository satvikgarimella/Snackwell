from pathlib import Path
from html import escape
import json
root=Path(__file__).resolve().parents[1]
posts=json.loads((root/'source/posts.json').read_text())
style='''*{box-sizing:border-box}body{margin:0;background:#fffefa;color:#22262f;font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}main{max-width:1150px;margin:0 auto;padding:48px 28px 80px}h1{font-size:42px;line-height:1.12;letter-spacing:-1px;margin:16px 0}h2{font-size:25px;margin:48px 0 14px}.eyebrow{color:#606674;font-size:13px;letter-spacing:2px;font-weight:700}p{max-width:850px}a{color:#1971c2}video{display:block;width:min(100%,800px);margin:18px auto;background:#fffefa;border:1px solid #ddd9d0;border-radius:12px}img{display:block;width:100%;height:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.media{border:1px solid #d9d8d0;border-radius:12px;overflow:hidden;background:white}.links{padding:14px 18px;font-size:14px}.post{padding:22px 26px;border-left:3px solid #7048c6;background:#f6f3fc;margin:22px 0}.post p{white-space:pre-wrap;margin:8px 0 0}.meta{font-size:13px;color:#606674}code{background:#f0ede6;padding:2px 5px;border-radius:4px}.note{padding:14px 18px;border:1px solid #d9d8d0;border-radius:8px;font-size:14px}.loglink{display:inline-block;margin-top:12px}@media(max-width:700px){.grid{grid-template-columns:1fr}h1{font-size:34px}main{padding:26px 16px}}'''
html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',f'<title>Snackwell — FP32 + tensor update</title><style>{style}</style><main>',
'<div class="eyebrow">SNACKWELL / BUILD LOG 03</div><h1>FP32 + the first tensor instruction.</h1>',
'<p>A draft thread and its media. Diagrams describe the implemented SM; animation values were checked against an RTL run. These files stay local unless you choose to share them.</p>',
'<p><a href="thread-draft.md">Thread draft</a> · <a href="alt-text.md">Media alt text</a> · <a href="single-post.txt">Opening post</a></p>',
'<h2>Tensor flow animation</h2><video controls playsinline preload="metadata" poster="assets/tensor-flow-poster.png"><source src="assets/tensor-flow.mp4" type="video/mp4"></video>',
'<p class="links"><a href="assets/tensor-flow.mp4" download>MP4 · 1080 × 1080</a> · <a href="assets/tensor-flow.gif" download>GIF · 720 × 720</a> · <a href="assets/tensor-flow-poster.png" download>Poster frame</a></p>',
'<div class="note">Ten logical steps, with time compressed for explanation. Every PE finishes before operands advance. FP32 multiplication and addition round separately.</div>',
'<h2>Editable diagrams</h2><div class="grid">']
for name,title in [('01-sm-architecture','SM architecture'),('02-processing-element','Inside one PE'),('03-tmul-registers','TMUL register layout'),('04-systolic-array','Systolic array overview')]:
 html += [f'<div class="media"><img src="assets/{name}.png" alt="{title}"><div class="links">{title} · <a href="assets/{name}.png" download>PNG</a> · <a href="assets/{name}.excalidraw" download>Excalidraw source</a></div></div>']
html += ['</div><h2>Fresh regression output</h2><div class="media"><img src="assets/05-test-results.png" alt="35 of 35 testbenches passed in the fresh Icarus Verilog regression."><div class="links"><a href="assets/05-test-results.png" download>Test results image</a> · <a href="assets/06-testing-screenshot.png" download>Log screenshot</a> · <a href="testing-log.html">Verbatim captured stdout</a> · <a href="source/regression.txt">Plain-text log</a></div></div>',
'<h2>The thread</h2>']
for p in posts:
 attach='' if not p['asset'] else f' · Attach {p["asset"]}'
 html += [f'<article class="post"><div class="meta">{p["number"]}/7 · {p["characters"]} characters{attach}</div><p>{escape(p["text"])}</p></article>']
html += ['<p class="meta">Fixed 4×4 FP32 tiles, one tensor job at a time. Verified in simulation; FPGA synthesis and hardware execution remain future work.</p></main></html>']
(root/'review.html').write_text('\n'.join(html))
raw=(root/'source/regression.txt').read_text()
meta=json.loads((root/'source/regression-metadata.json').read_text())
rawhtml='\n'.join('<span class="pass">PASS</span>'+escape(line[4:]) if line.startswith('PASS ') else escape(line) for line in raw.splitlines())
loghtml='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Snackwell — captured regression output</title><style>*{box-sizing:border-box}html,body{margin:0;width:1600px;background:#111821;color:#e7f5ff;font-family:Menlo,monospace}main{padding:42px 64px 44px}h1{margin:0 0 14px;font:600 38px Arial,sans-serif}.caption{font:20px Arial,sans-serif;color:#a6b1c0;margin-bottom:28px}.command{color:#83d5a4;font-size:24px;margin:0 0 25px}pre{font:22px/1.22 Menlo,monospace;margin:0;white-space:pre-wrap}.pass{color:#83d5a4}</style><main><h1>Snackwell simulation regression</h1>'''+f'<div class="caption">Verbatim stdout · captured {escape(meta["timestamp"][:10])} UTC · Icarus Verilog</div><div class="command">$ python3 scripts/run_tests.py</div><pre>{rawhtml}</pre></main></html>'
(root/'testing-log.html').write_text(loghtml)
print('Review gallery and verbatim log view written.')
