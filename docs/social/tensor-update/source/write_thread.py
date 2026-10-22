from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
posts=[
("Snackwell can do matrix math now 😎\n\nSince the last update, I’ve added FP32 multiply/add/subtract and a 4×4 systolic array, then connected the tensor unit to the SM with a TMUL instruction.\n\nHere’s how the data moves through it 👇",'01-sm-architecture.png'),
("First came FP32: FMUL, FADD, and FSUB.\n\nThese take multiple cycles, so the SM has to remember the issuing warp, destination register, and active lanes until the result returns.\n\nDependent instructions wait; independent work can keep running.",None),
("Then came the systolic array.\n\nA values move right. B values move down. Each of the 16 processing elements keeps one output value and accumulates four products.\n\nThe animation shows the operands moving and the partial sums building up 👇",'tensor-flow.mp4'),
("Inside each PE: an FP32 multiplier, an adder, and a local accumulator.\n\nThis version rounds after both operations, so the MAC is non-fused.\n\nThe whole array advances when every PE finishes. Ten logical steps here take more than ten clock cycles.",'02-processing-element.png'),
("The array now runs through the real SM:\n\nTMUL R8, R0, R4\n\nA lives in R0–R3, B in R4–R7, and C goes to R8–R11. Each register holds one row across four lanes.\n\nAll four destination registers stay reserved until the last result row commits.",'03-tmul-registers.png'),
("Then I tested the full path.\n\n35/35 testbenches pass. The SM tensor suite checks 185 matrix cases / 2,960 outputs, plus overlapping registers, all 16 lane masks, competing warps, divergence, reset, and writeback backpressure.",'05-test-results.png'),
("Snackwell now has FP32 execution and its first integrated tensor unit.\n\nIt’s still a fixed 4×4 design with one tensor job at a time, verified in simulation.\n\nNext: shared-memory conflict replay, a small assembler, and the first reusable kernels. More soon!",None)
]
for i,(text,asset) in enumerate(posts,1):
    assert len(text)<=275,(i,len(text))
    print(f'Post {i}: {len(text)} characters')
(root/'source/posts.json').write_text(json.dumps([{'number':i,'text':text,'asset':asset,'characters':len(text)}for i,(text,asset) in enumerate(posts,1)],ensure_ascii=False,indent=2)+'\n')
md=['# Snackwell update 3 — FP32 + tensor execution\n','Draft thread. Nothing has been posted. Each post stays under 280 characters; media notes are separate from the copy.\n']
for i,(text,asset) in enumerate(posts,1):
 md += [f'## {i}/7\n',text+'\n']
 if asset:md += [f'**Attach:** `assets/{asset}`\n']
(root/'thread-draft.md').write_text('\n'.join(md))
(root/'single-post.txt').write_text(posts[0][0]+'\n')
(root/'alt-text.md').write_text('''# Media alt text

## 01-sm-architecture.png
Snackwell SM block diagram. Four resident warps feed a scheduler and dependency checks, then a register file. Operands go to integer SIMD, FP32 SIMD, a tensor unit with sixteen FP32 MAC processing elements, and a shared-memory load/store unit. Results share a writeback port with saved warp, destination, and lane-mask tags. The tensor tile remains reserved until its final row commits.

## tensor-flow.mp4 / tensor-flow.gif
Animated multiplication of two small 4-by-4 matrices on sixteen processing elements. Blue A values move right, orange B values move down, and each cell retains a green output accumulator. A diagonal wave of activity fills and drains the array over ten logical steps. Counters show 64 completed multiply-accumulates and the resulting matrix. Each logical step waits for every PE to finish and does not represent one clock cycle.

## 02-processing-element.png
One FP32 processing element. An A operand and a B operand feed a multiplier; the product then passes to an adder with feedback from the local accumulator C[row,column]. Each output element accumulates four products in increasing k order. Multiplication and addition round separately, so this is a non-fused MAC. Retained A and B tokens are forwarded right and down.

## 03-tmul-registers.png
Register layout for TMUL R8, R0, R4. A uses R0 through R3, B uses R4 through R7, and C uses R8 through R11. Each vector register is a row, with lanes 0 through 3 holding its columns. The SM gathers both input tiles, executes the array, then writes four result rows. All destinations stay reserved until the last row commits. The saved active mask selects output columns.

## 04-systolic-array.png
Static 4-by-4 systolic array diagram. Sixteen processing elements each retain one C[row,column] value. A operands move horizontally to the right and B operands move vertically downward. Every PE accumulates four products, for 64 MACs over ten logical steps with variable clock latency.

## 05-test-results.png
A typeset view of real output from python3 scripts/run_tests.py, captured on September 17, 2026 UTC. All 35 testbenches passed, including FP32, the systolic array, the tensor word interface, and full-SM tensor execution. The tensor suite checks 185 reference matrices and 2,960 output elements plus hazards, masks, multiwarp execution, divergence, reset, and backpressure. This is simulation evidence, not an FPGA result.
''')
with (root/'alt-text.md').open('a') as out:
 out.write('\n## 06-testing-screenshot.png\n\nBrowser screenshot of verbatim stdout from the fresh September 17, 2026 UTC regression. It shows python3 scripts/run_tests.py, 35 PASS lines, and the 35/35 testbenches passed summary. Only empty margins were cropped.\n')
print('Thread and alt text written.')
