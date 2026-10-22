# Media alt text

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

## 06-testing-screenshot.png
Browser screenshot of the verbatim stdout captured from a fresh Snackwell regression on September 17, 2026 UTC. It shows the command python3 scripts/run_tests.py, a PASS line for each of the 35 testbenches, and the final 35/35 testbenches passed summary. Only empty margins were cropped.
