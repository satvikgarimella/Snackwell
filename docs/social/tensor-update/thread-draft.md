# Snackwell update 3 — FP32 + tensor execution

Draft thread. Nothing has been posted. Each post stays under 280 characters; media notes are separate from the copy.

## 1/7

Snackwell can do matrix math now 😎

Since the last update, I’ve added FP32 multiply/add/subtract and a 4×4 systolic array, then connected the tensor unit to the SM with a TMUL instruction.

Here’s how the data moves through it 👇

**Attach:** `assets/01-sm-architecture.png`

## 2/7

First came FP32: FMUL, FADD, and FSUB.

These take multiple cycles, so the SM has to remember the issuing warp, destination register, and active lanes until the result returns.

Dependent instructions wait; independent work can keep running.

## 3/7

Then came the systolic array.

A values move right. B values move down. Each of the 16 processing elements keeps one output value and accumulates four products.

The animation shows the operands moving and the partial sums building up 👇

**Attach:** `assets/tensor-flow.mp4`

## 4/7

Inside each PE: an FP32 multiplier, an adder, and a local accumulator.

This version rounds after both operations, so the MAC is non-fused.

The whole array advances when every PE finishes. Ten logical steps here take more than ten clock cycles.

**Attach:** `assets/02-processing-element.png`

## 5/7

The array now runs through the real SM:

TMUL R8, R0, R4

A lives in R0–R3, B in R4–R7, and C goes to R8–R11. Each register holds one row across four lanes.

All four destination registers stay reserved until the last result row commits.

**Attach:** `assets/03-tmul-registers.png`

## 6/7

Then I tested the full path.

35/35 testbenches pass. The SM tensor suite checks 185 matrix cases / 2,960 outputs, plus overlapping registers, all 16 lane masks, competing warps, divergence, reset, and writeback backpressure.

**Attach:** `assets/05-test-results.png`

## 7/7

Snackwell now has FP32 execution and its first integrated tensor unit.

It’s still a fixed 4×4 design with one tensor job at a time, verified in simulation.

Next: shared-memory conflict replay, a small assembler, and the first reusable kernels. More soon!
