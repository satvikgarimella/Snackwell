# Snackwell

A GPU written in SystemVerilog, with a custom ISA, Python assembler, and kernels
that run through the full streaming multiprocessor (SM) in RTL simulation.

The SM has four resident warps, four lanes per warp, integer and FP32 execution,
divergence/reconvergence, 4 KB of banked shared memory, and a 4×4 FP32 systolic
array. Shared-memory conflicts are serialized; scoreboarding tracks dependencies
across multi-cycle execution and writeback.

## Run it

Requires Python 3.10+ and Icarus Verilog (`iverilog` and `vvp` on `PATH`).
Simulation, assembly, and verification use only the Python standard library.
For faster model inference, install Verilator (`brew install verilator` on macOS)
with a C++ compiler and `make`. The model runner selects it automatically and
caches the compiled RTL. Icarus remains the default for kernel and RTL tests.

```sh
python3 scripts/run_tests.py
python3 scripts/run_kernel.py vector_add
python3 scripts/run_kernel.py matmul_compare
python3 scripts/run_kernel.py all
python3 scripts/run_model.py --tokens 1
python3 scripts/train_snackwell.py
python3 scripts/train_transformer.py
```

The kernel suite covers vector operations, tiled matrix multiplication, reductions,
LayerNorm, GELU, softmax, linear layers, attention, and a complete transformer block.
The model runner uses a saved CPU-trained character-transformer checkpoint.
Outputs are checked against independent numerical references.

`train_snackwell.py` trains an 8→16→4 GELU network from scratch on a nonlinear
regression task. Its forward pass, loss, backward pass and SGD updates all execute
on the SM in RTL simulation. The terminal shows batch loss, before/after held-out
predictions, and the saved checkpoint. The default 200-step run reduced training
loss from 0.172689 to 0.033176 and held-out loss from 0.137247 to 0.038508.
See [training](docs/TRAINING.md) for the equations, checks and resume command.
`train_transformer.py` trains the complete causal character transformer, including
attention, LayerNorm, embeddings, cross-entropy and SGD. It reads the local
TinyShakespeare file at `build/data/tinyshakespeare.txt` (or `--data PATH`). The
default starts a 4,657-parameter model; `--checkpoint PATH` continues training a
larger exported model. Both paths execute all forward/backward arithmetic on the
SM. See [full transformer training](docs/TRANSFORMER_TRAINING.md) for commands,
measurements and the distinction between training correctness and language quality.

To enter prompts and watch the transformer predict in your terminal:

```sh
python3 scripts/run_model.py --interactive --tokens 4
```

Type a prompt and press Enter. The text grows as each verified RTL prediction
finishes, with the current kernel and cycle count shown underneath. `/tokens N`
changes the number of new characters; `/quit` exits. Ctrl+C cancels a prediction.
The first character processes the prompt; later characters reuse cached keys
and values until the context needs rebuilding.

For quick language-quality experiments, use the separate CPU preview:

```sh
build/model-venv/bin/python scripts/preview_model.py --interactive
```

This optional PyTorch tool does not execute RTL or verify the hardware. It uses
the same checkpoint and sampling settings, so you can inspect longer text before
running a slower hardware check. See [language-quality testing](docs/LANGUAGE_QUALITY.md).

Generation samples from the model's probabilities by default. Character selection
runs as an SM kernel; the host supplies seeded random draws. Add `--seed 1729`
to reproduce a run, or `--decode greedy` to always choose the most likely character.
`--temperature 0.7` favors more likely characters; `--temperature 1` uses the
original distribution. Interactive mode also accepts `/temperature T` and `/seed N`.
The default checkpoint has four transformer blocks, four attention heads,
width 64, a 32-character context and 144,449 parameters. The larger experimental
checkpoint was removed as the default: its modest loss improvement did not fix
sentence quality and more than doubled generation time. It remains available in
`models/character_transformer/wide-context64/`. Neither model reliably produces
coherent sentences. See the [default rollback and kernel changes](docs/INFERENCE_REPAIR.md).
The latest [simulation-host changes and model experiments](docs/INFERENCE_HOST.md)
keep numerical verification and language-quality evidence separate.
The [subword experiment](docs/SUBWORD_MODEL.md) adds a 512-entry tokenizer and
compares a new candidate on the same held-out text. It also extends RTL inference
and full training to that vocabulary without enlarging the SM's shared memory.
Try it with `--checkpoint models/subword_transformer/checkpoint.json`; the default
remains the character model. Subword predictions can emit several characters.
Generation uses a KV cache with periodic context rebuilds. `--no-kv-cache` restores
the padded, sliding-window computation.

For one prompt, use `python3 scripts/run_model.py --stream --prompt 'Hello ' --tokens 4`.
An optional local browser view with attention weights and character probabilities
is available with `--live`.

Each run writes assembly, memory images, and checked results under `build/`.
Model inference skips per-cycle trace logging by default; add `--trace` to save
traces and detailed profiles, or `--simulator icarus --trace` for the original
diagnostic path. All numerical checks run regardless of these options. See
[inference measurements](docs/MODEL_INFERENCE.md#simulation-speed) for timings.
A failed check exits with a nonzero status. See [the development guide](CONTRIBUTING.md)
for focused tests and formatting commands.

## Repository

| Directory | Contents |
| --- | --- |
| [`rtl/`](rtl/) | SM, frontend, execution units, memory, and ISA package |
| [`tb/`](tb/) | SystemVerilog testbenches |
| [`sim/`](sim/) | Verilator session host and simulation memory access |
| [`snackwell/`](snackwell/) | Assembler, kernel generators, simulator, and numerical references |
| [`scripts/`](scripts/) | Command-line launchers |
| [`tests/`](tests/) | Python regression tests |
| [`kernels/`](kernels/) | Hand-written kernels and generated examples grouped by workload |
| [`models/`](models/) | Trained checkpoint, evaluation inputs, and verification record |
| [`docs/`](docs/) | Design notes, walkthroughs, and archived build posts |

## Read the design

- Hardware: [FP32](docs/FP32.md), [tensor array and TMUL](docs/TENSOR.md),
  [shared memory](docs/MEMORY.md), [FPGA host interface](docs/FPGA.md)
- Programs: [ISA and first kernels](docs/KERNELS.md),
  [tensor versus SIMD matmul](docs/MATMUL.md), [inference kernels](docs/INFERENCE_KERNELS.md)
- Models: [transformer block](docs/TRANSFORMER.md),
  [trained-model inference](docs/MODEL_INFERENCE.md),
  [training on the SM](docs/TRAINING.md),
  [full transformer training](docs/TRANSFORMER_TRAINING.md),
  [RTL fine-tuning experiment](docs/RTL_FINETUNING.md),
  [whiteboard walkthrough](docs/INFERENCE_WALKTHROUGH.md),
  [report walkthrough](docs/REPORT_WALKTHROUGH.md)
- Visuals: [architecture diagrams](docs/diagrams/), [build posts](docs/social/README.md)

## Current scope

The complete design is verified in RTL simulation. Model arithmetic runs on the
SM; the simulation host moves data between launches. The hardware currently has
4 KB of shared memory and a 1,024-instruction program memory. The board-independent
FPGA endpoint supports program/data loading, kernel launch and result readback;
`python3 scripts/check_fpga.py` verifies it through its public bus in simulation.
The full device has also passed Yosys synthesis and preliminary Cyclone V
mapping. [Synthesis commands and the provisional DE1-SoC Quartus project](docs/SYNTHESIS.md)
are available; Quartus fit, timing closure, and board execution remain unverified.

The GELU network and complete character transformer support verified
forward/backward passes and SGD on Snackwell. Longer training runs, language
quality, training throughput, global memory, a physical host transport and board
driver, cooperating warp barriers, and FPGA resource/timing validation still need
work. No physical FPGA run has been verified. The default character checkpoint
was originally trained with PyTorch on the CPU; RTL training exports separate
checkpoints that the same inference runner can load. Earlier teaching cores in
`rtl/core/` implement a smaller instruction subset than the full SM.

The original Jonathan P. Dawson multiplier and its copyright/license notice are
preserved in [`multiplier.v`](rtl/execution/fp32/multiplier.v). The active FP path
now uses Snackwell's bounded-latency multiplier; see [FP32](docs/FP32.md).
