# Working on Snackwell

Run commands from the repository root. The launchers in `scripts/` also work by
absolute path from another directory; output defaults remain under this checkout.
Python 3.10+ and Icarus Verilog are required. Set `IVERILOG` or `VVP` to override
the compiler/runtime executable.

Verilator is optional for faster model inference and additional simulator
equivalence tests. It requires a C++ compiler and `make`; use
`brew install verilator` on macOS and set `VERILATOR` to override its path.
Model commands select it automatically when available. The first build is
cached under `build/simulators/verilator/` with a key derived from the RTL,
testbench, simulation host, tools and build options. Build logs are kept beside the executable.

## Checks

```sh
python3 scripts/run_tests.py
python3 scripts/run_tests.py tb_snackwell_sm_fp32 tb_snackwell_sm_tensor
python3 -m unittest tests.test_assembler -v
python3 -m unittest tests.test_training -v
python3 -m unittest tests.test_transformer_training -v
python3 scripts/write_kernel_examples.py --check
```

The full runner executes every RTL bench, then the Python tests. Logs and a
machine-readable summary are saved in `build/tests/`. Kernel and model runners
remove stale success reports before execution and check results before writing
`result.json`.

RTL benches and kernel tests continue to use Icarus. When Verilator is installed,
`tests/test_simulator.py` also compares exact outputs, register values, cycle
counts and tensor traces across both simulators. These extra comparisons are
skipped when Verilator is absent. Terminal and browser tests explicitly use
Icarus so they do not need to compile a new Verilator executable on startup.

For cycle profiling, use `python3 scripts/run_model.py --tokens 1 --trace`.
Add `--simulator icarus` to reproduce the original inference simulation path.
Tracing changes diagnostic output only; all memory and numerical checks remain
enabled without it.

Install the optional formatter/linter in a local environment:

```sh
python3 -m venv build/dev-venv
build/dev-venv/bin/python -m pip install -r requirements-dev.txt
make check RUFF=build/dev-venv/bin/ruff
make format RUFF=build/dev-venv/bin/ruff
```

`make check` runs lint, formatting, generated-example checks, and the regression
suite. `make kernels` runs all default workloads; `make model` verifies one
prediction from the saved model. Neither command trains or downloads a model.
CPU reference training has separate dependencies in `requirements-model.txt`.
`python3 scripts/train_snackwell.py` runs the separate RTL training demo without
PyTorch. The training tests additionally compare gradients with PyTorch autograd
when that optional dependency is installed, and compare simulators when Verilator
is available.

## Where to change things

- **Hardware:** `rtl/`, with regression benches in `tb/`. `rtl/snackwell_pkg.sv`
  defines the instruction opcodes. Preserve the license in the third-party multiplier.
- **Assembly:** `snackwell/assembler.py`. Its opcode table is checked against the RTL.
- **Kernel generation:** `snackwell/kernels/`. The generated examples live under
  `kernels/nn/<workload>/`; edit the emitters, then run `make examples`.
- **Model execution:** `snackwell/model_runner.py` defines the forward pass;
  `model_runtime.py` breaks operations into shared-memory-sized launches.
- **Training:** `snackwell/training.py` connects the GELU network's forward/backward
  passes and SGD; `training_runtime.py` splits loss, reduction and update kernels
  into bounded launches. The independent backward reference lives in
  `snackwell/reference/training.py`.
- **Transformer training:** `transformer_training.py` connects the complete model;
  `transformer_training_runtime.py` launches the additional backward kernels and
  loss. Its independent double-precision reverse-mode reference is in
  `snackwell/reference/transformer_training.py`; optional PyTorch checks provide a
  second implementation. Never use reference activations or gradients as RTL inputs.
- **Simulation:** `snackwell/simulator.py` owns Icarus/Verilator invocation, memory-image
  transport, and completion checks. All callers use the same source discovery.
- **Verification:** `snackwell/reference/` contains independent numerical models.
  `character_model.py` also contains the checkpoint validator and whole-model oracle.
  Keep references independent of RTL intermediates and assembly approximations.
- **CLI:** `snackwell/cli/`. `scripts/` contains small launchers, not reusable logic.

Use descriptive names at module boundaries, small functions with a clear purpose,
and comments for numerical assumptions, ordering, or handshake behavior. Formatting
is configured in `pyproject.toml`. Avoid decorative section banners and comments
that restate the next line.

Build posts in `docs/social/` are historical snapshots. Keep their captured logs,
images, and manifests together. Current simulation artifacts belong in `build/`.
The model's verification record identifies the source it tested; refresh it only
after rerunning the recorded checks.
