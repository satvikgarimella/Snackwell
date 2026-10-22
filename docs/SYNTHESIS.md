# Synthesizing Snackwell

`scripts/synthesize.py` synthesizes the **full** `snackwell_device`: four warps,
integer and FP32 SIMD, the 4×4 FP32 tensor array, register file, instruction
memory, shared memory, and host loading/launch interface. It uses the actual RTL
with `SYNTHESIS` defined. The device disables simulation memory initialization;
programs and data enter through the host bus.

## Run on macOS, Linux, or Windows

Use Yosys with the `read_slang` SystemVerilog frontend. The portable YoWASP build
can be installed in a local environment (on Windows use its `Scripts` directory
instead of `bin`):

```sh
python3 -m venv build/synthesis-venv
build/synthesis-venv/bin/python -m pip install yowasp-yosys==0.70.0.0.post1259
python3 scripts/synthesize.py --verify
```

The script finds `yosys` or `yowasp-yosys` on PATH, then the local environment.
`--yosys /path/to/yosys` selects a specific executable. It needs `read_slang`
built in; an older Yosys executable with only `read_verilog -sv` is insufficient.
`-j 1` avoids unsupported parser threads in WebAssembly.

The generic flow lowers memories, arithmetic, and control into gates and
flip-flops, runs ABC optimization, and rejects structural errors, inferred
latches, and unresolved operators. `--verify` compiles **only the exported gate
netlist** as the DUT, using Verilator. Icarus remains available through
`--simulator icarus`, but is much slower on these large gate netlists. It checks the host protocol and
executes vector add, vector multiply, tensor matmul, and SIMD matmul. Each kernel
must match the numerical reference, every shared-memory word, and the RTL cycle
count. The ISA package supplies constants to the testbench; no original RTL
implementation is linked into the netlist simulation.

Artifacts go into `build/synthesis/generic/`:

- `synthesize.ys`, `sources.f`: the exact synthesis input commands.
- `synthesis.log`: compiler diagnostics and synthesis passes.
- `netlist.v`, `netlist.json`: the synthesized circuit.
- `stats.json`: gate/register counts, **not FPGA LUT/ALM utilization**.
- `result.json`: tool version, source/netlist SHA256 hashes, and verification results.
- `synthesis-result.json`: synthesis status before optional netlist tests finish.
- Kernel directories and `protocol.log`: post-synthesis simulation evidence.

A failed run removes the previous success report and generated netlist before
starting. Synthesis can also run without `--verify`; that report explicitly has
no netlist verification result. These numerical tests are not an exhaustive
formal equivalence proof and do not check physical propagation delays.

## Measured synthesis result: 2026-10-21

Yosys 0.70 (`28ba3cb92`) synthesized the unchanged hardware design. Both a
generic gate netlist and a Cyclone V primitive netlist were produced. Both
passed host protocol tests and all four vector/matrix kernel tests against the
original RTL and independent numerical references. The recorded reports are in
[`fpga/reports/`](../fpga/reports/).

| Cyclone V mapping cell | Count |
| --- | ---: |
| 27×27 DSP multiplier primitives | 20 |
| Flip-flops | 20,697 |
| Logic ALUT primitives (2–6 inputs) | 37,730 |
| Arithmetic ALUT primitives | 6,234 |
| 32×1-bit MLAB primitives | 3,072 |
| M10K block-memory primitives | 0 |

An ALUT count is **not an ALM count**. These MLAB primitives are individual
32×1-bit slices; 20 such slices represent a physical 32×20-bit MLAB, subject to
legal packing. Do not compare the primitive total directly with the board's
physical MLAB count. Quartus must determine the actual packed resource use and
whether the design fits. The absence of M10Ks makes memory inference a useful
next optimization to investigate after the first Quartus report.

Verified execution cycles were 73 for vector add, 73 for vector multiply, 136
for tensor matmul, and 303 for SIMD matmul. These exclude loading/readback and
are simulation cycle counts, not measured hardware throughput.

## Provisional target: DE1-SoC

The board model still needs confirmation. Terasic lists the DE1-SoC's part as
**5CSEMA5F31C6**, a Cyclone V SoC. The current target is its FPGA fabric; the ARM
HPS and external memory are not connected to Snackwell.

```sh
python3 scripts/synthesize.py --target cyclonev --verify
```

This uses Yosys's `synth_intel_alm` mapper, including DSP and memory inference.
The resulting MISTRAL primitive counts are a preliminary mapping result, not a
Quartus fit report. There is no pin, I/O-buffer, or board clock-buffer mapping in
this flow. It does not create an Intel `.sof` programming file or establish a
maximum clock frequency. `--verify` tests the mapped netlist with the matching
Yosys installation's primitive simulation models and records their hashes.
If those models are outside the usual installation directories, provide
`--cell-library-dir /path/to/yosys/share/intel_alm/common`.

The generic netlist contains hundreds of thousands of individual gates, so
compiling its testbench can take several minutes. Compilation has a ten-minute
limit and each kernel simulation has a five-minute limit; reaching either limit
is a failed verification run, not a pass. These are wall-clock test budgets, not
changes to the kernel's simulated cycle limit or numerical acceptance criteria.

## Quartus Analysis & Synthesis

`fpga/de1_soc/snackwell.qpf` is an **Analysis & Synthesis only** project for
Quartus Prime Lite/Standard with Cyclone V device support. It consumes the
original RTL so Quartus can infer its own memories and DSPs. From a supported
Windows/Linux machine with Quartus on PATH:

```sh
cd fpga/de1_soc
quartus_map snackwell --read_settings_files=on --write_settings_files=off
```

Inspect `output_files/snackwell.map.rpt` and the warnings. This project has not
been run in Quartus on the Mac. `core.sdc` specifies a proposed 50 MHz core
clock; it is a constraint to meet, not a measured result. There are deliberately
no board pin assignments or external I/O timing assumptions. Do not program
this raw register-bus top as a board design.

## What remains after synthesis

1. Confirm the board and obtain a supported Quartus Windows/Linux environment.
2. Check real Quartus ALM/register/DSP/RAM use and address any fit problems.
   Asynchronous instruction reads and the shared-memory port structure are
   important memory-inference checks; generic gate counts cannot answer these.
3. Add the board transport, clock/reset handling, pin assignments, and complete
   timing constraints. The current bus is synchronous and is not USB/UART/AXI.
4. Run placement, routing, and timing analysis, then generate the `.sof` file.
5. Program the board and repeat vector/matrix checks over the physical link.
6. Connect the inference/training runtime and arrange model/data transfers.

Synthesis, fitting/timing, and executing on a physical board are separate
milestones. No FPGA inference throughput is inferred from simulator timings.

Sources: [Yosys slang frontend](https://yosyshq.readthedocs.io/projects/yosys/en/stable/cmd/index_frontends.html),
[YoWASP](https://github.com/YoWASP/yosys),
[Terasic DE1-SoC specifications](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=836&PartNo=2),
[Quartus operating systems](https://www.intel.com/content/www/us/en/docs/programmable/683472/25-1/starting-the-software.html).
