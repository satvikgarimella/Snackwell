# DE1-SoC synthesis project

Provisional part: `5CSEMA5F31C6`. Confirm the board model before adding pins or
attempting to program hardware.

This project runs **Analysis & Synthesis only** on the full `snackwell_device`
using Quartus Prime Lite/Standard with Cyclone V support:

```sh
quartus_map snackwell --read_settings_files=on --write_settings_files=off
```

It has no physical transport or pin assignments. `core.sdc` sets a proposed
50 MHz clock, not an achieved frequency. Do not use a full `quartus_sh --flow
compile` as a substitute for defining and checking the board interface.

The top level now includes a global-memory request/response master, in addition
to the host register bus. A board-specific memory endpoint must implement the
[handshake and coordinated-reset contract](../../docs/GLOBAL_MEMORY.md). No
external RAM controller is included. Earlier synthesis/resource measurements
predate the tensor-memory and global-memory additions and do not prove they fit.

The project was prepared on macOS; it needs execution and review in Quartus
on Windows/Linux. See [the synthesis guide](../../docs/SYNTHESIS.md) for the
local Yosys flow, verification, limitations, and remaining board work.

## Core-only fit check

To check core capacity before building the physical board interface, run from
the repository root in a Quartus Prime Lite/Standard command prompt with
Cyclone V device support installed:

```sh
quartus_sh -t fpga/de1_soc/run_fit.tcl
```

This creates a fresh project in `build/quartus-fit/<timestamp>-<pid>/`, runs
Analysis & Synthesis, the Fitter, then static timing analysis. It uses the full
`snackwell_device` RTL with virtual pins for the bus and a real clock input.
It does not run the assembler or generate a board bitstream.

Read `run-status.txt` and `output_files/snackwell_fit.fit.rpt` for the actual
fitter outcome and utilization. Review `snackwell_fit.sta.rpt` for core timing
slack at the proposed 50 MHz, plus unconstrained paths. A successful STA process
does not imply passing timing. External I/O delays are intentionally unspecified
because the board transport has not been designed.

A passing result establishes that this core, with these virtual ports, fits.
The physical transport, model/KV storage, and their integration logic are not
included. They need another fit and timing run when integrated.

The script has been checked locally for source selection, Tcl control flow,
and failure handling, but has **not been executed by Quartus**. Quartus is not
available on the current Mac. The
[November 5 resource audit](../../docs/measurements/fpga-fit-20261105/README.md)
is an estimate, not a replacement for this run.
