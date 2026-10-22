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

The project was prepared on macOS; it needs execution and review in Quartus
on Windows/Linux. See [the synthesis guide](../../docs/SYNTHESIS.md) for the
local Yosys flow, verification, limitations, and remaining board work.
