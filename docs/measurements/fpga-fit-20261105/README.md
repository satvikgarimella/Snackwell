# DE1-SoC fit assessment, November 5, 2026

**Result: inconclusive until the FPGA fitter runs.** No place-and-route, timing
closure, bitstream generation, or physical test was performed in this audit.
The current host is an Apple Silicon Mac without Quartus. The provisional
target remains `5CSEMA5F31C6`; the physical board identity is not confirmed.

All current RTL source hashes and the complete source-file set match the saved
October 21 Cyclone V synthesis report. The local mapped netlist's cell counts
and exported Verilog hash also match. Repeating unchanged synthesis would not
answer the remaining placement/routing question.

## Capacity comparison

| Resource | Existing core mapping | Target capacity |
| --- | ---: | ---: |
| 27-by-27 DSP multipliers | 20 | 87 |
| Flip-flops | 20,697 | 128,300 registers |
| Dedicated M10K memory blocks | 0 | 397 |
| Physical MLABs | 192 under the packing assumptions below | 768 |
| ALMs | Approximately 27,241 under favorable packing | 32,070 |

ALMs are the chip's physical logic units. The approximately **84.94%** estimate
is not a measured utilization figure or a guaranteed upper bound. It leaves
4,829 ALMs before overhead and incompatible packing. Registers must share
occupied logic units for this estimate to hold; adding transport/storage logic
will also consume resources. Routing can fail even when counts are below capacity.

The estimate uses the actual mapped cells:

- 6,541 six-input LUTs each occupy an ALM. Pair the 9,638 five-input LUTs with
  9,638 of the 13,998 two-/three-input LUTs, then pair the remaining small LUTs
  and 7,553 four-input LUTs. This gives 22,136 logic ALMs.
- 6,234 arithmetic cells form 408 carry chains. Rounding each chain to pairs
  gives 3,185 arithmetic ALMs before placement restrictions.
- 3,072 one-bit MLAB slices form 96 groups of 32 bits with matching address,
  enable, and clock signals. Two 20-bit physical MLABs per group gives 192
  MLABs, occupying 1,920 ALMs. A simple 3,072/20 division misses group padding.
- Assume the 20,697 flip-flops share these ALMs and absorb the inverter cells.
  Control-set compatibility, LAB restrictions, routing, clocks, and virtual-pin
  overhead are not modeled. Quartus can also remap the original RTL differently.

The mapping stores two instruction-memory read copies in MLABs and uses MLABs
for all four shared-memory banks. Dedicated block RAM inference deserves review
if the actual Quartus result is too large or too slow.

## Model storage is a separate integration question

The compact sentence checkpoint has 41,569 FP32 parameters: 166,276 bytes
(162.38 KiB). Its full-context K/V tensors require 32,768 bytes (32 KiB), for
199,044 bytes (194.38 KiB) combined. This is below the chip's 508,160 bytes
(496.25 KiB) of nominal M10K capacity, before activation buffers, layout padding,
memory replication, and controller overhead. This comparison covers inference;
gradients and optimizer state are additional training storage.

**These model buffers are absent from the synthesized core.** The current SM
exposes 4 KiB of data memory. Chunk transfers or a larger memory interface must
be integrated before this capacity can be used by the model.

## Get the actual result

On Windows/Linux with Quartus Prime Lite/Standard and Cyclone V support, run:

```sh
quartus_sh -t fpga/de1_soc/run_fit.tcl
```

The script runs a core-only fit with virtual bus pins in a fresh output directory.
No board transport or programming file is produced. See the
[fit instructions](../../../fpga/de1_soc/README.md#core-only-fit-check).
Its vendor-tool execution remains unverified until Quartus is available.

Machine-readable evidence and source hashes: [resource-audit.json](resource-audit.json).

Resource capacities come from the
[Altera Cyclone V product table](https://docs.altera.com/api/khub/documents/s60vJiu_kjIh2yag_Ea_yg/content).
The board part comes from
[Terasic's DE1-SoC specifications](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=836&PartNo=2).
Packing assumptions follow the `intel_alm/common/alm_sim.v` and `mem_sim.v`
definitions in the same Yosys 0.70 installation that produced the netlist.
