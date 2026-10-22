# Bringing Snackwell onto an FPGA

The board-independent loading and launch interface is implemented and tested in
RTL simulation. The full device has also passed Yosys synthesis, including a
preliminary Cyclone V mapping. The provisional board is a DE1-SoC; its identity
still needs confirmation. No Quartus fit, timing closure, bitstream, or physical
board execution has been verified. See [the synthesis guide](SYNTHESIS.md) for
commands, generated netlists, and the Quartus project.
`scripts/check_fpga.py` checks the RTL interface; it does not create a bitstream.

```sh
python3 scripts/check_fpga.py
python3 -m unittest tests.test_device -v
```

The first command needs Icarus Verilog and Verilator. It disables the old
simulation initializers, loads programs and all 1,024 shared-memory words through
the host bus, launches vector add/multiply and tensor/SIMD matrix multiplication,
and checks every output and untouched memory word against independent references.
The Python tests also compare outputs and execution cycles against the existing
SM runner, including partial lane masks, sparse warp selection and a nonzero PC.
Logs and numerical outputs are saved under `build/fpga/` and `build/tests/device_python/`.

## Hardware path

`rtl/fpga/snackwell_device.sv` instantiates the same full SM execution core as
`snackwell_sm.sv`. A board transport must drive its synchronous register bus:

```text
Laptop host driver
        | physical connection (board-specific; still to be implemented)
Board transport / clock-domain crossing
        | request and response bus
snackwell_device
        | program loading, shared-memory access, launch and status
Full SM: scheduler, integer/FP32 units, tensor array, register file, shared memory
```

Requests are accepted on a rising clock edge with `request_valid && request_ready`.
Hold address, write flag and data stable until acceptance. Each accepted request,
including a write, produces exactly one response. Hold `response_ready` high to
consume it. `response_valid`, `response_data` and `response_error` remain stable
under backpressure. Only one request is outstanding at a time. All signals use
the SM clock; this bus is not AXI, UART, USB or a clock-domain crossing itself.

Addresses are byte addresses; accesses are aligned, full 32-bit words. Invalid
addresses, misalignment, invalid control values and writes to read-only registers
return `response_error=1` without applying the requested change.

| Address | Access | Meaning |
| --- | --- | --- |
| `0x0000` | Read | ID/version: `0x534e4b31` (`SNK1`) |
| `0x0004` | Read | Status: bit 0 busy, bit 1 done, bit 2 fault; bits 11:8 halted warps |
| `0x0008` | Write | `1` starts execution; `2` aborts and resets execution |
| `0x000c` | Read/write | Entry PC: aligned byte address, 0–4092 |
| `0x0010` | Read/write | Warp mask, 1–15; default 15 |
| `0x0014` | Read/write | Four lane-mask nibbles, warp 0 in bits 3:0; default `0xffff` |
| `0x0018` | Read/write | Execution-cycle limit, nonzero; default 1,000,000 |
| `0x001c` | Read | Execution cycles from the latest launch, excluding loading/launch setup |
| `0x1000`–`0x1ffc` | Read/write | 1,024 program words |
| `0x2000`–`0x2ffc` | Read/write | 1,024 shared-memory words |

Memory access and configuration changes are rejected while busy. Status and
cycle counters remain readable; abort remains available. Start acknowledges the
command before execution completes. Poll until busy clears, then require done=1
and fault=0 before trusting outputs. Done stays set until a new start, abort or
reset. Timeout/abort clear execution and set fault; partially written outputs
must be discarded. Invalid bus requests report an error on that response and do
not themselves set the execution fault bit.

## Loading and reset

1. Reset the device; read and check its ID.
2. Load a complete program image, padding unused instructions with `EXIT`.
3. Load a complete shared-memory image, including inputs and output guards.
4. Set the entry PC, warp/lane masks and cycle limit.
5. Write start, poll status, then read the output words after successful completion.

Launch resets registers, scoreboards, warps and execution units. It preserves
loaded program and data memories. Device reset also preserves those memories;
their contents after power-up are unspecified. The host must initialize them
before the first launch. A reset cancels outstanding bus requests/responses.

Host memory operations reuse the banked shared-memory controller while the SM is
idle. There is no second data-memory port or concurrent host/SM memory access.
The original simulation SM keeps its default memory-clearing reset behavior and
smoke-test program. The device disables both conveniences; its tests never read
or modify internal DUT arrays.

## Remaining board work

- Identify the board, FPGA part, available memory and supported build environment.
- Implement the physical host transport and matching laptop driver, including
  clock-domain crossing and board clock/reset handling.
- Add actual pin assignments and timing constraints from the board documentation.
- Run the prepared project in Quartus and complete place-and-route; inspect
  actual ALM/DSP/memory use and timing.
  The current asynchronous instruction reads and shared-memory access structure
  may need changes for efficient block-RAM inference. Lint is not synthesis.
- Program the device and repeat the numerical tests over the physical connection.
- Connect the model runtime to this transport and arrange model/KV storage and
  transfers. The current model has 577,796 bytes of FP32 weights, while the SM
  has 4,096 bytes of shared memory. Loading chunks remains necessary unless the
  memory architecture changes. Thousands of small host transfers could dominate
  performance; no FPGA inference speed is measured or promised here.

The first physical milestone is a verified vector kernel, followed by matrix
multiplication and then the model. Text remains a laptop input/output concern:
the host maps characters to vocabulary IDs and prints the returned character.
Moving inference to hardware will not change the checkpoint's language quality.
