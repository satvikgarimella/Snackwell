# Snackwell full SM architecture

Archived build-post snapshot. Its diagrams and captured results describe that
milestone. See the [current repository overview](../../../README.md) for the
implemented hardware and latest commands.

- **snackwell-full-architecture.excalidraw** — editable boxes, labels, groups and arrows. Open or drag this file into Excalidraw.
- **snackwell-full-architecture.png** — 3820 × 2810 rendered preview.
- **snackwell-full-architecture.svg** — scalable vector preview.
- **code/make_full_architecture.py** — source for all three formats; uses Python 3, Pillow, and macOS Arial fonts.
- **code/rtl-coverage.json** — all 28 module types reachable from the current `snackwell_sm` top, with source paths and SHA-256 hashes.
- **code/diagram-manifest.json** — diagram block/source mapping and connections.

This is a functional architecture drawing of the currently implemented SM.
It includes instruction memory and decode, warp state/scheduling, dependency
checks, register storage and selection, integer/FP32/tensor execution, the LSU
and shared memory, branches, divergence/reconvergence, writeback, and state
update feedback. Clock/reset and the external launch/debug/status interfaces
are identified. Arrows group related signals; they are not pipeline-stage or
cycle boundaries.

The three top-level control modules are `warp_control_unit`, `issue_control`
and `issue_stage`; their functions are expanded into the state, scheduler,
scoreboard and issue blocks. Arithmetic hierarchies are shown inside the
execution groups. Every PE owns its own FP32 multiply/add hardware and local
accumulator; it does not share the SIMD FP32 unit.

The tensor adapter gathers both matrices over four register-read cycles, feeds
the word interface, and packs the result into four writes. Destination
reservations remain active until the final row commits. The array uses separate
multiply and add rounding and waits for every PE before advancing operands.

The branch subsystem compares active lanes, records a rejoin PC with `SSY`,
stores deferred paths in a four-entry stack per warp, and restores PCs/masks at
reconvergence. State changes occur on accepted control events. `EXIT` waits for
the issuing warp's ordinary and tensor writes to finish.

The memory block documents the present behavior: bank conflicts are detected;
conflicting stores are suppressed and conflicting loads return zero. Replay
has not been implemented. Instruction memory currently contains sixteen
32-bit words and indexes with `PC[5:2]`.

Standalone alternatives in the repository, such as `single_warp_core`,
`instruction_frontend`, `program_counter`, `simd_execution_unit`, and
`simd_fp32_mul`, are not instantiated by `snackwell_sm` and therefore are not
drawn as extra hardware. PC storage is inside `warp_state_table` in this SM.
The ISA constants are defined by the shared `snackwell_pkg` package.

## Regenerate

From the Snackwell repository, using a Python installation with Pillow:

```sh
python3 docs/social/post-03/code/make_full_architecture.py
```

The generator checks text bounds, unique element identifiers, valid arrow
bindings, and coverage of all module types reachable from `snackwell_sm`.
It records the source hashes used for the diagram. The PNG and SVG use the
same geometry and labels as the Excalidraw scene. The preview was visually
checked for clipping, text collisions, and arrow placement. The file was
generated locally; it has not been imported into an external web service.
