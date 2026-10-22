"""Summarize rising-edge observations from the real SM kernel testbench."""

import csv
from collections import Counter

from snackwell.assembler import OPCODES


def summarize_trace(path, cycles):
    with path.open() as stream:
        rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if [row["cycle"] for row in rows] != list(range(1, cycles + 1)):
        raise ValueError("trace must contain exactly one row per simulated execution cycle")
    names = {value: name for name, value in OPCODES.items()}
    issue_cycles, instructions, tensor_phases = Counter(), Counter(), Counter()
    loads = stores = muls = adds = macs = arithmetic_pe_cycles = 0
    steps, step_start = [], None
    for row in rows:
        if row["instruction_accepted"]:
            issue_cycles["instruction_issue"] += 1
            name = names.get(row["opcode"], f"OP_{row['opcode']:02x}")
            instructions[name] += 1
            lanes = row["active_mask"].bit_count()
            loads += lanes if name == "LD_SHARED" else 0
            stores += lanes if name == "ST_SHARED" else 0
            muls += lanes if name == "FMUL" else 0
            adds += lanes if name in ("FADD", "FSUB") else 0
        elif row["reconvergence_accepted"]:
            issue_cycles["reconvergence"] += 1
        elif row["dependency_stall"]:
            issue_cycles["dependency_check"] += 1
        elif row["resource_stall"]:
            issue_cycles["resource_check"] += 1
        elif not row["issue_valid"]:
            issue_cycles["no_eligible_warp"] += 1
        else:
            raise ValueError("unclassified SM issue cycle")

        # Stable SM adapter trace IDs; LOAD=2 is retained for older traces.
        state = row["tensor_state"]
        if state == 4:
            phase = "result_stream" if row["array_result_valid"] else "array_compute"
        else:
            phase = {
                0: "idle",
                1: "register_gather",
                2: "operand_load",
                3: "array_launch",
                5: "final_commit",
            }[state]
        tensor_phases[phase] += 1
        macs += row["pe_mac_accept"].bit_count()
        arithmetic_pe_cycles += row["pe_arithmetic_active"].bit_count()
        if row["array_step_start"]:
            if step_start is not None:
                raise ValueError("overlapping array steps in trace")
            step_start = {
                "step": row["array_step"],
                "start_cycle": row["cycle"],
                "useful_mac_pairs": row["pe_mac_accept"].bit_count(),
            }
        if row["array_step_advance"]:
            if step_start is None or row["array_step"] != step_start["step"]:
                raise ValueError("array advanced without a matching step launch")
            steps.append({**step_start, "cycles": row["cycle"] - step_start["start_cycle"] + 1})
            step_start = None
    if step_start is not None:
        raise ValueError("unfinished array step in completed kernel")
    compute_cycles = tensor_phases["array_compute"]
    return {
        "cycles": cycles,
        "issue_cycles": dict(issue_cycles),
        "instructions_issued": sum(instructions.values()),
        "opcode_counts": dict(sorted(instructions.items())),
        "load_words": loads,
        "store_words": stores,
        "shared_inflight_cycles": sum(row["shared_inflight"] for row in rows),
        "shared_service_cycles": sum(row["shared_servicing"] for row in rows),
        "shared_response_wait_cycles": sum(
            row["shared_response_valid"] and not row["shared_response_ready"] for row in rows
        ),
        "conflicting_memory_requests": sum(
            row["shared_start"] and row["shared_bank_conflict"] for row in rows
        ),
        "simd_fp_inflight_cycles": sum(row["fp_inflight"] for row in rows),
        "simd_lane_multiplies": muls,
        "simd_lane_adds": adds,
        "tensor_phases": dict(tensor_phases),
        "tensor_mac_pairs": macs,
        "tensor_row_commits": sum(row["tensor_row_commit"] for row in rows),
        "tensor_steps": steps,
        # State occupancy, not completed FLOPs / theoretical peak throughput.
        "tensor_pe_arithmetic_cycles": arithmetic_pe_cycles,
        "tensor_pe_arithmetic_fraction": (
            arithmetic_pe_cycles / (16 * compute_cycles) if compute_cycles else None
        ),
        "tensor_useful_step_slot_fraction": macs / (16 * len(steps)) if steps else None,
    }
