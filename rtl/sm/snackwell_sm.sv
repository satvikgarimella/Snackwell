module snackwell_sm (
    input logic clk,
    input logic reset,
    input logic run_enable,

    // =================================================
    // WARP LAUNCH
    // =================================================

    input logic        launch_enable,
    input logic [1:0]  launch_warp_id,
    input logic [31:0] launch_pc,
    input logic [3:0]  launch_active_mask,

    // =================================================
    // DEBUG REGISTER PRELOAD
    // =================================================

    input logic        debug_write_enable,
    input logic [1:0]  debug_warp_id,
    input logic [1:0]  debug_lane_id,
    input logic [3:0]  debug_reg_addr,
    input logic [31:0] debug_write_data,

    // =================================================
    // ISSUE / SCHEDULER STATUS
    // =================================================

    output logic       issue_valid,
    output logic       issue_accept,
    output logic [1:0] selected_warp_id,

    output logic [31:0] selected_pc,
    output logic [31:0] instruction,
    output logic [5:0]  opcode,

    output logic [3:0] selected_active_mask,

    // =================================================
    // WARP STATUS
    // =================================================

    output logic [3:0] valid_mask,
    output logic [3:0] ready_mask,
    output logic [3:0] halted_mask,
    output logic [3:0] blocked_mask,

    // =================================================
    // STALL STATUS
    // =================================================

    output logic dependency_stall,
    output logic resource_stall,

    // =================================================
    // EXECUTION DEBUG
    // =================================================

    output logic [127:0] rs1_data,
    output logic [127:0] rs2_data,
    output logic [127:0] alu_result,

    // =================================================
    // SHARED MEMORY DEBUG
    // =================================================

    output logic       shared_bank_conflict,
    output logic [3:0] shared_conflict_mask,

    // =================================================
    // SIMT CONTROL-FLOW DEBUG
    // =================================================

    output logic       branch_divergent,
    output logic [3:0] branch_taken_mask,
    output logic [3:0] branch_not_taken_mask,

    output logic       reconvergence_event
);


    // =================================================
    // COMPLETE SNACKWELL SM
    // =================================================

    multiwarp_execution_core core (

        .clk                     (clk),
        .reset                   (reset),
        .run_enable              (run_enable),

        .launch_enable           (launch_enable),
        .launch_warp_id          (launch_warp_id),
        .launch_pc               (launch_pc),
        .launch_active_mask      (launch_active_mask),

        .debug_write_enable      (debug_write_enable),
        .debug_warp_id           (debug_warp_id),
        .debug_lane_id           (debug_lane_id),
        .debug_reg_addr          (debug_reg_addr),
        .debug_write_data        (debug_write_data),

        .issue_valid             (issue_valid),
        .issue_accept            (issue_accept),
        .selected_warp_id        (selected_warp_id),

        .selected_pc             (selected_pc),
        .instruction             (instruction),
        .opcode                  (opcode),

        .selected_active_mask    (selected_active_mask),

        .valid_mask              (valid_mask),
        .ready_mask              (ready_mask),
        .halted_mask             (halted_mask),
        .blocked_mask            (blocked_mask),

        .dependency_stall        (dependency_stall),
        .resource_stall          (resource_stall),

        .rs1_data                (rs1_data),
        .rs2_data                (rs2_data),
        .alu_result              (alu_result),

        .shared_bank_conflict    (shared_bank_conflict),
        .shared_conflict_mask    (shared_conflict_mask),

        .branch_divergent        (branch_divergent),
        .branch_taken_mask       (branch_taken_mask),
        .branch_not_taken_mask   (branch_not_taken_mask),

        .reconvergence_event     (reconvergence_event)

    );


endmodule