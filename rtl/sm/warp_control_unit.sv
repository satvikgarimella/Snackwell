module warp_control_unit (
    input logic clk,
    input logic reset,

    // ------------------------------------------------
    // GLOBAL ISSUE ENABLE
    //
    // 0 = warp state may exist, but scheduler must
    //     not attempt to issue instructions.
    //
    // Used during launch/debug preload.
    // ------------------------------------------------

    input logic issue_enable,

    // ------------------------------------------------
    // WARP STATE UPDATES
    // ------------------------------------------------

    input logic       update_enable,
    input logic [1:0] update_warp_id,

    input logic        update_pc_enable,
    input logic [31:0] update_pc,

    input logic       update_mask_enable,
    input logic [3:0] update_active_mask,

    input logic update_valid_enable,
    input logic update_valid,

    input logic update_ready_enable,
    input logic update_ready,

    input logic update_halted_enable,
    input logic update_halted,

    // ------------------------------------------------
    // CURRENT DECODED INSTRUCTION
    // ------------------------------------------------

    input logic [3:0] rs1_addr,
    input logic [3:0] rs2_addr,
    input logic [3:0] rd_addr,

    input logic uses_rs1,
    input logic uses_rs2,
    input logic writes_rd,
    input logic additional_dependency,

    // ------------------------------------------------
    // EXECUTION RESOURCE STATUS
    // ------------------------------------------------

    input logic execution_unit_ready,

    // ------------------------------------------------
    // WRITEBACK / SCOREBOARD CLEAR
    // ------------------------------------------------

    input logic       clear_busy_enable,
    input logic [1:0] clear_busy_warp_id,
    input logic [3:0] clear_busy_reg,

    // ------------------------------------------------
    // ISSUE OUTPUT
    // ------------------------------------------------

    output logic       issue_valid,
    output logic       issue_accept,
    output logic [1:0] selected_warp_id,

    output logic dependency_stall,
    output logic resource_stall,

    // ------------------------------------------------
    // SELECTED WARP STATE
    // ------------------------------------------------

    output logic [31:0] selected_pc,
    output logic [3:0]  selected_active_mask,

    output logic selected_valid,
    output logic selected_ready,
    output logic selected_halted,

    // ------------------------------------------------
    // STATUS
    // ------------------------------------------------

    output logic [3:0] valid_mask,
    output logic [3:0] ready_mask,
    output logic [3:0] halted_mask,

    output logic [3:0] blocked_mask,
    output logic [3:0] eligible_mask,
    output logic [15:0] selected_warp_busy_mask
);

    // =================================================
    // INTERNAL SIGNALS
    // =================================================

    logic rs1_busy;
    logic rs2_busy;


    // This is the mask actually presented to the
    // scheduler/issue-control logic.
    //
    // The real warp state remains untouched.
    logic [3:0] issue_valid_mask;


    // =================================================
    // ISSUE ENABLE GATING
    // =================================================

    assign issue_valid_mask =
        issue_enable
            ? valid_mask
            : 4'b0000;


    // =================================================
    // WARP STATE TABLE
    // =================================================

    warp_state_table state_table (

        .clk                  (clk),
        .reset                (reset),

        // Scheduler-selected warp
        .read_warp_id         (selected_warp_id),

        .read_pc              (selected_pc),
        .read_active_mask     (selected_active_mask),

        .read_valid           (selected_valid),
        .read_ready           (selected_ready),
        .read_halted          (selected_halted),

        // State update interface
        .update_enable        (update_enable),
        .update_warp_id       (update_warp_id),

        .update_pc_enable     (update_pc_enable),
        .update_pc            (update_pc),

        .update_mask_enable   (update_mask_enable),
        .update_active_mask   (update_active_mask),

        .update_valid_enable  (update_valid_enable),
        .update_valid         (update_valid),

        .update_ready_enable  (update_ready_enable),
        .update_ready         (update_ready),

        .update_halted_enable (update_halted_enable),
        .update_halted        (update_halted),

        // Full architectural state masks
        .valid_mask           (valid_mask),
        .ready_mask           (ready_mask),
        .halted_mask          (halted_mask)

    );


    // =================================================
    // SCHEDULER + SCOREBOARD + ISSUE STAGE
    // =================================================

    issue_control issue_control_unit (

        .clk                     (clk),
        .reset                   (reset),

        // IMPORTANT:
        //
        // Use the gated mask here instead of the
        // architectural valid_mask.
        //
        // run_enable=0 therefore means no candidate
        // warps exist from the scheduler's perspective.
        .valid_mask              (issue_valid_mask),

        .ready_mask              (ready_mask),
        .halted_mask             (halted_mask),

        // Decoded instruction
        .rs1_addr                (rs1_addr),
        .rs2_addr                (rs2_addr),
        .rd_addr                 (rd_addr),

        .uses_rs1                (uses_rs1),
        .uses_rs2                (uses_rs2),
        .writes_rd               (writes_rd),
        .additional_dependency   (additional_dependency),

        // Execution resource
        .execution_unit_ready    (execution_unit_ready),

        // Writeback scoreboard clear
        .clear_busy_enable       (clear_busy_enable),
        .clear_busy_warp_id      (clear_busy_warp_id),
        .clear_busy_reg          (clear_busy_reg),

        // Scheduler / issue outputs
        .issue_valid             (issue_valid),
        .issue_accept            (issue_accept),
        .selected_warp_id        (selected_warp_id),

        .dependency_stall        (dependency_stall),
        .resource_stall          (resource_stall),

        .blocked_mask            (blocked_mask),

        // Debug
        .rs1_busy                (rs1_busy),
        .rs2_busy                (rs2_busy),

        .selected_warp_busy_mask (selected_warp_busy_mask),
        .eligible_mask           (eligible_mask)

    );


endmodule
