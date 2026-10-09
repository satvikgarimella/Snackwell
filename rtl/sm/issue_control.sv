module issue_control (
    input logic clk,
    input logic reset,

    // ------------------------------------------------
    // WARP STATUS
    // ------------------------------------------------

    input logic [3:0] valid_mask,
    input logic [3:0] ready_mask,
    input logic [3:0] halted_mask,

    // ------------------------------------------------
    // CURRENT CANDIDATE INSTRUCTION FIELDS
    //
    // These are assumed to correspond to the warp
    // selected by the scheduler.
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

    output logic [3:0] blocked_mask,

    // ------------------------------------------------
    // DEBUG
    // ------------------------------------------------

    output logic       rs1_busy,
    output logic       rs2_busy,
    output logic [15:0] selected_warp_busy_mask,
    output logic [3:0] eligible_mask
);

    // =================================================
    // SCOREBOARD MARK-BUSY SIGNALS
    // =================================================

    logic       mark_busy_enable;
    logic [1:0] mark_busy_warp_id;
    logic [3:0] mark_busy_reg;

    // =================================================
    // IMPORTANT:
    //
    // blocked_mask is generated from the CURRENT
    // candidate warp's dependency/resource condition.
    //
    // Since the scheduler itself consumes blocked_mask,
    // we must avoid creating a combinational loop.
    //
    // Therefore dependency blocking is registered below.
    // =================================================

    logic [3:0] registered_blocked_mask;
    logic [3:0] resource_blocked_mask;

    assign blocked_mask = registered_blocked_mask | resource_blocked_mask;


    // =================================================
    // WARP SCHEDULER
    // =================================================

    warp_scheduler scheduler (

        .clk              (clk),
        .reset            (reset),

        .valid_mask       (valid_mask),
        .ready_mask       (ready_mask),
        .halted_mask      (halted_mask),

        .blocked_mask     (blocked_mask),

        .issue_accept     (issue_accept),

        .issue_valid      (issue_valid),
        .selected_warp_id (selected_warp_id),
        .eligible_mask    (eligible_mask)

    );


    // =================================================
    // SCOREBOARD
    // =================================================

    scoreboard sb (

        .clk                     (clk),
        .reset                   (reset),

        .mark_busy_enable        (mark_busy_enable),
        .mark_busy_warp_id       (mark_busy_warp_id),
        .mark_busy_reg           (mark_busy_reg),

        .clear_busy_enable       (clear_busy_enable),
        .clear_busy_warp_id      (clear_busy_warp_id),
        .clear_busy_reg          (clear_busy_reg),

        .check_warp_id           (selected_warp_id),
        .check_rs1               (rs1_addr),
        .check_rs2               (rs2_addr),

        .rs1_busy                (rs1_busy),
        .rs2_busy                (rs2_busy),
        .sources_ready           (),

        .selected_warp_busy_mask (selected_warp_busy_mask)

    );


    // =================================================
    // ISSUE STAGE
    // =================================================

    issue_stage issue (

        .candidate_valid      (issue_valid),
        .candidate_warp_id    (selected_warp_id),

        .rs1_addr             (rs1_addr),
        .rs2_addr             (rs2_addr),
        .rd_addr              (rd_addr),

        .uses_rs1             (uses_rs1),
        .uses_rs2             (uses_rs2),
        .writes_rd            (writes_rd),

        .rs1_busy             (rs1_busy),
        .rs2_busy             (rs2_busy),
        .rd_busy              (selected_warp_busy_mask[rd_addr]),
        .additional_dependency (additional_dependency),

        .execution_unit_ready (execution_unit_ready),

        .issue_accept         (issue_accept),

        .dependency_stall     (dependency_stall),
        .resource_stall       (resource_stall),

        .mark_busy_enable     (mark_busy_enable),
        .mark_busy_warp_id    (mark_busy_warp_id),
        .mark_busy_reg        (mark_busy_reg)

    );


    // =================================================
    // BLOCKED MASK UPDATE
    //
    // A dependency-stalled warp waits for a writeback.
    //
    // Resource-stalled warps retry after one skipped cycle.
    //
    // This is intentionally simple for now:
    // separate dependency and resource block bits per warp.
    // =================================================

    always_ff @(posedge clk) begin

        if (reset) begin

            registered_blocked_mask <= 4'b0000;
            resource_blocked_mask <= 4'b0000;

        end

        else begin

            // Resource waits need no writeback from this warp to recover.
            // Skip the candidate for one cycle, then retry with fresh status.
            resource_blocked_mask <= 4'b0000;
            if (resource_stall)
                resource_blocked_mask[selected_warp_id] <= 1'b1;

            // -----------------------------------------
            // Register dependency failure
            // -----------------------------------------

            if (
                dependency_stall
            ) begin

                registered_blocked_mask[selected_warp_id]
                    <= 1'b1;

            end


            // -----------------------------------------
            // Writeback can unblock the warp
            //
            // For this stage, clearing any completed
            // destination for that warp makes it
            // schedulable again.
            // -----------------------------------------

            if (clear_busy_enable) begin

                registered_blocked_mask[clear_busy_warp_id]
                    <= 1'b0;

            end

        end

    end


endmodule
