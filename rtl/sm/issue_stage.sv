module issue_stage (
    // ------------------------------------------------
    // CANDIDATE INSTRUCTION
    // ------------------------------------------------

    input logic       candidate_valid,
    input logic [1:0] candidate_warp_id,

    input logic [3:0] rs1_addr,
    input logic [3:0] rs2_addr,
    input logic [3:0] rd_addr,

    // ------------------------------------------------
    // INSTRUCTION BEHAVIOR
    //
    // Decoder/control logic will eventually generate
    // these signals.
    // ------------------------------------------------

    input logic uses_rs1,
    input logic uses_rs2,
    input logic writes_rd,

    // ------------------------------------------------
    // SCOREBOARD STATUS
    // ------------------------------------------------

    input logic rs1_busy,
    input logic rs2_busy,
    input logic rd_busy,
    // Dependencies spanning multiple registers (for example a tensor tile).
    input logic additional_dependency,

    // ------------------------------------------------
    // EXECUTION RESOURCE STATUS
    //
    // For now the integer unit will generally be ready.
    // Later FP/LSU/Tensor may deassert this.
    // ------------------------------------------------

    input logic execution_unit_ready,

    // ------------------------------------------------
    // ISSUE RESULT
    // ------------------------------------------------

    output logic issue_accept,

    output logic dependency_stall,
    output logic resource_stall,

    // ------------------------------------------------
    // SCOREBOARD MARK-BUSY REQUEST
    // ------------------------------------------------

    output logic       mark_busy_enable,
    output logic [1:0] mark_busy_warp_id,
    output logic [3:0] mark_busy_reg
);


    logic register_dependency;


    // =================================================
    // RAW / WAW DEPENDENCY CHECK
    // =================================================

    assign register_dependency =
        additional_dependency ||
        (uses_rs1 && rs1_busy) ||
        (uses_rs2 && rs2_busy) ||
        (writes_rd && rd_busy);


    // =================================================
    // STALL REASONS
    // =================================================

    assign dependency_stall =
        candidate_valid &&
        register_dependency;


    assign resource_stall =
        candidate_valid &&
        !register_dependency &&
        !execution_unit_ready;


    // =================================================
    // ISSUE ACCEPT
    // =================================================

    assign issue_accept =
        candidate_valid &&
        !register_dependency &&
        execution_unit_ready;


    // =================================================
    // SCOREBOARD DESTINATION ALLOCATION
    //
    // Only mark destination busy if the instruction
    // actually issued.
    // =================================================

    assign mark_busy_enable =
        issue_accept &&
        writes_rd;


    assign mark_busy_warp_id =
        candidate_warp_id;


    assign mark_busy_reg =
        rd_addr;


endmodule
