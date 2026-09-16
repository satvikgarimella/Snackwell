module branch_unit (
    input  logic [31:0] current_pc,

    // Four lane values for rs1 and rs2
    input  logic [127:0] operand_a,
    input  logic [127:0] operand_b,

    // Current active lanes in the warp
    input  logic [3:0] active_mask,

    // Already sign-extended immediate
    input  logic [31:0] immediate,

    // ------------------------------------------------
    // BRANCH RESULT
    // ------------------------------------------------

    output logic [3:0] taken_mask,
    output logic [3:0] not_taken_mask,

    output logic       uniform_taken,
    output logic       uniform_not_taken,
    output logic       divergent,

    output logic [31:0] branch_target,
    output logic [31:0] fallthrough_pc
);


    // =================================================
    // PER-LANE CONDITIONS
    // =================================================

    logic lane0_equal;
    logic lane1_equal;
    logic lane2_equal;
    logic lane3_equal;


    assign lane0_equal =
        operand_a[31:0] ==
        operand_b[31:0];

    assign lane1_equal =
        operand_a[63:32] ==
        operand_b[63:32];

    assign lane2_equal =
        operand_a[95:64] ==
        operand_b[95:64];

    assign lane3_equal =
        operand_a[127:96] ==
        operand_b[127:96];


    // =================================================
    // ACTIVE-LANE BRANCH MASKS
    // =================================================

    assign taken_mask[0] =
        active_mask[0] && lane0_equal;

    assign taken_mask[1] =
        active_mask[1] && lane1_equal;

    assign taken_mask[2] =
        active_mask[2] && lane2_equal;

    assign taken_mask[3] =
        active_mask[3] && lane3_equal;


    assign not_taken_mask =
        active_mask &
        ~taken_mask;


    // =================================================
    // BRANCH CLASSIFICATION
    // =================================================

    assign uniform_taken =
        (active_mask != 4'b0000) &&
        (taken_mask == active_mask);


    assign uniform_not_taken =
        (active_mask != 4'b0000) &&
        (taken_mask == 4'b0000);


    assign divergent =
        (taken_mask != 4'b0000) &&
        (not_taken_mask != 4'b0000);


    // =================================================
    // PC CALCULATION
    //
    // Branch immediate is treated as an instruction
    // displacement.
    //
    // target = PC + 4 + (imm << 2)
    // =================================================

    assign fallthrough_pc =
        current_pc + 32'd4;


    assign branch_target =
        current_pc +
        32'd4 +
        (immediate << 2);


endmodule
