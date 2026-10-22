module simd_integer_execute (
    input  logic [5:0]   opcode,

    input  logic [127:0] operand_a,
    input  logic [127:0] operand_b,

    output logic [127:0] result
);

    logic [31:0] lane0_result;
    logic [31:0] lane1_result;
    logic [31:0] lane2_result;
    logic [31:0] lane3_result;

    integer_alu alu0 (
        .opcode    (opcode),
        .operand_a (operand_a[31:0]),
        .operand_b (operand_b[31:0]),
        .result    (lane0_result)
    );

    integer_alu alu1 (
        .opcode    (opcode),
        .operand_a (operand_a[63:32]),
        .operand_b (operand_b[63:32]),
        .result    (lane1_result)
    );

    integer_alu alu2 (
        .opcode    (opcode),
        .operand_a (operand_a[95:64]),
        .operand_b (operand_b[95:64]),
        .result    (lane2_result)
    );

    integer_alu alu3 (
        .opcode    (opcode),
        .operand_a (operand_a[127:96]),
        .operand_b (operand_b[127:96]),
        .result    (lane3_result)
    );

    assign result = {
        lane3_result,
        lane2_result,
        lane1_result,
        lane0_result
    };

endmodule
