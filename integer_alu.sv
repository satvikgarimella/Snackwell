module integer_alu (
    input  logic [5:0]  opcode,
    input  logic [31:0] operand_a,
    input  logic [31:0] operand_b,

    output logic [31:0] result
);

    import snackwell_pkg::*;

    always_comb begin

        case (opcode)

            OP_ADD: result = operand_a + operand_b;
            OP_SUB: result = operand_a - operand_b;

            OP_AND: result = operand_a & operand_b;
            OP_OR:  result = operand_a | operand_b;
            OP_XOR: result = operand_a ^ operand_b;

            default: result = 32'b0;

        endcase

    end

endmodule