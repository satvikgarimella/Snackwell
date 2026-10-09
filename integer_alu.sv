module integer_alu (
    input  logic [5:0]  opcode,
    input  logic [31:0] operand_a,
    input  logic [31:0] operand_b,

    output logic [31:0] result
);

    import snackwell_pkg::*;

    always_comb begin

        case (opcode)

            OP_ADD, OP_ADDI: result = operand_a + operand_b;
            OP_SUB: result = operand_a - operand_b;

            OP_AND: result = operand_a & operand_b;
            OP_OR:  result = operand_a | operand_b;
            OP_XOR: result = operand_a ^ operand_b;
            OP_SHR: result = operand_a >> operand_b[4:0];
            OP_SLTU: result = {31'd0, operand_a < operand_b};
            OP_FLT: begin
                // Compare encodings directly: NaNs are unordered, signed
                // zeros compare equal, and negative magnitudes reverse order.
                if (operand_a[30:0] > 31'h7f800000 || operand_b[30:0] > 31'h7f800000 ||
                    (operand_a[30:0] == 0 && operand_b[30:0] == 0))
                    result = 32'd0;
                else if (operand_a[31] != operand_b[31])
                    result = {31'd0, operand_a[31]};
                else if (operand_a[31])
                    result = {31'd0, operand_a[30:0] > operand_b[30:0]};
                else
                    result = {31'd0, operand_a[30:0] < operand_b[30:0]};
            end

            default: result = 32'b0;

        endcase

    end

endmodule
