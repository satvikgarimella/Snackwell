module instruction_memory (
    input  logic [31:0] pc,
    output logic [31:0] instruction
);

    import snackwell_pkg::*;

    logic [31:0] memory [0:15];

    initial begin

        // ADD R3, R1, R2
        memory[0] = {
            OP_ADD,
            4'd3,
            4'd1,
            4'd2,
            14'd0
        };

        // SUB R4, R1, R2
        memory[1] = {
            OP_SUB,
            4'd4,
            4'd1,
            4'd2,
            14'd0
        };

        // XOR R5, R1, R2
        memory[2] = {
            OP_XOR,
            4'd5,
            4'd1,
            4'd2,
            14'd0
        };

        // EXIT
        memory[3] = {
            OP_EXIT,
            4'd0,
            4'd0,
            4'd0,
            14'd0
        };

        // Remaining locations contain NOP
        for (integer i = 4; i < 16; i = i + 1) begin
            memory[i] = 32'd0;
        end

    end

    assign instruction = memory[pc[5:2]];

endmodule