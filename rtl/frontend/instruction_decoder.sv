module instruction_decoder (

    input  logic [31:0] instruction,

    output logic [5:0]  opcode,

    output logic [3:0]  rd_addr,
    output logic [3:0]  rs1_addr,
    output logic [3:0]  rs2_addr,

    output logic [13:0] immediate,
    output logic [31:0] immediate_extended

);

    assign opcode   = instruction[31:26];

    assign rd_addr  = instruction[25:22];
    assign rs1_addr = instruction[21:18];
    assign rs2_addr = instruction[17:14];

    assign immediate = instruction[13:0];

    // Sign-extend 14-bit immediate to 32 bits
    assign immediate_extended = {
        {18{instruction[13]}},
        instruction[13:0]
    };

endmodule
