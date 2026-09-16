module instruction_frontend (
    input  logic        clk,
    input  logic        reset,
    input  logic        enable,

    output logic [31:0] pc,
    output logic [31:0] instruction,

    output logic [5:0]  opcode,

    output logic [3:0]  rd_addr,
    output logic [3:0]  rs1_addr,
    output logic [3:0]  rs2_addr,

    output logic [13:0] immediate,
    output logic [31:0] immediate_extended
);

    // -----------------------------------------
    // PROGRAM COUNTER
    // -----------------------------------------

    program_counter pc_unit (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .pc     (pc)
    );


    // -----------------------------------------
    // INSTRUCTION MEMORY
    // -----------------------------------------

    instruction_memory imem (
        .pc          (pc),
        .instruction (instruction)
    );


    // -----------------------------------------
    // INSTRUCTION DECODER
    // -----------------------------------------

    instruction_decoder decoder (
        .instruction        (instruction),

        .opcode             (opcode),

        .rd_addr            (rd_addr),
        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),

        .immediate          (immediate),
        .immediate_extended (immediate_extended)
    );

endmodule