module simd_lane (
    input  logic        clk,
    input  logic        reset,

    input  logic [5:0]  opcode,

    input  logic [3:0]  rs1_addr,
    input  logic [3:0]  rs2_addr,
    input  logic [3:0]  rd_addr,

    input  logic        write_enable,

    // debug/test preload path
    input  logic        debug_write_enable,
    input  logic [3:0]  debug_write_addr,
    input  logic [31:0] debug_write_data,

    output logic [31:0] debug_rs1_data,
    output logic [31:0] debug_rs2_data,
    output logic [31:0] debug_alu_result
);

    logic [31:0] rs1_data;
    logic [31:0] rs2_data;
    logic [31:0] alu_result;

    logic        reg_write_enable;
    logic [3:0]  reg_write_addr;
    logic [31:0] reg_write_data;

    always_comb begin

        if (debug_write_enable) begin

            reg_write_enable = 1'b1;
            reg_write_addr   = debug_write_addr;
            reg_write_data   = debug_write_data;

        end
        else begin

            reg_write_enable = write_enable;
            reg_write_addr   = rd_addr;
            reg_write_data   = alu_result;

        end

    end

    register_file regfile (
        .clk          (clk),
        .reset        (reset),

        .rs1_addr     (rs1_addr),
        .rs2_addr     (rs2_addr),

        .rs1_data     (rs1_data),
        .rs2_data     (rs2_data),

        .write_enable (reg_write_enable),
        .rd_addr      (reg_write_addr),
        .write_data   (reg_write_data)
    );

    integer_alu alu (
        .opcode    (opcode),
        .operand_a (rs1_data),
        .operand_b (rs2_data),
        .result    (alu_result)
    );

    assign debug_rs1_data   = rs1_data;
    assign debug_rs2_data   = rs2_data;
    assign debug_alu_result = alu_result;

endmodule
