module simd_execution_unit (
    input logic clk,
    input logic reset,

    // Shared instruction/control
    input logic [5:0] opcode,

    input logic [3:0] rs1_addr,
    input logic [3:0] rs2_addr,
    input logic [3:0] rd_addr,

    input logic write_enable,

    // Debug preload path
    input logic        debug_write_enable,
    input logic [1:0]  debug_lane_select,
    input logic [3:0]  debug_write_addr,
    input logic [31:0] debug_write_data,

    // Flattened debug outputs:
    // [31:0]    = lane 0
    // [63:32]   = lane 1
    // [95:64]   = lane 2
    // [127:96]  = lane 3
    output logic [127:0] debug_rs1_data,
    output logic [127:0] debug_rs2_data,
    output logic [127:0] debug_alu_result
);

    logic [31:0] lane0_rs1;
    logic [31:0] lane1_rs1;
    logic [31:0] lane2_rs1;
    logic [31:0] lane3_rs1;

    logic [31:0] lane0_rs2;
    logic [31:0] lane1_rs2;
    logic [31:0] lane2_rs2;
    logic [31:0] lane3_rs2;

    logic [31:0] lane0_result;
    logic [31:0] lane1_result;
    logic [31:0] lane2_result;
    logic [31:0] lane3_result;
    // Lane 0

    simd_lane lane0 (
        .clk                (clk),
        .reset              (reset),

        .opcode             (opcode),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),
        .rd_addr            (rd_addr),

        .write_enable       (write_enable),

        .debug_write_enable (
            debug_write_enable &&
            (debug_lane_select == 2'd0)
        ),

        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .debug_rs1_data     (lane0_rs1),
        .debug_rs2_data     (lane0_rs2),
        .debug_alu_result   (lane0_result)
    );
    // Lane 1

    simd_lane lane1 (
        .clk                (clk),
        .reset              (reset),

        .opcode             (opcode),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),
        .rd_addr            (rd_addr),

        .write_enable       (write_enable),

        .debug_write_enable (
            debug_write_enable &&
            (debug_lane_select == 2'd1)
        ),

        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .debug_rs1_data     (lane1_rs1),
        .debug_rs2_data     (lane1_rs2),
        .debug_alu_result   (lane1_result)
    );
    // Lane 2

    simd_lane lane2 (
        .clk                (clk),
        .reset              (reset),

        .opcode             (opcode),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),
        .rd_addr            (rd_addr),

        .write_enable       (write_enable),

        .debug_write_enable (
            debug_write_enable &&
            (debug_lane_select == 2'd2)
        ),

        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .debug_rs1_data     (lane2_rs1),
        .debug_rs2_data     (lane2_rs2),
        .debug_alu_result   (lane2_result)
    );
    // Lane 3

    simd_lane lane3 (
        .clk                (clk),
        .reset              (reset),

        .opcode             (opcode),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),
        .rd_addr            (rd_addr),

        .write_enable       (write_enable),

        .debug_write_enable (
            debug_write_enable &&
            (debug_lane_select == 2'd3)
        ),

        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .debug_rs1_data     (lane3_rs1),
        .debug_rs2_data     (lane3_rs2),
        .debug_alu_result   (lane3_result)
    );

    // Pack the four lane outputs onto wider buses.

    always_comb begin

        debug_rs1_data = {
            lane3_rs1,
            lane2_rs1,
            lane1_rs1,
            lane0_rs1
        };

        debug_rs2_data = {
            lane3_rs2,
            lane2_rs2,
            lane1_rs2,
            lane0_rs2
        };

        debug_alu_result = {
            lane3_result,
            lane2_result,
            lane1_result,
            lane0_result
        };

    end

endmodule
