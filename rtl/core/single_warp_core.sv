module single_warp_core (
    input  logic         clk,
    input  logic         reset,

    // Controls whether the core is actually executing
    input  logic         run_enable,

    // Debug preload interface
    input  logic         debug_write_enable,
    input  logic [1:0]   debug_lane_select,
    input  logic [3:0]   debug_write_addr,
    input  logic [31:0]  debug_write_data,

    // Status/debug outputs
    output logic [31:0]  pc,
    output logic [31:0]  instruction,
    output logic [5:0]   opcode,

    output logic         halted,

    output logic [127:0] debug_rs1_data,
    output logic [127:0] debug_rs2_data,
    output logic [127:0] debug_alu_result
);

    import snackwell_pkg::*;


    // -----------------------------------------
    // Decoded instruction fields
    // -----------------------------------------

    logic [3:0] rd_addr;
    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;

    logic [13:0] immediate;
    logic [31:0] immediate_extended;


    // -----------------------------------------
    // Internal control
    // -----------------------------------------

    logic pc_enable;
    logic write_enable;


    // -----------------------------------------
    // HALT CONTROL
    // -----------------------------------------

    always_comb begin

        halted = (opcode == OP_EXIT);

        // PC only advances while the core is running
        // and has not reached EXIT.
        pc_enable = run_enable && !halted;

    end


    // -----------------------------------------
    // REGISTER WRITEBACK CONTROL
    // -----------------------------------------

    always_comb begin

        // Default: don't write anything
        write_enable = 1'b0;

        // Normal instruction execution is disabled
        // while the core is in preload/configuration mode.
        if (run_enable) begin

            case (opcode)

                OP_ADD,
                OP_SUB,
                OP_AND,
                OP_OR,
                OP_XOR:
                    write_enable = 1'b1;

                default:
                    write_enable = 1'b0;

            endcase

        end

    end


    // -----------------------------------------
    // INSTRUCTION FRONTEND
    //
    // PC
    //  ↓
    // instruction memory
    //  ↓
    // decoder
    // -----------------------------------------

    instruction_frontend frontend (
        .clk                (clk),
        .reset              (reset),
        .enable             (pc_enable),

        .pc                 (pc),
        .instruction        (instruction),

        .opcode             (opcode),

        .rd_addr            (rd_addr),
        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),

        .immediate          (immediate),
        .immediate_extended (immediate_extended)
    );


    // -----------------------------------------
    // 4-LANE SIMD EXECUTION
    // -----------------------------------------

    simd_execution_unit execution (
        .clk                (clk),
        .reset              (reset),

        .opcode             (opcode),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),
        .rd_addr            (rd_addr),

        .write_enable       (write_enable),

        .debug_write_enable (debug_write_enable),
        .debug_lane_select  (debug_lane_select),
        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .debug_rs1_data     (debug_rs1_data),
        .debug_rs2_data     (debug_rs2_data),
        .debug_alu_result   (debug_alu_result)
    );

endmodule