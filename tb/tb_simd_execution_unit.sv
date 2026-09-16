`timescale 1ns/1ps

module tb_simd_execution_unit;

    import snackwell_pkg::*;

    logic clk;
    logic reset;

    logic [5:0] opcode;

    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;
    logic [3:0] rd_addr;

    logic write_enable;

    logic        debug_write_enable;
    logic [1:0]  debug_lane_select;
    logic [3:0]  debug_write_addr;
    logic [31:0] debug_write_data;

    logic [127:0] debug_rs1_data;
    logic [127:0] debug_rs2_data;
    logic [127:0] debug_alu_result;


    simd_execution_unit dut (
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


    always #5 clk = ~clk;


    initial begin
        $dumpfile("tb_simd_execution_unit.vcd");
        $dumpvars(0, tb_simd_execution_unit);
    end


    // -----------------------------------------
    // Debug preload helper
    // -----------------------------------------

    task preload_register(
        input logic [1:0]  lane,
        input logic [3:0]  reg_addr,
        input logic [31:0] value
    );

        begin

            debug_lane_select  = lane;
            debug_write_addr   = reg_addr;
            debug_write_data   = value;
            debug_write_enable = 1'b1;

            @(posedge clk);
            #1;

            debug_write_enable = 1'b0;

        end

    endtask


    initial begin

        clk = 0;
        reset = 1;

        opcode = OP_NOP;

        rs1_addr = 0;
        rs2_addr = 0;
        rd_addr  = 0;

        write_enable = 0;

        debug_write_enable = 0;
        debug_lane_select  = 0;
        debug_write_addr   = 0;
        debug_write_data   = 0;


        $display("");
        $display("=== Snackwell 4-Lane SIMD Test ===");
        $display("");


        // -----------------------------------------
        // RESET
        // -----------------------------------------

        @(posedge clk);
        #1;

        reset = 0;


        // -----------------------------------------
        // PRELOAD LANE 0
        //
        // R1 = 7
        // R2 = 8
        // -----------------------------------------

        preload_register(2'd0, 4'd1, 32'd7);
        preload_register(2'd0, 4'd2, 32'd8);


        // -----------------------------------------
        // PRELOAD LANE 1
        //
        // R1 = 7
        // R2 = 5
        // -----------------------------------------

        preload_register(2'd1, 4'd1, 32'd7);
        preload_register(2'd1, 4'd2, 32'd5);


        // -----------------------------------------
        // PRELOAD LANE 2
        //
        // R1 = 20
        // R2 = 30
        // -----------------------------------------

        preload_register(2'd2, 4'd1, 32'd20);
        preload_register(2'd2, 4'd2, 32'd30);


        // -----------------------------------------
        // PRELOAD LANE 3
        //
        // R1 = 100
        // R2 = 1
        // -----------------------------------------

        preload_register(2'd3, 4'd1, 32'd100);
        preload_register(2'd3, 4'd2, 32'd1);


        // -----------------------------------------
        // BROADCAST ADD R3, R1, R2
        // -----------------------------------------

        rs1_addr = 4'd1;
        rs2_addr = 4'd2;
        rd_addr  = 4'd3;

        opcode = OP_ADD;

        #1;


        // -----------------------------------------
        // CHECK THE FOUR PARALLEL ALU RESULTS
        // -----------------------------------------

        if (debug_alu_result[31:0] !== 32'd15) begin
            $display("FAIL Lane 0");
            $fatal;
        end

        if (debug_alu_result[63:32] !== 32'd12) begin
            $display("FAIL Lane 1");
            $fatal;
        end

        if (debug_alu_result[95:64] !== 32'd50) begin
            $display("FAIL Lane 2");
            $fatal;
        end

        if (debug_alu_result[127:96] !== 32'd101) begin
            $display("FAIL Lane 3");
            $fatal;
        end


        $display(
            "PASS ADD results: [%0d, %0d, %0d, %0d]",
            debug_alu_result[31:0],
            debug_alu_result[63:32],
            debug_alu_result[95:64],
            debug_alu_result[127:96]
        );


        // -----------------------------------------
        // WRITE ALL FOUR RESULTS INTO R3
        // -----------------------------------------

        write_enable = 1'b1;

        @(posedge clk);
        #1;

        write_enable = 1'b0;


        // -----------------------------------------
        // READ R3 FROM EVERY LANE
        // -----------------------------------------

        rs1_addr = 4'd3;

        #1;


        if (debug_rs1_data[31:0] !== 32'd15) begin
            $display("FAIL Lane 0 R3");
            $fatal;
        end

        if (debug_rs1_data[63:32] !== 32'd12) begin
            $display("FAIL Lane 1 R3");
            $fatal;
        end

        if (debug_rs1_data[95:64] !== 32'd50) begin
            $display("FAIL Lane 2 R3");
            $fatal;
        end

        if (debug_rs1_data[127:96] !== 32'd101) begin
            $display("FAIL Lane 3 R3");
            $fatal;
        end


        $display(
            "PASS R3 writeback: [%0d, %0d, %0d, %0d]",
            debug_rs1_data[31:0],
            debug_rs1_data[63:32],
            debug_rs1_data[95:64],
            debug_rs1_data[127:96]
        );


        // -----------------------------------------
        // SECOND INTEGRATION TEST: SUB R4, R1, R2
        // -----------------------------------------

        rs1_addr = 4'd1;
        rs2_addr = 4'd2;
        rd_addr  = 4'd4;

        opcode = OP_SUB;

        #1;


        if (debug_alu_result[31:0] !== 32'hFFFFFFFF) begin
            $display("FAIL Lane 0 SUB");
            $fatal;
        end

        if (debug_alu_result[63:32] !== 32'd2) begin
            $display("FAIL Lane 1 SUB");
            $fatal;
        end

        if (debug_alu_result[95:64] !== 32'hFFFFFFF6) begin
            $display("FAIL Lane 2 SUB");
            $fatal;
        end

        if (debug_alu_result[127:96] !== 32'd99) begin
            $display("FAIL Lane 3 SUB");
            $fatal;
        end


        $display(
            "PASS SUB broadcast across all lanes"
        );


        $display("");
        $display("ALL 4-LANE SIMD TESTS PASSED");
        $display("");

        $finish;

    end

endmodule