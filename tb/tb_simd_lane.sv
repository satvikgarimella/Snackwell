`timescale 1ns/1ps

module tb_simd_lane;

    import snackwell_pkg::*;

    logic clk;
    logic reset;

    logic [5:0] opcode;

    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;
    logic [3:0] rd_addr;

    logic write_enable;

    logic        debug_write_enable;
    logic [3:0]  debug_write_addr;
    logic [31:0] debug_write_data;

    logic [31:0] debug_rs1_data;
    logic [31:0] debug_rs2_data;
    logic [31:0] debug_alu_result;


    simd_lane dut (
        .clk                (clk),
        .reset              (reset),

        .opcode             (opcode),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),
        .rd_addr            (rd_addr),

        .write_enable       (write_enable),

        .debug_write_enable (debug_write_enable),
        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .debug_rs1_data     (debug_rs1_data),
        .debug_rs2_data     (debug_rs2_data),
        .debug_alu_result   (debug_alu_result)
    );


    always #5 clk = ~clk;


    initial begin
        $dumpfile("tb_simd_lane.vcd");
        $dumpvars(0, tb_simd_lane);
    end


    task debug_write(
        input logic [3:0] addr,
        input logic [31:0] data
    );

        begin

            debug_write_enable = 1'b1;
            debug_write_addr   = addr;
            debug_write_data   = data;

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
        debug_write_addr   = 0;
        debug_write_data   = 0;


        $display("");
        $display("=== Snackwell SIMD Lane Test ===");
        $display("");


        // Reset
        @(posedge clk);
        #1;
        reset = 0;


        // Preload R1 = 7
        debug_write(4'd1, 32'd7);

        // Preload R2 = 8
        debug_write(4'd2, 32'd8);


        // Select R1 and R2
        rs1_addr = 4'd1;
        rs2_addr = 4'd2;

        opcode = OP_ADD;

        #1;


        // Verify ALU sees the correct operands
        if (debug_rs1_data !== 32'd7) begin
            $display("FAIL: R1 should be 7");
            $fatal;
        end

        if (debug_rs2_data !== 32'd8) begin
            $display("FAIL: R2 should be 8");
            $fatal;
        end

        if (debug_alu_result !== 32'd15) begin
            $display(
                "FAIL: ALU result=%0d expected=15",
                debug_alu_result
            );
            $fatal;
        end

        $display(
            "PASS: R1=%0d R2=%0d ADD result=%0d",
            debug_rs1_data,
            debug_rs2_data,
            debug_alu_result
        );


        // Write result into R3
        rd_addr = 4'd3;
        write_enable = 1'b1;

        @(posedge clk);
        #1;

        write_enable = 1'b0;


        // Read R3
        rs1_addr = 4'd3;

        #1;

        if (debug_rs1_data !== 32'd15) begin

            $display(
                "FAIL: R3=%0d expected=15",
                debug_rs1_data
            );

            $fatal;

        end


        $display(
            "PASS: R3 contains writeback result = %0d",
            debug_rs1_data
        );


        $display("");
        $display("ALL SIMD LANE TESTS PASSED");
        $display("");

        $finish;

    end

endmodule