`timescale 1ns/1ps

module tb_multiwarp_register_file;

    logic clk;
    logic reset;

    logic [1:0] warp_id;

    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;

    logic [127:0] rs1_data;
    logic [127:0] rs2_data;

    logic         write_enable;
    logic [3:0]   rd_addr;
    logic [127:0] write_data;
    logic [3:0]   write_mask;

    logic         debug_write_enable;
    logic [1:0]   debug_warp_id;
    logic [1:0]   debug_lane_id;
    logic [3:0]   debug_reg_addr;
    logic [31:0]  debug_write_data;


    // =================================================
    // DUT
    // =================================================

    multiwarp_register_file dut (

        .clk                (clk),
        .reset              (reset),

        .read_warp_id       (warp_id),
        .write_warp_id      (warp_id),

        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),

        .rs1_data           (rs1_data),
        .rs2_data           (rs2_data),

        .write_enable       (write_enable),
        .rd_addr            (rd_addr),
        .write_data         (write_data),
        .write_mask         (write_mask),

        .debug_write_enable (debug_write_enable),
        .debug_warp_id      (debug_warp_id),
        .debug_lane_id      (debug_lane_id),
        .debug_reg_addr     (debug_reg_addr),
        .debug_write_data   (debug_write_data)

    );


    // =================================================
    // CLOCK
    // =================================================

    always #5 clk = ~clk;


    // =================================================
    // WAVEFORM
    // =================================================

    initial begin

        $dumpfile("tb_multiwarp_register_file.vcd");
        $dumpvars(0, tb_multiwarp_register_file);

    end


    // =================================================
    // DEBUG WRITE HELPER
    // =================================================

    task debug_write (
        input logic [1:0]  test_warp,
        input logic [1:0]  test_lane,
        input logic [3:0]  test_reg,
        input logic [31:0] value
    );

        begin

            debug_warp_id      = test_warp;
            debug_lane_id      = test_lane;
            debug_reg_addr     = test_reg;
            debug_write_data   = value;

            debug_write_enable = 1'b1;

            @(posedge clk);
            #1;

            debug_write_enable = 1'b0;

        end

    endtask


    // =================================================
    // CHECK ONE 4-LANE READ
    // =================================================

    task check_rs1 (
        input logic [1:0]  test_warp,
        input logic [3:0]  test_reg,

        input logic [31:0] lane0_expected,
        input logic [31:0] lane1_expected,
        input logic [31:0] lane2_expected,
        input logic [31:0] lane3_expected
    );

        begin

            warp_id  = test_warp;
            rs1_addr = test_reg;

            #1;


            if (rs1_data[31:0] !== lane0_expected) begin

                $display(
                    "FAIL W%0d L0 R%0d = %h expected=%h",
                    test_warp,
                    test_reg,
                    rs1_data[31:0],
                    lane0_expected
                );

                $fatal;

            end


            if (rs1_data[63:32] !== lane1_expected) begin

                $display(
                    "FAIL W%0d L1 R%0d = %h expected=%h",
                    test_warp,
                    test_reg,
                    rs1_data[63:32],
                    lane1_expected
                );

                $fatal;

            end


            if (rs1_data[95:64] !== lane2_expected) begin

                $display(
                    "FAIL W%0d L2 R%0d = %h expected=%h",
                    test_warp,
                    test_reg,
                    rs1_data[95:64],
                    lane2_expected
                );

                $fatal;

            end


            if (rs1_data[127:96] !== lane3_expected) begin

                $display(
                    "FAIL W%0d L3 R%0d = %h expected=%h",
                    test_warp,
                    test_reg,
                    rs1_data[127:96],
                    lane3_expected
                );

                $fatal;

            end


            $display(
                "PASS W%0d R%0d = [%0d, %0d, %0d, %0d]",
                test_warp,
                test_reg,
                rs1_data[31:0],
                rs1_data[63:32],
                rs1_data[95:64],
                rs1_data[127:96]
            );

        end

    endtask


    // =================================================
    // MAIN TEST
    // =================================================

    initial begin

        clk = 0;
        reset = 1;

        warp_id  = 0;

        rs1_addr = 0;
        rs2_addr = 0;

        write_enable = 0;
        rd_addr      = 0;
        write_data   = 0;
        write_mask   = 0;

        debug_write_enable = 0;
        debug_warp_id      = 0;
        debug_lane_id      = 0;
        debug_reg_addr     = 0;
        debug_write_data   = 0;


        $display("");
        $display(
            "=== Snackwell Multi-Warp Register File Test ==="
        );
        $display("");


        // =================================================
        // RESET
        // =================================================

        @(posedge clk);
        #1;

        reset = 0;


        check_rs1(
            2'd0,
            4'd1,

            32'd0,
            32'd0,
            32'd0,
            32'd0
        );


        check_rs1(
            2'd3,
            4'd15,

            32'd0,
            32'd0,
            32'd0,
            32'd0
        );


        $display("PASS: reset cleared register storage");
        $display("");


        // =================================================
        // TEST 1
        //
        // Different values in SAME register,
        // SAME warp, DIFFERENT lanes.
        // =================================================

        debug_write(2'd0, 2'd0, 4'd1, 32'd10);
        debug_write(2'd0, 2'd1, 4'd1, 32'd20);
        debug_write(2'd0, 2'd2, 4'd1, 32'd30);
        debug_write(2'd0, 2'd3, 4'd1, 32'd40);


        check_rs1(
            2'd0,
            4'd1,

            32'd10,
            32'd20,
            32'd30,
            32'd40
        );


        $display(
            "PASS: per-lane register state is independent"
        );
        $display("");


        // =================================================
        // TEST 2
        //
        // SAME register number in DIFFERENT warps.
        // =================================================

        debug_write(2'd1, 2'd0, 4'd1, 32'd100);
        debug_write(2'd1, 2'd1, 4'd1, 32'd200);
        debug_write(2'd1, 2'd2, 4'd1, 32'd300);
        debug_write(2'd1, 2'd3, 4'd1, 32'd400);


        check_rs1(
            2'd1,
            4'd1,

            32'd100,
            32'd200,
            32'd300,
            32'd400
        );


        // Warp 0 must remain untouched

        check_rs1(
            2'd0,
            4'd1,

            32'd10,
            32'd20,
            32'd30,
            32'd40
        );


        $display(
            "PASS: per-warp register state is independent"
        );
        $display("");


        // =================================================
        // TEST 3
        //
        // SIMD writeback into Warp 2 R3
        // =================================================

        warp_id = 2'd2;

        rd_addr = 4'd3;

        write_data = {
            32'd44,
            32'd33,
            32'd22,
            32'd11
        };

        write_mask = 4'b1111;

        write_enable = 1;


        @(posedge clk);
        #1;


        write_enable = 0;


        check_rs1(
            2'd2,
            4'd3,

            32'd11,
            32'd22,
            32'd33,
            32'd44
        );


        $display(
            "PASS: 4-lane SIMD writeback"
        );
        $display("");


        // =================================================
        // TEST 4
        //
        // MASKED WRITEBACK
        //
        // Only lanes 0 and 2 should update.
        // =================================================

        warp_id = 2'd2;

        rd_addr = 4'd3;

        write_data = {
            32'd4000,
            32'd3000,
            32'd2000,
            32'd1000
        };

        write_mask = 4'b0101;

        write_enable = 1;


        @(posedge clk);
        #1;


        write_enable = 0;


        check_rs1(
            2'd2,
            4'd3,

            32'd1000, // lane 0 changed
            32'd22,   // lane 1 preserved
            32'd3000, // lane 2 changed
            32'd44    // lane 3 preserved
        );


        $display(
            "PASS: active-mask style writeback"
        );
        $display("");


        // =================================================
        // TEST 5
        //
        // TWO READ PORTS AT ONCE
        // =================================================

        // Put values into Warp 3 R4 and R7

        debug_write(2'd3, 2'd0, 4'd4, 32'd1);
        debug_write(2'd3, 2'd1, 4'd4, 32'd2);
        debug_write(2'd3, 2'd2, 4'd4, 32'd3);
        debug_write(2'd3, 2'd3, 4'd4, 32'd4);

        debug_write(2'd3, 2'd0, 4'd7, 32'd10);
        debug_write(2'd3, 2'd1, 4'd7, 32'd20);
        debug_write(2'd3, 2'd2, 4'd7, 32'd30);
        debug_write(2'd3, 2'd3, 4'd7, 32'd40);


        warp_id  = 2'd3;

        rs1_addr = 4'd4;
        rs2_addr = 4'd7;

        #1;


        if (
            rs1_data !== {
                32'd4,
                32'd3,
                32'd2,
                32'd1
            }
        ) begin

            $display(
                "FAIL: rs1 simultaneous read"
            );

            $fatal;

        end


        if (
            rs2_data !== {
                32'd40,
                32'd30,
                32'd20,
                32'd10
            }
        ) begin

            $display(
                "FAIL: rs2 simultaneous read"
            );

            $fatal;

        end


        $display(
            "PASS: two 4-lane read ports operate simultaneously"
        );


        // =================================================
        // VERIFY NO CORRUPTION OF OTHER WARPS
        // =================================================

        check_rs1(
            2'd0,
            4'd1,

            32'd10,
            32'd20,
            32'd30,
            32'd40
        );


        check_rs1(
            2'd1,
            4'd1,

            32'd100,
            32'd200,
            32'd300,
            32'd400
        );


        $display(
            "PASS: unrelated warp state preserved"
        );


        // =================================================

        $display("");
        $display(
            "ALL MULTI-WARP REGISTER FILE TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule
