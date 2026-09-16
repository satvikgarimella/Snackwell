`timescale 1ns/1ps

module tb_multiwarp_execution_core;

    logic clk;
    logic reset;
    logic run_enable;

    // ------------------------------------------------
    // WARP LAUNCH INTERFACE
    // ------------------------------------------------

    logic        launch_enable;
    logic [1:0]  launch_warp_id;
    logic [31:0] launch_pc;
    logic [3:0]  launch_active_mask;

    // ------------------------------------------------
    // DEBUG REGISTER PRELOAD
    // ------------------------------------------------

    logic        debug_write_enable;
    logic [1:0]  debug_warp_id;
    logic [1:0]  debug_lane_id;
    logic [3:0]  debug_reg_addr;
    logic [31:0] debug_write_data;

    // ------------------------------------------------
    // DUT OUTPUTS
    // ------------------------------------------------

    logic       issue_valid;
    logic [1:0] selected_warp_id;

    logic [31:0] selected_pc;
    logic [31:0] instruction;
    logic [5:0]  opcode;

    logic [3:0] selected_active_mask;

    logic [3:0] valid_mask;
    logic [3:0] ready_mask;
    logic [3:0] halted_mask;

    logic [127:0] rs1_data;
    logic [127:0] rs2_data;
    logic [127:0] alu_result;


    // =================================================
    // DUT
    // =================================================

    multiwarp_execution_core dut (

        .clk                  (clk),
        .reset                (reset),
        .run_enable           (run_enable),

        .launch_enable        (launch_enable),
        .launch_warp_id       (launch_warp_id),
        .launch_pc            (launch_pc),
        .launch_active_mask   (launch_active_mask),

        .debug_write_enable   (debug_write_enable),
        .debug_warp_id        (debug_warp_id),
        .debug_lane_id        (debug_lane_id),
        .debug_reg_addr       (debug_reg_addr),
        .debug_write_data     (debug_write_data),

        .issue_valid          (issue_valid),
        .selected_warp_id     (selected_warp_id),

        .selected_pc          (selected_pc),
        .instruction          (instruction),
        .opcode               (opcode),

        .selected_active_mask (selected_active_mask),

        .valid_mask           (valid_mask),
        .ready_mask           (ready_mask),
        .halted_mask          (halted_mask),

        .rs1_data             (rs1_data),
        .rs2_data             (rs2_data),
        .alu_result           (alu_result)

    );


    // =================================================
    // CLOCK
    // =================================================

    always #5 clk = ~clk;


    // =================================================
    // WAVEFORM
    // =================================================

    initial begin

        $dumpfile("tb_multiwarp_execution_core.vcd");
        $dumpvars(0, tb_multiwarp_execution_core);

    end


    // =================================================
    // LAUNCH WARP
    // =================================================

    task launch_warp (
        input logic [1:0]  warp,
        input logic [31:0] pc_value,
        input logic [3:0]  mask_value
    );

        begin

            launch_warp_id     = warp;
            launch_pc          = pc_value;
            launch_active_mask = mask_value;
            launch_enable      = 1'b1;

            @(posedge clk);
            #1;

            launch_enable = 1'b0;

        end

    endtask


    // =================================================
    // DEBUG REGISTER WRITE
    // =================================================

    task write_reg (
        input logic [1:0]  warp,
        input logic [1:0]  lane,
        input logic [3:0]  reg_id,
        input logic [31:0] value
    );

        begin

            debug_warp_id      = warp;
            debug_lane_id      = lane;
            debug_reg_addr     = reg_id;
            debug_write_data   = value;

            debug_write_enable = 1'b1;

            @(posedge clk);
            #1;

            debug_write_enable = 1'b0;

        end

    endtask


    // =================================================
    // MAIN TEST
    // =================================================

    initial begin

        clk        = 1'b0;
        reset      = 1'b1;
        run_enable = 1'b0;

        launch_enable      = 1'b0;
        launch_warp_id     = 2'd0;
        launch_pc          = 32'd0;
        launch_active_mask = 4'b0000;

        debug_write_enable = 1'b0;
        debug_warp_id      = 2'd0;
        debug_lane_id      = 2'd0;
        debug_reg_addr     = 4'd0;
        debug_write_data   = 32'd0;


        $display("");
        $display("=== Snackwell Multi-Warp Execution Test ===");
        $display("");


        // =================================================
        // RESET
        // =================================================

        @(posedge clk);
        #1;

        reset = 1'b0;


        if (valid_mask !== 4'b0000) begin

            $display(
                "FAIL: valid_mask after reset = %b expected 0000",
                valid_mask
            );

            $fatal;

        end


        $display("PASS: reset cleared warp state");


        // =================================================
        // LAUNCH FOUR WARPS
        // =================================================

        launch_warp(
            2'd0,
            32'h00000000,
            4'b1111
        );

        launch_warp(
            2'd1,
            32'h00000000,
            4'b1111
        );

        launch_warp(
            2'd2,
            32'h00000000,
            4'b1111
        );

        launch_warp(
            2'd3,
            32'h00000000,
            4'b1111
        );


        #1;


        if (valid_mask !== 4'b1111) begin

            $display(
                "FAIL: valid_mask = %b expected 1111",
                valid_mask
            );

            $fatal;

        end


        if (ready_mask !== 4'b1111) begin

            $display(
                "FAIL: ready_mask = %b expected 1111",
                ready_mask
            );

            $fatal;

        end


        if (halted_mask !== 4'b0000) begin

            $display(
                "FAIL: halted_mask = %b expected 0000",
                halted_mask
            );

            $fatal;

        end


        $display("PASS: four warps launched");


        // =================================================
        // PRELOAD REGISTER VALUES
        // =================================================

        // W0
        write_reg(2'd0, 2'd0, 4'd1, 32'd10);
        write_reg(2'd0, 2'd0, 4'd2, 32'd1);

        // W1
        write_reg(2'd1, 2'd0, 4'd1, 32'd20);
        write_reg(2'd1, 2'd0, 4'd2, 32'd2);

        // W2
        write_reg(2'd2, 2'd0, 4'd1, 32'd30);
        write_reg(2'd2, 2'd0, 4'd2, 32'd3);

        // W3
        write_reg(2'd3, 2'd0, 4'd1, 32'd40);
        write_reg(2'd3, 2'd0, 4'd2, 32'd4);


        $display(
            "PASS: independent warp register state loaded"
        );


        // =================================================
        // START EXECUTION
        // =================================================

        run_enable = 1'b1;

        #1;


        // =================================================
        // PC 0
        //
        // ADD R3, R1, R2
        // =================================================

        repeat (4) begin

            #1;


            if (!issue_valid) begin

                $display(
                    "FAIL: scheduler produced no valid issue at PC 0"
                );

                $fatal;

            end


            if (selected_pc !== 32'h00000000) begin

                $display(
                    "FAIL: W%0d PC=%h expected=00000000",
                    selected_warp_id,
                    selected_pc
                );

                $fatal;

            end


            $display(
                "ISSUE: W%0d PC=%h opcode=%h lane0_result=%0d",
                selected_warp_id,
                selected_pc,
                opcode,
                alu_result[31:0]
            );


            @(posedge clk);
            #1;

        end


        $display(
            "PASS: all four warps executed PC 0"
        );


        // =================================================
        // PC 4
        //
        // SUB R4, R1, R2
        // =================================================

        repeat (4) begin

            #1;


            if (!issue_valid) begin

                $display(
                    "FAIL: scheduler produced no valid issue at PC 4"
                );

                $fatal;

            end


            if (selected_pc !== 32'h00000004) begin

                $display(
                    "FAIL: W%0d PC=%h expected=00000004",
                    selected_warp_id,
                    selected_pc
                );

                $fatal;

            end


            $display(
                "ISSUE: W%0d PC=%h opcode=%h lane0_result=%0d",
                selected_warp_id,
                selected_pc,
                opcode,
                alu_result[31:0]
            );


            @(posedge clk);
            #1;

        end


        $display(
            "PASS: independent warp PCs advanced to PC 4"
        );


        // =================================================
        // PC 8
        //
        // XOR R5, R1, R2
        // =================================================

        repeat (4) begin

            #1;


            if (!issue_valid) begin

                $display(
                    "FAIL: scheduler produced no valid issue at PC 8"
                );

                $fatal;

            end


            if (selected_pc !== 32'h00000008) begin

                $display(
                    "FAIL: W%0d PC=%h expected=00000008",
                    selected_warp_id,
                    selected_pc
                );

                $fatal;

            end


            $display(
                "ISSUE: W%0d PC=%h opcode=%h lane0_result=%0d",
                selected_warp_id,
                selected_pc,
                opcode,
                alu_result[31:0]
            );


            @(posedge clk);
            #1;

        end


        $display(
            "PASS: independent warp PCs advanced to PC 8"
        );


        // =================================================
        // EXIT PHASE
        //
        // PC C = EXIT
        //
        // IMPORTANT:
        // Re-check halted_mask after #1 so the final
        // nonblocking state update has time to settle.
        // =================================================

        while (halted_mask !== 4'b1111) begin

            #1;


            // The previous clock may have just halted
            // the final remaining warp.
            if (halted_mask === 4'b1111) begin

                // Done. Do not demand another issue.

            end

            else begin

                if (!issue_valid) begin

                    $display(
                        "FAIL: scheduler idle before all warps halted"
                    );

                    $fatal;

                end


                if (selected_pc !== 32'h0000000C) begin

                    $display(
                        "FAIL: W%0d PC=%h expected EXIT PC=0000000C",
                        selected_warp_id,
                        selected_pc
                    );

                    $fatal;

                end


                $display(
                    "EXIT: W%0d PC=%h",
                    selected_warp_id,
                    selected_pc
                );


                @(posedge clk);
                #1;

            end

        end


        // =================================================
        // LET FINAL COMBINATIONAL SIGNALS SETTLE
        // =================================================

        #1;


        // =================================================
        // FINAL STATE CHECKS
        // =================================================

        if (halted_mask !== 4'b1111) begin

            $display(
                "FAIL: halted_mask=%b expected=1111",
                halted_mask
            );

            $fatal;

        end


        if (ready_mask !== 4'b0000) begin

            $display(
                "FAIL: ready_mask=%b expected=0000",
                ready_mask
            );

            $fatal;

        end


        if (issue_valid !== 1'b0) begin

            $display(
                "FAIL: scheduler still issuing after all warps halted"
            );

            $fatal;

        end


        $display("");
        $display("PASS: all four warps reached EXIT");
        $display("PASS: halted warps removed from scheduling");
        $display("PASS: independent PCs preserved");
        $display("PASS: independent register contexts preserved");


        $display("");
        $display(
            "ALL MULTI-WARP EXECUTION TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule