`timescale 1ns/1ps

module tb_warp_scheduler;

    logic clk;
    logic reset;

    logic [3:0] valid_mask;
    logic [3:0] ready_mask;
    logic [3:0] halted_mask;

    logic       issue_valid;
    logic [1:0] selected_warp_id;
    logic [3:0] eligible_mask;


    warp_scheduler dut (
        .clk              (clk),
        .reset            (reset),

        .valid_mask       (valid_mask),
        .ready_mask       (ready_mask),
        .halted_mask      (halted_mask),

        .issue_valid      (issue_valid),
        .selected_warp_id (selected_warp_id),
        .eligible_mask    (eligible_mask)
    );


    // -----------------------------------------
    // CLOCK
    // -----------------------------------------

    always #5 clk = ~clk;


    // -----------------------------------------
    // WAVEFORM
    // -----------------------------------------

    initial begin
        $dumpfile("tb_warp_scheduler.vcd");
        $dumpvars(0, tb_warp_scheduler);
    end


    // -----------------------------------------
    // CHECK CURRENT SCHEDULER DECISION
    // -----------------------------------------

    task check_selected (
        input logic       expected_valid,
        input logic [1:0] expected_warp
    );

        begin

            #1;

            if (issue_valid !== expected_valid) begin

                $display(
                    "FAIL issue_valid=%b expected=%b",
                    issue_valid,
                    expected_valid
                );

                $fatal;

            end


            if (
                expected_valid &&
                selected_warp_id !== expected_warp
            ) begin

                $display(
                    "FAIL selected warp=%0d expected=%0d",
                    selected_warp_id,
                    expected_warp
                );

                $fatal;

            end


            if (expected_valid) begin

                $display(
                    "PASS: selected Warp %0d",
                    selected_warp_id
                );

            end

            else begin

                $display(
                    "PASS: no eligible warp"
                );

            end

        end

    endtask


    // =========================================
    // MAIN TEST
    // =========================================

    initial begin

        clk = 0;
        reset = 1;

        valid_mask  = 4'b0000;
        ready_mask  = 4'b0000;
        halted_mask = 4'b0000;


        $display("");
        $display("=== Snackwell Warp Scheduler Test ===");
        $display("");


        // =====================================
        // RESET
        // =====================================

        @(posedge clk);
        #1;

        reset = 0;


        // =====================================
        // TEST 1
        // ALL WARPS READY
        //
        // W0 -> W1 -> W2 -> W3 -> W0
        // =====================================

        valid_mask  = 4'b1111;
        ready_mask  = 4'b1111;
        halted_mask = 4'b0000;


        check_selected(1'b1, 2'd0);

        @(posedge clk);
        check_selected(1'b1, 2'd1);

        @(posedge clk);
        check_selected(1'b1, 2'd2);

        @(posedge clk);
        check_selected(1'b1, 2'd3);

        @(posedge clk);
        check_selected(1'b1, 2'd0);


        $display("");
        $display("PASS: full round-robin rotation");
        $display("");


        // =====================================
        // TEST 2
        // WARP 2 STALLED
        //
        // ready = 1011
        //
        // W1 -> W3 -> W0
        // =====================================

        ready_mask = 4'b1011;


        @(posedge clk);
        check_selected(1'b1, 2'd1);

        @(posedge clk);
        check_selected(1'b1, 2'd3);

        @(posedge clk);
        check_selected(1'b1, 2'd0);


        $display("");
        $display("PASS: stalled warp skipped");
        $display("");


        // =====================================
        // TEST 3
        // WARP 3 HALTED
        // =====================================

        ready_mask  = 4'b1111;
        halted_mask = 4'b1000;


        @(posedge clk);
        check_selected(1'b1, 2'd1);

        @(posedge clk);
        check_selected(1'b1, 2'd2);

        @(posedge clk);
        check_selected(1'b1, 2'd0);


        $display("");
        $display("PASS: halted warp skipped");
        $display("");


        // =====================================
        // TEST 4
        // ONLY WARP 2 READY
        // =====================================

        ready_mask  = 4'b0100;
        halted_mask = 4'b0000;


        @(posedge clk);
        check_selected(1'b1, 2'd2);

        @(posedge clk);
        check_selected(1'b1, 2'd2);

        @(posedge clk);
        check_selected(1'b1, 2'd2);


        $display("");
        $display(
            "PASS: single eligible warp repeatedly selected"
        );
        $display("");


        // =====================================
        // TEST 5
        // NO READY WARPS
        // =====================================

        ready_mask = 4'b0000;


        @(posedge clk);
        check_selected(1'b0, 2'd0);

        @(posedge clk);
        check_selected(1'b0, 2'd0);


        $display("");
        $display("PASS: scheduler idles correctly");
        $display("");


        // =====================================
        // TEST 6
        // ONLY WARPS 0 AND 1 VALID
        // =====================================

        valid_mask  = 4'b0011;
        ready_mask  = 4'b1111;
        halted_mask = 4'b0000;


        // IMPORTANT:
        //
        // last_issued_warp is still W2.
        //
        // Current combinational scheduler decision
        // must therefore be W0.
        //
        // We check BEFORE advancing the clock.

        check_selected(1'b1, 2'd0);


        // Commit W0
        @(posedge clk);

        // Next must be W1
        check_selected(1'b1, 2'd1);


        // Commit W1
        @(posedge clk);

        // Wrap back to W0
        check_selected(1'b1, 2'd0);


        // Commit W0
        @(posedge clk);

        // And W1 again
        check_selected(1'b1, 2'd1);


        $display("");
        $display("PASS: invalid warps skipped");
        $display("");


        // =====================================

        $display("");
        $display(
            "ALL WARP SCHEDULER TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule