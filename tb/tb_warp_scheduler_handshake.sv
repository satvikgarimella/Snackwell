`timescale 1ns/1ps

module tb_warp_scheduler_handshake;

    logic clk;
    logic reset;

    logic [3:0] valid_mask;
    logic [3:0] ready_mask;
    logic [3:0] halted_mask;
    logic [3:0] blocked_mask;

    logic issue_accept;

    logic       issue_valid;
    logic [1:0] selected_warp_id;
    logic [3:0] eligible_mask;


    warp_scheduler dut (

        .clk              (clk),
        .reset            (reset),

        .valid_mask       (valid_mask),
        .ready_mask       (ready_mask),
        .halted_mask      (halted_mask),
        .blocked_mask     (blocked_mask),

        .issue_accept     (issue_accept),

        .issue_valid      (issue_valid),
        .selected_warp_id (selected_warp_id),
        .eligible_mask    (eligible_mask)

    );


    always #5 clk = ~clk;


    initial begin

        $dumpfile("tb_warp_scheduler_handshake.vcd");
        $dumpvars(0, tb_warp_scheduler_handshake);

    end


    task expect_warp (
        input logic       expected_valid,
        input logic [1:0] expected_warp
    );

        begin

            #1;

            if (issue_valid !== expected_valid) begin

                $display(
                    "FAIL: issue_valid=%b expected=%b",
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
                    "FAIL: selected W%0d expected W%0d",
                    selected_warp_id,
                    expected_warp
                );

                $fatal;

            end


            if (expected_valid) begin

                $display(
                    "PASS: selected W%0d",
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


    task accept_issue;

        begin

            issue_accept = 1'b1;

            @(posedge clk);
            #1;

            issue_accept = 1'b0;

        end

    endtask


    initial begin

        clk = 0;
        reset = 1;

        valid_mask   = 4'b0000;
        ready_mask   = 4'b0000;
        halted_mask  = 4'b0000;
        blocked_mask = 4'b0000;

        issue_accept = 0;


        $display("");
        $display(
            "=== Snackwell Scheduler Handshake Test ==="
        );
        $display("");


        // =================================================
        // RESET
        // =================================================

        @(posedge clk);
        #1;

        reset = 0;


        // =================================================
        // ALL FOUR READY
        // =================================================

        valid_mask = 4'b1111;
        ready_mask = 4'b1111;


        expect_warp(
            1'b1,
            2'd0
        );


        // =================================================
        // NO ACCEPT
        //
        // Scheduler must HOLD W0.
        // =================================================

        @(posedge clk);
        #1;

        expect_warp(
            1'b1,
            2'd0
        );


        @(posedge clk);
        #1;

        expect_warp(
            1'b1,
            2'd0
        );


        $display(
            "PASS: scheduler holds candidate during backpressure"
        );


        // =================================================
        // ACCEPT W0
        // =================================================

        accept_issue();


        expect_warp(
            1'b1,
            2'd1
        );


        $display(
            "PASS: scheduler advances only after issue_accept"
        );


        // =================================================
        // ACCEPT W1
        // =================================================

        accept_issue();


        expect_warp(
            1'b1,
            2'd2
        );


        // =================================================
        // BLOCK W2
        // =================================================

        blocked_mask = 4'b0100;

        #1;


        expect_warp(
            1'b1,
            2'd3
        );


        $display(
            "PASS: blocked warp skipped"
        );


        // =================================================
        // ACCEPT W3
        // =================================================

        accept_issue();


        expect_warp(
            1'b1,
            2'd0
        );


        // =================================================
        // BLOCK W0 + W2
        //
        // rr_pointer is W0.
        // W0 blocked -> W1 selected.
        //
        // IMPORTANT:
        // W1 is selected but NOT accepted.
        // Therefore pointer remains W0.
        // =================================================

        blocked_mask = 4'b0101;


        expect_warp(
            1'b1,
            2'd1
        );


        $display(
            "PASS: multiple blocked warps skipped"
        );


        // =================================================
        // BLOCK EVERYTHING
        // =================================================

        blocked_mask = 4'b1111;


        expect_warp(
            1'b0,
            2'd0
        );


        @(posedge clk);
        #1;


        expect_warp(
            1'b0,
            2'd0
        );


        $display(
            "PASS: scheduler idles when all warps blocked"
        );


        // =================================================
        // UNBLOCK
        //
        // W1 was only selected, never accepted.
        //
        // Therefore rr_pointer is STILL W0.
        //
        // Scheduler must select W0 after unblock.
        // =================================================

        blocked_mask = 4'b0000;

        #1;


        expect_warp(
            1'b1,
            2'd0
        );


        $display(
            "PASS: round-robin pointer preserved without issue_accept"
        );


        $display("");
        $display(
            "ALL SCHEDULER HANDSHAKE TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule