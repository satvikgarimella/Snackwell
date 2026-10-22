`timescale 1ns/1ps

module tb_scoreboard;

    logic clk;
    logic reset;

    logic       mark_busy_enable;
    logic [1:0] mark_busy_warp_id;
    logic [3:0] mark_busy_reg;

    logic       clear_busy_enable;
    logic [1:0] clear_busy_warp_id;
    logic [3:0] clear_busy_reg;

    logic [1:0] check_warp_id;
    logic [3:0] check_rs1;
    logic [3:0] check_rs2;

    logic       rs1_busy;
    logic       rs2_busy;
    logic       sources_ready;

    logic [15:0] selected_warp_busy_mask;

    scoreboard dut (
        .clk                    (clk),
        .reset                  (reset),

        .mark_busy_enable       (mark_busy_enable),
        .mark_busy_warp_id      (mark_busy_warp_id),
        .mark_busy_reg          (mark_busy_reg),

        .clear_busy_enable      (clear_busy_enable),
        .clear_busy_warp_id     (clear_busy_warp_id),
        .clear_busy_reg         (clear_busy_reg),

        .check_warp_id          (check_warp_id),
        .check_rs1              (check_rs1),
        .check_rs2              (check_rs2),

        .rs1_busy               (rs1_busy),
        .rs2_busy               (rs2_busy),
        .sources_ready          (sources_ready),

        .selected_warp_busy_mask(selected_warp_busy_mask)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_scoreboard.vcd");
        $dumpvars(0, tb_scoreboard);
    end

    task mark_busy (
        input logic [1:0] warp,
        input logic [3:0] reg_id
    );

        begin

            mark_busy_warp_id = warp;
            mark_busy_reg     = reg_id;
            mark_busy_enable  = 1'b1;

            @(posedge clk);
            #1;

            mark_busy_enable = 1'b0;

        end

    endtask

    task clear_busy (
        input logic [1:0] warp,
        input logic [3:0] reg_id
    );

        begin

            clear_busy_warp_id = warp;
            clear_busy_reg     = reg_id;
            clear_busy_enable  = 1'b1;

            @(posedge clk);
            #1;

            clear_busy_enable = 1'b0;

        end

    endtask

    task check_sources (
        input logic [1:0] warp,
        input logic [3:0] rs1,
        input logic [3:0] rs2,
        input logic       expected_rs1_busy,
        input logic       expected_rs2_busy
    );

        begin

            check_warp_id = warp;
            check_rs1     = rs1;
            check_rs2     = rs2;

            #1;

            if (rs1_busy !== expected_rs1_busy) begin

                $display(
                    "FAIL: W%0d R%0d rs1_busy=%b expected=%b",
                    warp,
                    rs1,
                    rs1_busy,
                    expected_rs1_busy
                );

                $fatal;

            end

            if (rs2_busy !== expected_rs2_busy) begin

                $display(
                    "FAIL: W%0d R%0d rs2_busy=%b expected=%b",
                    warp,
                    rs2,
                    rs2_busy,
                    expected_rs2_busy
                );

                $fatal;

            end

            if (
                sources_ready !==
                ~(expected_rs1_busy | expected_rs2_busy)
            ) begin

                $display(
                    "FAIL: sources_ready=%b",
                    sources_ready
                );

                $fatal;

            end

            $display(
                "PASS: W%0d sources R%0d/R%0d busy=%b/%b ready=%b",
                warp,
                rs1,
                rs2,
                rs1_busy,
                rs2_busy,
                sources_ready
            );

        end

    endtask

    initial begin

        clk = 0;
        reset = 1;

        mark_busy_enable  = 0;
        mark_busy_warp_id = 0;
        mark_busy_reg     = 0;

        clear_busy_enable  = 0;
        clear_busy_warp_id = 0;
        clear_busy_reg     = 0;

        check_warp_id = 0;
        check_rs1     = 0;
        check_rs2     = 0;

        $display("");
        $display("=== Snackwell Scoreboard Test ===");
        $display("");
        // RESET

        @(posedge clk);
        #1;

        reset = 0;

        check_sources(
            2'd0,
            4'd1,
            4'd2,
            1'b0,
            1'b0
        );

        $display("PASS: reset cleared scoreboard");
        $display("");

        // TEST 1
        // Mark W0 R3 busy

        mark_busy(
            2'd0,
            4'd3
        );

        check_sources(
            2'd0,
            4'd3,
            4'd5,
            1'b1,
            1'b0
        );

        $display("PASS: dependency detected");
        $display("");

        // TEST 2
        // Different warp must remain independent

        check_sources(
            2'd1,
            4'd3,
            4'd5,
            1'b0,
            1'b0
        );

        $display(
            "PASS: scoreboard state independent per warp"
        );
        $display("");

        // TEST 3
        // Multiple busy registers

        mark_busy(
            2'd0,
            4'd7
        );

        check_sources(
            2'd0,
            4'd3,
            4'd7,
            1'b1,
            1'b1
        );

        $display(
            "PASS: multiple dependencies detected"
        );
        $display("");

        // TEST 4
        // Clear R3

        clear_busy(
            2'd0,
            4'd3
        );

        check_sources(
            2'd0,
            4'd3,
            4'd7,
            1'b0,
            1'b1
        );

        $display(
            "PASS: writeback clears dependency"
        );
        $display("");

        // TEST 5
        // Clear R7

        clear_busy(
            2'd0,
            4'd7
        );

        check_sources(
            2'd0,
            4'd3,
            4'd7,
            1'b0,
            1'b0
        );

        $display(
            "PASS: sources become ready after writeback"
        );
        $display("");

        // TEST 6
        // Separate dependencies in multiple warps

        mark_busy(2'd0, 4'd4);
        mark_busy(2'd1, 4'd8);
        mark_busy(2'd2, 4'd12);

        check_sources(
            2'd0,
            4'd4,
            4'd8,
            1'b1,
            1'b0
        );

        check_sources(
            2'd1,
            4'd4,
            4'd8,
            1'b0,
            1'b1
        );

        check_sources(
            2'd2,
            4'd12,
            4'd1,
            1'b1,
            1'b0
        );

        check_sources(
            2'd3,
            4'd4,
            4'd8,
            1'b0,
            1'b0
        );

        $display(
            "PASS: independent dependencies across all warps"
        );
        $display("");

        // TEST 7
        // Same-cycle clear and re-mark same register
        //
        // New instruction wins, so register remains busy.

        clear_busy_warp_id = 2'd0;
        clear_busy_reg     = 4'd4;
        clear_busy_enable  = 1'b1;

        mark_busy_warp_id = 2'd0;
        mark_busy_reg     = 4'd4;
        mark_busy_enable  = 1'b1;

        @(posedge clk);
        #1;

        clear_busy_enable = 1'b0;
        mark_busy_enable  = 1'b0;

        check_sources(
            2'd0,
            4'd4,
            4'd1,
            1'b1,
            1'b0
        );

        $display(
            "PASS: same-cycle writeback + new dependency handled"
        );

        $display("");
        $display(
            "ALL SCOREBOARD TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule
