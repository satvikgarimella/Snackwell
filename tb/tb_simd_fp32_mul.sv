`timescale 1ns/1ps

module tb_simd_fp32_mul;

    logic clk;
    logic rst;

    logic start;

    logic [127:0] operand_a;
    logic [127:0] operand_b;

    logic [3:0] active_mask;

    logic busy;
    logic done;

    logic [127:0] result;

    simd_fp32_mul dut (
        .clk         (clk),
        .rst         (rst),
        .start       (start),

        .operand_a   (operand_a),
        .operand_b   (operand_b),

        .active_mask (active_mask),

        .busy        (busy),
        .done        (done),

        .result      (result)
    );

    always #5 clk = ~clk;

    initial begin

        $dumpfile("tb_simd_fp32_mul.vcd");
        $dumpvars(0, tb_simd_fp32_mul);

        clk = 0;
        rst = 1;

        start = 0;

        operand_a = 128'd0;
        operand_b = 128'd0;

        active_mask = 4'b1111;

        repeat (3) @(posedge clk);

        @(negedge clk);
        rst = 0;

        @(posedge clk);

        // LANE TESTS
        //
        // L0: 2.5 * 4.0  = 10.0
        // L1: 1.5 * 2.0  = 3.0
        // L2: 3.0 * 3.0  = 9.0
        // L3: -2.0 * 4.0 = -8.0

        operand_a = {
            32'hC0000000, // -2.0
            32'h40400000, // 3.0
            32'h3FC00000, // 1.5
            32'h40200000  // 2.5
        };

        operand_b = {
            32'h40800000, // 4.0
            32'h40400000, // 3.0
            32'h40000000, // 2.0
            32'h40800000  // 4.0
        };

        @(negedge clk);
        start = 1;

        @(posedge clk);

        @(negedge clk);
        start = 0;

        wait(done);

        // L0 = 10.0
        if (result[31:0] !== 32'h41200000) begin
            $display("FAIL: lane 0 result=%h", result[31:0]);
            $fatal;
        end

        // L1 = 3.0
        if (result[63:32] !== 32'h40400000) begin
            $display("FAIL: lane 1 result=%h", result[63:32]);
            $fatal;
        end

        // L2 = 9.0
        if (result[95:64] !== 32'h41100000) begin
            $display("FAIL: lane 2 result=%h", result[95:64]);
            $fatal;
        end

        // L3 = -8.0
        if (result[127:96] !== 32'hC1000000) begin
            $display("FAIL: lane 3 result=%h", result[127:96]);
            $fatal;
        end

        $display("PASS: lane 0 2.5 * 4.0 = 10.0");
        $display("PASS: lane 1 1.5 * 2.0 = 3.0");
        $display("PASS: lane 2 3.0 * 3.0 = 9.0");
        $display("PASS: lane 3 -2.0 * 4.0 = -8.0");

        $display(
            "ALL 4-LANE FP32 SIMD MULTIPLIER TESTS PASSED"
        );

        // Lane 0 finishes early on zero; lane 1 runs the finite datapath. A start
        // while SIMD busy must not restart lane 0 and overwrite its result.
        @(negedge clk);
        operand_a = {32'h40200000, 32'h7f800000, 32'h00000001, 32'h00000000};
        operand_b = {32'h40800000, 32'h3f800000, 32'h3f800000, 32'h3f800000};
        active_mask = 15;
        start = 1;
        @(negedge clk);
        start = 0;
        while (!dut.completed_mask[0]) @(negedge clk);
        if (!busy || dut.completed_mask[1]) $fatal(1, "Expected mixed lane latencies");
        operand_a = {4{32'h40400000}};
        operand_b = {4{32'h40400000}};
        start = 1;
        @(negedge clk);
        start = 0;
        wait(done);
        if (result !== {32'h41200000, 32'h7f800000, 32'h00000001, 32'h00000000})
            $fatal(1, "Busy request corrupted an early-finishing lane: %h", result);

        @(negedge clk);
        operand_a = {4{32'h40000000}};
        operand_b = {4{32'h40800000}};
        active_mask = 5;
        start = 1;
        @(negedge clk);
        start = 0;
        wait(done);
        if (result !== {32'h41200000, 32'h41000000, 32'h00000001, 32'h41000000})
            $fatal(1, "Masked SIMD multiply changed inactive lanes");

        @(negedge clk);
        active_mask = 0;
        start = 1;
        @(negedge clk);
        start = 0;
        wait(done);
        if (busy || result !== {32'h41200000, 32'h41000000, 32'h00000001, 32'h41000000})
            $fatal(1, "Empty-mask multiply failed to complete without lane work");
        @(negedge clk);
        @(negedge clk);
        if (done) $fatal(1, "Duplicate SIMD completion");
        $display("PASS: mixed lane latency, busy rejection, inactive lanes, and empty masks");

        #10;
        $finish;

    end

    initial begin

        repeat (300) @(posedge clk);

        $display(
            "FAIL: timeout waiting for SIMD FP32 multiplier"
        );

        $fatal;

    end

endmodule
