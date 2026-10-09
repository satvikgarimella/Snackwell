`timescale 1ns/1ps
module tb_fp32_mac_pe;
    logic clk = 0, reset = 1, clear = 0;
    logic step_valid = 0, a_valid = 0, b_valid = 0, advance = 0;
    logic [31:0] a_data = 0, b_data = 0;
    wire step_ready, step_done, a_out_valid, b_out_valid;
    wire [31:0] a_out, b_out, accumulator;
    fp32_mac_pe dut (.*);
    always #5 clk = ~clk;

    task automatic do_step(input logic av, bv, input logic [31:0] a, b, expected);
        integer cycles;
        reg [31:0] previous_a, previous_b;
        begin
            @(negedge clk);
            if (!step_ready) $fatal(1, "PE did not become ready");
            previous_a = a_out; previous_b = b_out;
            step_valid = 1; a_valid = av; b_valid = bv; a_data = a; b_data = b;
            @(negedge clk);
            // Change input data and leave valid asserted. Backpressure must
            // prevent accepting a duplicate operation while this step is held.
            a_data = 32'h7fc00000; b_data = 32'h7fc00000;
            cycles = 0;
            while (!step_done && cycles < 600) begin
                if (step_ready) $fatal(1, "PE accepted work before retirement");
                advance = 1; // Early advance must not move an incomplete step.
                @(negedge clk);
                advance = 0;
                cycles = cycles + 1;
            end
            if (!step_done) $fatal(1, "MAC timeout");
            if (expected[30:23] == 8'hff && expected[22:0] != 0) begin
                if (accumulator[30:23] !== 8'hff || (|accumulator[22:0]) !== 1'b1)
                    $fatal(1, "Expected NaN, got %h", accumulator);
            end else if (accumulator !== expected)
                $fatal(1, "MAC got=%h expected=%h", accumulator, expected);
            repeat (4) begin
                if (!step_done || step_ready || a_out !== previous_a || b_out !== previous_b)
                    $fatal(1, "Completion/forwarding changed without advance");
                @(negedge clk);
            end
            step_valid = 0;
            advance = 1;
            @(negedge clk);
            advance = 0;
            if (!step_ready || step_done || a_out !== a || b_out !== b ||
                a_out_valid !== av || b_out_valid !== bv)
                $fatal(1, "PE forwarded incorrect tokens");
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        reset = 0;
        do_step(1, 1, 32'h40000000, 32'h40400000, 32'h40c00000); // 2*3=6
        do_step(1, 1, 32'hbf800000, 32'h40800000, 32'h40000000); // 6-4=2
        do_step(1, 0, 32'h7fc00000, 32'h7f800000, 32'h40000000);
        do_step(0, 1, 32'h7f800000, 32'h12345678, 32'h40000000);
        do_step(0, 0, 32'habcdef01, 32'h12345678, 32'h40000000);
        do_step(1, 1, 32'h00000000, 32'h7f800000, 32'h7fc00000);
        $display("PASS: accumulation, bubbles, zero operands, token capture, and completion backpressure");

        @(negedge clk); clear = 1;
        @(negedge clk); clear = 0;
        do_step(1, 1, 32'hbf800000, 32'h3f800000, 32'hbf800000);
        // Rounded multiply is 1, so the separate addition produces +0.
        // A fused operation would instead produce -2^-46.
        do_step(1, 1, 32'h3f800001, 32'h3f7ffffe, 32'h00000000);
        do_step(1, 1, 32'h00000001, 32'h3f800000, 32'h00000001);
        $display("PASS: non-fused rounding and subnormal accumulation");

        for (integer use_reset = 0; use_reset < 2; use_reset = use_reset + 1) begin
            @(negedge clk);
            a_data = 1; b_data = 32'h00000001; a_valid = 1; b_valid = 1; step_valid = 1;
            @(negedge clk); step_valid = 0;
            repeat (4) @(negedge clk);
            if (use_reset) reset = 1; else clear = 1;
            @(negedge clk); reset = 0; clear = 0;
            repeat (100) begin
                @(negedge clk);
                if (!step_ready || step_done || accumulator !== 0 || a_out_valid || b_out_valid)
                    $fatal(1, "Stale PE state after clear/reset abort");
            end
            do_step(1, 1, 32'h3f800000, 32'h3f800000, 32'h3f800000);
        end
        $display("PASS: clear/reset abort in-flight arithmetic and permit reuse");
        $display("ALL FP32 MAC PE TESTS PASSED");
        $finish;
    end
    initial begin
        #1000000;
        $fatal(1, "PE test timeout");
    end
endmodule
