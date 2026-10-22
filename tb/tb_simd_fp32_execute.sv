`timescale 1ns/1ps
module tb_simd_fp32_execute;
    import snackwell_pkg::*;
    logic clk = 0, rst = 1, start = 0;
    logic [5:0] opcode = OP_FMUL;
    logic [127:0] operand_a = 0, operand_b = 0;
    logic [3:0] active_mask = 0;
    wire busy, done;
    wire [127:0] result;
    simd_fp32_execute dut (.*);
    always #5 clk = ~clk;

    integer fd, parsed, count = 0, cycles;
    reg [31:0] op_word, mask_word;
    reg [31:0] a[0:3], b[0:3], expected[0:3];
    reg [127:0] saved_result;
    string vector_path;

    function automatic bit fp_equal(input logic [31:0] actual, wanted);
        if (wanted[30:23] == 8'hff && wanted[22:0] != 0)
            fp_equal = actual[30:23] === 8'hff && (|actual[22:0]) === 1'b1;
        else fp_equal = actual === wanted;
    endfunction

    initial begin
        if ($test$plusargs("TRACE")) begin $dumpfile("fp.vcd"); $dumpvars(0, tb_simd_fp32_execute); end
        if (!$value$plusargs("VECTORS=%s", vector_path)) $fatal(1, "Missing vector file");
        fd = $fopen(vector_path, "r");
        if (!fd) $fatal(1, "Cannot open vector file");
        repeat (3) @(negedge clk);
        rst = 0;
        while (!$feof(fd)) begin
            parsed = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
                op_word, mask_word, a[0], b[0], expected[0], a[1], b[1], expected[1],
                a[2], b[2], expected[2], a[3], b[3], expected[3]);
            if (parsed != 14) $fatal(1, "Malformed vector %0d", count);
            @(negedge clk);
            opcode = op_word[5:0];
            active_mask = mask_word[3:0];
            operand_a = {a[3], a[2], a[1], a[0]};
            operand_b = {b[3], b[2], b[1], b[0]};
            start = 1;
            @(negedge clk);
            start = 0;
            // External inputs may change immediately after acceptance.
            operand_a = '1;
            operand_b = '1;
            opcode = OP_FMUL;
            active_mask = 4'b1111;
            cycles = 0;
            while (!done && cycles < 400) begin
                @(negedge clk);
                cycles = cycles + 1;
                // A stray start while busy must not restart completed lanes.
                start = busy && cycles == 3;
            end
            start = 0;
            if (!done || busy) $fatal(1, "FP timeout vector %0d busy=%b done=%b complete=%b lane_done=%b", count, busy, done, dut.completed_mask, dut.lane_done);
            for (integer lane = 0; lane < 4; lane = lane + 1)
                if (mask_word[lane] && !fp_equal(result[lane*32 +: 32], expected[lane]))
                    $fatal(1, "vector %0d op=%h lane=%0d a=%h b=%h got=%h expected=%h",
                        count, op_word, lane, a[lane], b[lane], result[lane*32 +: 32], expected[lane]);
            saved_result = result;
            @(negedge clk);
            if (done || busy || result !== saved_result)
                $fatal(1, "Completion must pulse once and retain its result");
            count = count + 1;
        end
        $fclose(fd);

        // Reset aborts an outstanding operation, including every scalar lane.
        opcode = OP_FMUL;
        operand_a = {4{32'h00000001}};
        operand_b = {4{32'h3f800000}};
        active_mask = 15;
        start = 1;
        @(negedge clk);
        start = 0;
        repeat (4) @(negedge clk);
        rst = 1;
        @(negedge clk);
        rst = 0;
        repeat (100) begin
            @(negedge clk);
            if (busy || done) $fatal(1, "Stale completion after reset");
        end
        $display("PASS: %0d SIMD FP vector groups, masks, busy rejection, and reset", count);
        $finish;
    end

    initial begin
        #20000000;
        $fatal(1, "Global timeout");
    end
endmodule
