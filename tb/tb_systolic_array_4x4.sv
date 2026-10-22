`timescale 1ns/1ps
module tb_systolic_array_4x4;
    logic clk = 0, reset = 1, enable = 1, request_valid = 0, result_ready = 0;
    logic [511:0] a_tile = 0, b_tile = 0;
    wire request_ready, busy, result_valid, step_advance;
    wire [511:0] result_tile;
    wire [3:0] step_index;
    wire [15:0] pe_done_mask;
    systolic_array_4x4 dut (.*);
    always #5 clk = ~clk;

    reg [511:0] source_a, source_b, expected, held_result;
    integer step_starts = 0, advances = 0, macs = 0;
    integer overlapped_steps = 0;
    integer fd, scan, word_scan, case_id, cases = 0, cycles;
    integer latency_min = 100000, latency_max = 0;
    reg [31:0] word_value;
    string vector_path;

    // Check the actual neighbor-to-neighbor operand streams, independently
    // of the final numerical comparison. Each valid pair must have the same k.
    always @(posedge clk) begin
        if (!reset) begin
            if (!enable && (step_advance || dut.pe_start))
                $fatal(1, "Paused array moved operands or launched a step");
            if (result_valid && (|dut.pe_arithmetic_busy))
                $fatal(1, "Tile published before every accumulator drained");
            if (dut.pe_start) begin
                if (|dut.pe_arithmetic_busy) overlapped_steps = overlapped_steps + 1;
                if (step_index !== step_starts[3:0]) $fatal(1, "Repeated/skipped array step");
                for (integer r = 0; r < 4; r = r + 1)
                    for (integer c = 0; c < 4; c = c + 1) begin
                        if (int'(step_index) >= r+c && int'(step_index) < r+c+4) begin
                            if (dut.a_valid_links[r][c] !== 1'b1 || dut.b_valid_links[r][c] !== 1'b1)
                                $fatal(1, "Missing operand at PE %0d,%0d step %0d", r, c, step_index);
                            if (dut.a_links[r][c] !== source_a[(r*4+int'(step_index)-r-c)*32 +: 32] ||
                                dut.b_links[r][c] !== source_b[((int'(step_index)-r-c)*4+c)*32 +: 32])
                                $fatal(1, "Misaligned operand at PE %0d,%0d step %0d", r, c, step_index);
                            macs = macs + 1;
                        end else if (dut.a_valid_links[r][c] !== 1'b0 || dut.b_valid_links[r][c] !== 1'b0)
                            $fatal(1, "Bubble became valid at PE %0d,%0d", r, c);
                    end
                step_starts = step_starts + 1;
            end
            if (step_advance) begin
                if (pe_done_mask !== 16'hffff) $fatal(1, "Advanced before every product reached its adder");
                advances = advances + 1;
            end
        end
    end

    task automatic compare_tile;
        reg [31:0] actual, wanted;
        begin
            for (integer i = 0; i < 16; i = i + 1) begin
                actual = result_tile[i*32 +: 32];
                wanted = expected[i*32 +: 32];
                if (wanted[30:23] == 255 && wanted[22:0] != 0) begin
                    if (actual[30:23] !== 8'hff || (|actual[22:0]) !== 1'b1)
                        $fatal(1, "Case %0d C[%0d,%0d] expected NaN got %h", case_id, i/4, i%4, actual);
                end else if (actual !== wanted)
                    $fatal(1, "Case %0d C[%0d,%0d] got=%h expected=%h", case_id, i/4, i%4, actual, wanted);
            end
        end
    endtask

    task automatic run_tile;
        begin
            @(negedge clk);
            if (!request_ready || busy) $fatal(1, "Array did not return to idle");
            step_starts = 0; advances = 0; macs = 0;
            a_tile = source_a; b_tile = source_b; request_valid = 1;
            enable = 0; // Accept and retain the tile even while execution pauses.
            @(negedge clk);
            a_tile = '1; b_tile = '1; // Subsequent changes cannot alter the job.
            repeat (4) @(negedge clk);
            if (!busy || request_ready || result_valid || step_starts != 0)
                $fatal(1, "Request capture/pause failed");
            request_valid = 0;
            enable = 1;
            cycles = 0;
            while (!result_valid && cycles < 6000) begin
                @(negedge clk);
                cycles = cycles + 1;
                if (request_ready) $fatal(1, "Accepted a second in-flight tile");
                if (cycles == 5) enable = 0;
                if (cycles == 19) enable = 1;
            end
            if (!result_valid) $fatal(1, "Array timeout on case %0d step=%0d done=%h", case_id, step_index, pe_done_mask);
            compare_tile();
            if (step_starts != 10 || advances != 10 || macs != 64)
                $fatal(1, "Expected 10 steps/64 MACs, got %0d/%0d/%0d", step_starts, advances, macs);
            if (cycles < latency_min) latency_min = cycles;
            if (cycles > latency_max) latency_max = cycles;
            held_result = result_tile;
            enable = 0;
            repeat (5) begin
                @(negedge clk);
                if (!result_valid || !busy || request_ready || result_tile !== held_result)
                    $fatal(1, "Result changed under backpressure");
            end
            result_ready = 1;
            @(negedge clk);
            result_ready = 0;
            if (busy || result_valid || !request_ready) $fatal(1, "Result handshake did not retire job");
            enable = 1;
        end
    endtask

    initial begin
        if (!$value$plusargs("TENSOR_VECTORS=%s", vector_path)) $fatal(1, "Missing tensor vectors");
        fd = $fopen(vector_path, "r");
        if (!fd) $fatal(1, "Cannot open tensor vectors");
        repeat (3) @(negedge clk);
        reset = 0;
        scan = $fscanf(fd, "%h", case_id);
        while (scan == 1) begin
            for (integer i = 0; i < 48; i = i + 1) begin
                word_scan = $fscanf(fd, "%h", word_value);
                if (word_scan != 1) $fatal(1, "Malformed tensor vector");
                if (i < 16) source_a[i*32 +: 32] = word_value;
                else if (i < 32) source_b[(i-16)*32 +: 32] = word_value;
                else expected[(i-32)*32 +: 32] = word_value;
            end
            run_tile();
            cases = cases + 1;
            scan = $fscanf(fd, "%h", case_id);
        end
        if (!$feof(fd) || cases == 0) $fatal(1, "Malformed/empty vector file");
        $fclose(fd);
        if (overlapped_steps == 0) $fatal(1, "No multiplication overlapped a preceding addition");
        $display("PASS: %0d matrix tiles (%0d exact FP32 output checks), 10 steps and 64 aligned MACs each", cases, cases*16);
        $display("PASS: request capture, pause/resume, repeated tiles, output backpressure (latency %0d..%0d clocks with test pauses)", latency_min, latency_max);

        // Reset while MACs are busy must abort all sixteen scalar pipelines.
        @(negedge clk);
        step_starts = 0; advances = 0; macs = 0;
        source_a = {16{32'h00000001}}; source_b = {16{32'h00000001}};
        a_tile = source_a; b_tile = source_b; request_valid = 1;
        @(negedge clk); request_valid = 0;
        repeat (8) @(negedge clk);
        reset = 1;
        repeat (3) @(negedge clk);
        reset = 0;
        repeat (100) begin
            @(negedge clk);
            if (busy || result_valid || !request_ready || pe_done_mask != 0)
                $fatal(1, "Stale array completion after reset");
        end
        source_a = 0; source_b = 0; expected = 0;
        run_tile();
        $display("PASS: reset abort and reuse");
        $display("ALL SYSTOLIC ARRAY TESTS PASSED");
        $finish;
    end
    initial begin
        #15000000;
        $fatal(1, "Systolic array global timeout");
    end
endmodule
