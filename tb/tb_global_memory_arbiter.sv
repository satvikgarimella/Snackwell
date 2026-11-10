`timescale 1ns/1ps
module tb_global_memory_arbiter;
    logic clk = 0, reset = 1;
    logic lsu_valid = 1, lsu_write = 1, copy_valid = 1;
    wire lsu_ready, copy_ready, lsu_response_valid, copy_response_valid;
    logic [31:0] lsu_address = 4096, lsu_data = 77, copy_address = 8192;
    logic lsu_response_ready = 0, copy_response_ready = 0;
    wire lsu_error, copy_error;
    wire [31:0] lsu_result, copy_result;
    wire request_valid, request_write, response_ready, busy;
    logic request_ready = 0, response_valid = 0, response_error = 0;
    wire [31:0] request_address, request_data;
    logic [31:0] response_data = 99;
    global_memory_arbiter dut (.*);
    always #5 clk = ~clk;
    initial begin
        repeat (3) @(negedge clk); reset = 0;
        for (integer n = 0; n < 6; n = n + 1) begin
            repeat (4) begin
                @(negedge clk);
                if (!request_valid || request_address !== (n % 2 == 0 ? 32'd4096 : 32'd8192) ||
                    request_write !== (n % 2 == 0)) $fatal(1, "Arbiter changed a stalled grant");
            end
            request_ready = 1;
            @(negedge clk); request_ready = 0;
            response_valid = 1;
            // Only the other client is ready. The response must still wait.
            lsu_response_ready = n % 2 != 0;
            copy_response_ready = n % 2 == 0;
            repeat (3) begin
                @(negedge clk);
                if (response_ready || request_valid || lsu_response_valid !== (n % 2 == 0) ||
                    copy_response_valid !== (n % 2 != 0)) $fatal(1, "Response routed to wrong client");
            end
            lsu_response_ready = 1; copy_response_ready = 1;
            @(negedge clk); response_valid = 0;
            lsu_response_ready = 0; copy_response_ready = 0;
        end
        reset = 1; lsu_valid = 0; copy_valid = 0;
        repeat (2) @(negedge clk); reset = 0;
        if (busy || request_valid || response_ready) $fatal(1, "Arbiter reset retained a grant");
        $display("PASS global arbiter held grants, response routing and round-robin progress");
        $finish;
    end
    initial begin #20000; $fatal(1, "Arbiter timeout"); end
endmodule
