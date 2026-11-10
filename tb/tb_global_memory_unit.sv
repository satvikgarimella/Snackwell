`timescale 1ns/1ps
module tb_global_memory_unit;
    logic clk = 0, reset = 1, command_valid = 0;
    logic command_store = 0, command_status = 0;
    logic [1:0] command_warp = 2;
    logic [3:0] command_rd = 7, command_mask = 15;
    logic [127:0] command_bases = 0, command_data = 0;
    logic [31:0] command_offset = 0;
    wire command_ready, busy;
    wire [1:0] warp_id;
    wire [11:0] status_bits;
    wire request_valid, request_write, response_ready;
    wire [31:0] request_address, request_data;
    logic request_ready = 0, response_valid = 0, response_error = 0;
    logic [31:0] response_data = 0;
    wire completion_valid, completion_writes_rd;
    logic completion_ready = 0;
    wire [3:0] completion_rd, completion_mask;
    wire [127:0] completion_data;
    global_memory_unit dut (.*);
    always #5 clk = ~clk;

    task automatic clear;
        @(negedge clk); reset = 1; command_valid = 0;
        request_ready = 0; response_valid = 0; completion_ready = 0;
        repeat (2) @(negedge clk);
        reset = 0;
        if (busy || completion_valid || request_valid || response_ready || status_bits != 0)
            $fatal(1, "Reset retained a global transaction/status");
    endtask

    task automatic issue(input bit store_op, status_op, input logic [3:0] mask,
                         input logic [127:0] bases, data, input logic [31:0] offset);
        @(negedge clk);
        if (!command_ready) $fatal(1, "Command unexpectedly blocked");
        command_store = store_op; command_status = status_op; command_mask = mask;
        command_bases = bases; command_data = data; command_offset = offset;
        command_warp = 2; command_rd = 7; command_valid = 1;
        @(negedge clk); command_valid = 0;
        // The scheduler can now select different operands, mask and metadata.
        command_bases = '1; command_data = '1; command_mask = 0;
        command_warp = 1; command_rd = 9; command_offset = 123;
    endtask

    task automatic transfer(input bit store_op, input logic [31:0] address, data, reply,
                            input bit error_reply);
        integer waits;
        waits = 0;
        while (!request_valid && waits < 20) begin
            @(negedge clk); waits = waits + 1;
        end
        repeat (5) begin
            if (!request_valid || request_write !== store_op || request_address !== address ||
                (store_op && request_data !== data) || command_ready || response_ready)
                $fatal(1, "Request changed while stalled, address=%h expected=%h", request_address, address);
            @(negedge clk);
        end
        request_ready = 1;
        @(negedge clk); request_ready = 0;
        repeat (6) begin
            if (request_valid || !response_ready || completion_valid)
                $fatal(1, "Request duplicated or completed before response");
            @(negedge clk);
        end
        response_data = reply; response_error = error_reply; response_valid = 1;
        @(negedge clk); response_valid = 0; response_data = 'x;
    endtask

    task automatic finish(input bit writes_rd, input logic [3:0] mask,
                          input logic [127:0] expected);
        integer waits;
        waits = 0;
        while (!completion_valid && waits < 20) begin
            @(negedge clk); waits = waits + 1;
        end
        repeat (7) begin
            if (!completion_valid || completion_writes_rd !== writes_rd || warp_id != 2 ||
                completion_rd != 7 || completion_mask != mask || completion_data !== expected ||
                request_valid || command_ready)
                $fatal(1, "Completion lost or changed under backpressure");
            @(negedge clk);
        end
        completion_ready = 1;
        @(negedge clk); completion_ready = 0;
        if (busy || completion_valid) $fatal(1, "Duplicate completion");
    endtask

    task automatic repeated_loads;
        logic [127:0] expected;
        logic [1:0] seen;
        logic [31:0] address, value;
        // Two interleaved address groups; enumerate every active-lane mask.
        for (integer mask = 0; mask < 16; mask = mask + 1) begin
            issue(0, 0, 4'(mask), {32'h2004, 32'h1004, 32'h2004, 32'h1004}, 0, -32'd4);
            expected = 0;
            seen = 0;
            for (integer i = 0; i < 4; i = i + 1) begin
                address = (i % 2 == 0) ? 32'h1000 : 32'h2000;
                value = address ^ 32'habcdef01;
                if (mask & (1 << i)) begin
                    expected[i*32 +: 32] = value;
                    if (!dut.BROADCAST_LOADS || !seen[i % 2])
                        transfer(0, address, 0, value, 0);
                    seen[i % 2] = 1;
                end
            end
            finish(1, 4'(mask), expected);
        end
        if (status_bits != 0) $fatal(1, "Valid broadcast raised an error");
    endtask

    initial begin
        clear();
        issue(0, 0, 4'b1011, {32'h80000004, 32'd3, 32'h1008, 32'h1004}, 0, -32'd4);
        transfer(0, 32'h1000, 0, 11, 0);
        transfer(0, 32'h1004, 0, 22, 0);
        transfer(0, 32'h80000000, 0, 44, 0);
        finish(1, 4'b1011, {32'd44, 32'd0, 32'd22, 32'd11});
        if (status_bits != 0) $fatal(1, "Masked lane raised a fault");

        repeated_loads();

        // One failed response reaches every matching active lane. Distinct
        // addresses still proceed; inactive matches keep their zero payload.
        issue(0, 0, 15, {32'h2000, 32'h1000, 32'h2000, 32'h1000}, 0, 0);
        transfer(0, 32'h1000, 0, 32'hbad, 1);
        transfer(0, 32'h2000, 0, 23, 0);
        if (!dut.BROADCAST_LOADS) begin
            transfer(0, 32'h1000, 0, 32'hbad, 1);
            transfer(0, 32'h2000, 0, 23, 0);
        end
        finish(1, 15, {32'd23, 32'd0, 32'd23, 32'd0});
        if (status_bits !== 12'h100) $fatal(1, "Wrong broadcast error status");

        clear();
        issue(0, 0, 4'b0101, {4{32'h1000}}, 0, 0);
        transfer(0, 32'h1000, 0, 32'hbad, 1);
        if (!dut.BROADCAST_LOADS) transfer(0, 32'h1000, 0, 32'hbad, 1);
        finish(1, 4'b0101, 0);
        if (status_bits !== 12'h100) $fatal(1, "Sparse broadcast error lost");

        clear();
        // Invalid addresses are never merged into a valid response group.
        issue(0, 0, 15, {32'h1004, 32'd0, 32'h1004, 32'd6}, 0, -32'd4);
        transfer(0, 32'h1000, 0, 77, 0);
        if (!dut.BROADCAST_LOADS) transfer(0, 32'h1000, 0, 77, 0);
        finish(1, 15, {32'd77, 32'd0, 32'd77, 32'd0});
        if (status_bits !== 12'h0c0) $fatal(1, "Address errors lost next to broadcast");

        clear();
        issue(0, 0, 15, 0, 0, -32'd4);
        finish(1, 15, 0);
        if (status_bits !== 12'h080) $fatal(1, "Duplicate underflows made requests");
        clear();

        // Unaligned lane, address underflow, and failed external read.
        issue(0, 0, 15, {32'h1004, 32'd0, 32'd6, 32'h1000}, 0, -32'd4);
        transfer(0, 32'hffc, 0, 91, 0);
        transfer(0, 32'h1000, 0, 32'hdeadbeef, 1);
        finish(1, 15, {96'd0, 32'd91});
        if (status_bits !== 12'h1c0) $fatal(1, "Errors attributed to wrong warp");
        issue(0, 1, 15, 0, 0, 0);
        finish(1, 15, {4{32'd7}});

        clear();
        issue(0, 0, 1, {96'd0, 32'hfffffffc}, 0, 4);
        finish(1, 1, 0);
        if (status_bits !== 12'h080) $fatal(1, "Address wrapped at 4 GiB");

        clear();
        // Duplicate stores retain all four requests in lane order.
        issue(1, 0, 15, {4{32'h1000}}, {32'd4, 32'd3, 32'd2, 32'd1}, 0);
        for (integer i = 1; i <= 4; i = i + 1) transfer(1, 32'h1000, i, 0, 0);
        finish(0, 15, 0);
        // Each new instruction reads memory again; there is no retained cache.
        for (integer value = 4; value <= 5; value = value + 1) begin
            issue(0, 0, 15, {4{32'h1000}}, 0, 0);
            transfer(0, 32'h1000, 0, 32'(value), 0);
            if (!dut.BROADCAST_LOADS)
                repeat (3) transfer(0, 32'h1000, 0, 32'(value), 0);
            finish(1, 15, {4{32'(value)}});
        end
        issue(1, 0, 0, '1, '1, 0);
        finish(0, 0, 0);

        // Reset at each bus phase and while a result is held. The endpoint is
        // reset at the same time; software abort uses draining instead.
        issue(0, 0, 15, 0, 0, 0);
        wait (request_valid); clear();
        issue(0, 0, 15, 0, 0, 0);
        wait (request_valid); @(negedge clk); request_ready = 1;
        @(negedge clk); request_ready = 0;
        if (!response_ready) $fatal(1, "No pending request before reset");
        clear();
        issue(0, 0, 1, 0, 0, 0);
        transfer(0, 0, 0, 42, 0);
        wait (completion_valid); clear();
        issue(0, 0, 1, 0, 0, 0);
        transfer(0, 0, 0, 99, 0);
        finish(1, 1, {96'd0, 32'd99});
        $display("PASS global memory snapshots, broadcast=%0d, masks, stalls, errors, ordering and reset", dut.BROADCAST_LOADS);
        $finish;
    end
    initial begin #100000; $fatal(1, "Global unit timeout"); end
endmodule
