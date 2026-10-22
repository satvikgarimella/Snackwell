`timescale 1ns/1ps
module tb_shared_memory;
    logic clk = 0, reset = 1;
    logic request_valid = 0, response_ready = 0;
    wire request_ready, response_valid;
    logic [127:0] lane_addresses = 0, store_data = 0;
    logic [3:0] load_enable_mask = 0, store_enable_mask = 0;
    wire [127:0] load_data;
    wire [3:0] conflict_mask;
    wire bank_conflict;
    shared_memory dut (.*);
    always #5 clk = ~clk;

    reg [31:0] model [0:1023];
    integer transactions = 0;
    reg [127:0] addresses, values;

    task automatic reset_memory;
        @(negedge clk);
        reset = 1; request_valid = 0; response_ready = 0;
        repeat (2) @(negedge clk);
        reset = 0;
        for (integer i = 0; i < 1024; i = i + 1) model[i] = 0;
        #1;
        if (!request_ready || response_valid || load_data !== 0)
            $fatal(1, "Reset did not clear request/result state");
    endtask

    // A flat architectural memory is the oracle. Lane ordering only matters
    // for same-word accesses; independent banks can execute in any order.
    task automatic transfer(input reg [127:0] addr, data,
        input reg [3:0] loads, stores, input integer hold_cycles = 0);
        reg [127:0] expected;
        reg [3:0] conflicts, active;
        integer count [0:3];
        integer latency, elapsed, word_index;
        reg repeated_read;
        begin
            expected = 0; conflicts = 0; active = loads | stores;
            for (integer b = 0; b < 4; b = b + 1) count[b] = 0;
            for (integer lane = 0; lane < 4; lane = lane + 1) begin
                word_index = addr[lane*32+2 +: 10];
                if (loads[lane]) expected[lane*32 +: 32] = model[word_index];
                if (stores[lane]) model[word_index] = data[lane*32 +: 32];
                repeated_read = 0;
                for (integer earlier = 0; earlier < lane; earlier = earlier + 1)
                    if (stores == 0 && loads[earlier] &&
                        addr[earlier*32+2 +: 10] == word_index)
                        repeated_read = 1;
                if (active[lane] && !repeated_read)
                    count[word_index % 4] = count[word_index % 4] + 1;
            end
            latency = 1;
            for (integer b = 0; b < 4; b = b + 1)
                if (count[b] > latency) latency = count[b];
            for (integer lane = 0; lane < 4; lane = lane + 1)
                if (active[lane] && count[addr[lane*32+2 +: 2]] > 1) conflicts[lane] = 1;

            @(negedge clk);
            if (!request_ready) $fatal(1, "Memory never released previous transaction");
            request_valid = 1; lane_addresses = addr; store_data = data;
            load_enable_mask = loads; store_enable_mask = stores;
            #1;
            if (conflict_mask !== conflicts || bank_conflict !== (|conflicts))
                $fatal(1, "Conflict diagnostic got=%b expected=%b", conflict_mask, conflicts);
            @(negedge clk);
            // Keep a different request offered during replay/backpressure.
            // It must not change the captured request or execute accidentally.
            lane_addresses = {4{32'hffc}}; store_data = {4{32'hbadbad00}};
            load_enable_mask = 0; store_enable_mask = 15;
            elapsed = 0;
            while (!response_valid && elapsed < 6) begin
                if (request_ready) $fatal(1, "Accepted another request during replay");
                @(negedge clk);
                elapsed = elapsed + 1;
            end
            if (!response_valid || elapsed != latency || load_data !== expected)
                $fatal(1, "Transaction %0d: cycles=%0d expected=%0d data=%h expected=%h",
                    transactions, elapsed, latency, load_data, expected);
            repeat (hold_cycles) begin
                @(negedge clk);
                if (!response_valid || request_ready || load_data !== expected)
                    $fatal(1, "Response changed under backpressure");
            end
            request_valid = 0; response_ready = 1;
            @(negedge clk);
            response_ready = 0;
            if (response_valid || !request_ready) $fatal(1, "Completion did not retire");
            transactions = transactions + 1;
        end
    endtask

    task automatic check_memory;
        for (integer row = 0; row < 256; row = row + 1) begin
            if (dut.bank0[row] !== model[row*4] || dut.bank1[row] !== model[row*4+1] ||
                dut.bank2[row] !== model[row*4+2] || dut.bank3[row] !== model[row*4+3])
                $fatal(1, "Memory differs from reference at row %0d", row);
        end
    endtask

    initial begin
        reset_memory();
        // Every assignment of four lanes to four banks, with every active
        // mask. This includes no conflicts, pairs, triples, and four-way hits.
        for (integer pattern = 0; pattern < 256; pattern = pattern + 1)
            for (integer mask = 0; mask < 16; mask = mask + 1) begin
                for (integer lane = 0; lane < 4; lane = lane + 1) begin
                    addresses[lane*32 +: 32] = (((pattern + lane) % 256) * 16) +
                                               (((pattern >> (2*lane)) & 3) * 4);
                    values[lane*32 +: 32] = 32'h10000000 + pattern*256 + mask*4 + lane;
                end
                transfer(addresses, values, 0, mask, pattern % 3);
                transfer(addresses, 0, mask, 0, mask % 3);
            end
        check_memory();
        $display("PASS: all 256 bank mappings x 16 active masks, stores/loads, snapshots and backpressure");

        // All lanes accessing one word: stores commit in ascending lane order,
        // and every active load sees the last stored value. Inactive lanes zero.
        for (integer mask = 0; mask < 16; mask = mask + 1) begin
            transfer({4{32'hffc}}, {32'd44, 32'd33, 32'd22, 32'd11}, 0, mask, 2);
            transfer({4{32'hffc}}, 0, 15, 0, 2);
        end
        transfer({4{32'h40}}, {32'd4, 32'd3, 32'd2, 32'd1}, 15, 15, 2);
        // Multicast groups can be separated by another word in lane order.
        // Aliased byte addresses retain the architectural 4 KB word mapping.
        for (integer mask = 0; mask < 16; mask = mask + 1) begin
            transfer({32'h40, 32'h50, 32'h40, 32'h50}, 0, mask, 0, 2);
            transfer({32'h1040, 32'h40, 32'h1040, 32'h40}, 0, mask, 0, 1);
            transfer({32'h54, 32'h50, 32'h44, 32'h40}, 0, mask, 0);
        end
        // A store between same-word reads forbids multicast across the store.
        transfer({4{32'h40}}, {4{32'h1234}}, 4'b1101, 4'b0010, 2);
        check_memory();
        $display("PASS: same-address ordering, repeated reads, mixed read/write and empty masks");

        // Reset interrupts a four-way store after only its first lane commits.
        @(negedge clk);
        lane_addresses = {32'h30, 32'h20, 32'h10, 32'h0};
        store_data = {4{32'h12345678}}; load_enable_mask = 0; store_enable_mask = 15;
        request_valid = 1;
        repeat (2) @(negedge clk);
        request_valid = 0;
        if (dut.bank0[0] !== 32'h12345678 || dut.bank0[3] !== model[12])
            $fatal(1, "Reset test did not interrupt partial store replay");
        reset_memory();
        repeat (6) @(negedge clk);
        if (response_valid) $fatal(1, "Stale completion after reset during replay");
        check_memory();
        transfer({32'h30, 32'h20, 32'h10, 32'h0}, 0, 15, 0);

        // Reset also discards a completed result waiting for its consumer.
        @(negedge clk);
        request_valid = 1; load_enable_mask = 1; store_enable_mask = 0;
        repeat (2) @(negedge clk);
        request_valid = 0;
        if (!response_valid) $fatal(1, "Missing response before reset");
        reset_memory();
        repeat (6) @(negedge clk);
        if (response_valid) $fatal(1, "Stale held response after reset");
        transfer(0, 128'd42, 0, 1);
        transfer(0, 0, 1, 0);
        check_memory();
        $display("PASS: reset during replay/response and subsequent reuse");
        $display("ALL SHARED MEMORY TESTS PASSED (%0d checked transactions)", transactions);
        $finish;
    end
    initial begin
        #2000000;
        $fatal(1, "Shared memory timeout");
    end
endmodule
