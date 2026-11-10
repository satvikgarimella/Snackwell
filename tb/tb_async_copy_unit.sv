`timescale 1ns/1ps
module tb_async_copy_unit;
    import snackwell_pkg::*;
    logic clk = 0, reset = 1, command_valid = 0;
    wire command_ready;
    logic [5:0] opcode = OP_CPSTATUS;
    logic [1:0] warp_id = 0;
    logic [3:0] active_mask = 15, rd_addr = 0, rs1_addr = 0, rs2_addr = 0;
    logic [13:0] immediate = 0;
    logic [127:0] source = 0;
    logic memory_idle = 1;
    wire busy;
    wire [3:0] warp_pending;
    wire [31:0] status_bits;
    logic [127:0] probe_addresses = {4{32'd128}};
    logic probe_store = 0, matrix_valid = 1, matrix_nvfp4 = 0;
    logic [31:0] matrix_a = 32'h00010080, matrix_b = 32'h00010200;
    wire shared_access_ready, matrix_access_ready;
    wire global_valid, global_response_ready, shared_valid, shared_response_ready;
    logic global_ready = 0, global_response_valid = 0, global_error = 0;
    logic [31:0] global_data = 0;
    wire [31:0] global_address, shared_address, shared_data;
    logic shared_ready = 0, shared_response_valid = 0;
    wire row_valid;
    logic row_ready = 0;
    wire [1:0] row_warp;
    wire [3:0] row_rd, row_mask;
    wire [127:0] row_data;
    async_copy_unit dut (.*);
    always #5 clk = ~clk;

    function automatic logic [127:0] descriptor(input logic [31:0] base, stride, shape, target);
        descriptor = {target, shape, stride, base};
    endfunction
    task automatic clear;
        @(negedge clk); reset = 1; command_valid = 0;
        global_ready = 0; global_response_valid = 0; global_error = 0;
        shared_ready = 0; shared_response_valid = 0; row_ready = 0;
        repeat (2) @(negedge clk); reset = 0; memory_idle = 1;
        if (busy || row_valid || warp_pending || status_bits || global_valid || shared_valid)
            $fatal(1, "Copy reset retained state");
    endtask
    task automatic candidate(input logic [5:0] op, input logic [1:0] warp,
                             input logic [13:0] imm, input logic [127:0] data);
        @(negedge clk);
        opcode = op; warp_id = warp; immediate = imm; source = data;
        rd_addr = (op == OP_CPASYNC || op == OP_CPSTATUS) ? 7 : 0;
        rs1_addr = (op == OP_CPASYNC || op == OP_CPWAIT) ? 2 : 0;
        command_valid = 1;
        #1;
    endtask
    task automatic command(input logic [5:0] op, input logic [1:0] warp,
                           input logic [13:0] imm, input logic [127:0] data = 0);
        candidate(op, warp, imm, data);
        if (!command_ready) $fatal(1, "Unexpected copy command stall %h", op);
        @(negedge clk); command_valid = 0;
    endtask
    task automatic blocked(input logic [5:0] op, input logic [1:0] warp,
                           input logic [13:0] imm, input logic [127:0] data = 0);
        candidate(op, warp, imm, data);
        repeat (3) begin
            if (command_ready) $fatal(1, "Unsafe copy command accepted %h", op);
            @(negedge clk);
        end
        command_valid = 0;
    endtask
    task automatic take(input logic [31:0] expected);
        repeat (5) begin
            @(negedge clk);
            if (!row_valid || row_data !== {4{expected}} || row_warp != 0 || row_rd != 7 || row_mask != 15)
                $fatal(1, "Copy result changed under writeback backpressure");
        end
        row_ready = 1;
        @(negedge clk); row_ready = 0;
    endtask
    task automatic word_copy(input logic [31:0] from_address, to_address, data);
        repeat (4) begin
            @(negedge clk);
            if (!global_valid || global_address !== from_address || shared_valid)
                $fatal(1, "Copy source request lost or changed");
        end
        global_ready = 1;
        @(negedge clk); global_ready = 0;
        repeat (3) @(negedge clk);
        if (!global_response_ready || global_valid) $fatal(1, "Duplicate copy read");
        global_data = data; global_response_valid = 1;
        @(negedge clk); global_response_valid = 0; global_data = 'x;
        repeat (4) begin
            if (!shared_valid || shared_address !== to_address || shared_data !== data)
                $fatal(1, "Shared copy data changed under backpressure");
            @(negedge clk);
        end
        shared_ready = 1;
        @(negedge clk); shared_ready = 0;
        repeat (3) @(negedge clk);
        if (!shared_response_ready || shared_valid) $fatal(1, "Duplicate shared copy write");
        shared_response_valid = 1;
        @(negedge clk); shared_response_valid = 0;
    endtask

    initial begin
        clear();
        blocked(OP_CPWAIT, 2, 0, {4{32'd1}}); // Future generation, before producer.
        command(OP_CPASYNC, 0, 10, descriptor(4096, 20, 32'h0302, 32'h00020080));
        source = '1;
        take(1);
        blocked(OP_CPWAIT, 2, 0, {4{32'd1}});
        if (shared_access_ready || matrix_access_ready) $fatal(1, "Partial copy became visible");
        for (integer r = 0; r < 2; r = r + 1)
            for (integer c = 0; c < 3; c = c + 1)
                word_copy(4096 + r*20 + c*4, 128 + r*32 + c*4, 10+r*3+c);
        if (busy || shared_access_ready || matrix_access_ready) $fatal(1, "Copy skipped explicit wait");
        command(OP_CPWAIT, 0, 0, {4{32'd1}});
        #1;
        if (!shared_access_ready || !matrix_access_ready) $fatal(1, "Wait did not grant read access");
        probe_store = 1; #1;
        if (shared_access_ready) $fatal(1, "Leased operands were writable");
        probe_store = 0;
        memory_idle = 0;
        blocked(OP_CPRELEASE, 0, 0);
        memory_idle = 1;
        command(OP_CPRELEASE, 0, 0);
        blocked(OP_CPASYNC, 0, 2, descriptor(4096, 4, 32'h0101, 32'h00040080));
        blocked(OP_CPASYNC, 0, 3, descriptor(4096, 4, 32'h0101, 32'h00040080));
        command(OP_CPWAIT, 2, 0, {4{32'd1}});
        command(OP_CPRELEASE, 2, 0);
        command(OP_CPASYNC, 0, 2, descriptor(8192, 4, 32'h0101, 32'h00040080));
        take(2);
        command(OP_CPWAIT, 0, 0, {4{32'd1}});
        if (status_bits[7:0] != 8) $fatal(1, "Stale ticket not rejected");
        command(OP_CPRELEASE, 0, 0);
        if (status_bits[7:0] != 8'h48) $fatal(1, "Release before wait not rejected");
        word_copy(8192, 128, 99);
        command(OP_CPWAIT, 0, 0, {4{32'd2}});
        command(OP_CPRELEASE, 0, 0);

        clear();
        // Invalid descriptors produce a zero ticket and no traffic.
        command(OP_CPASYNC, 0, 2, descriptor(32'hfffffffc, 16, 32'h0401, 32'h00010080));
        take(0);
        command(OP_CPASYNC, 0, 2, descriptor(4096, 16, 32'h0402, 32'h00010ff0));
        take(0);
        command(OP_CPASYNC, 0, 2, descriptor(4096, 16, 32'h0400, 32'h00010080));
        take(0);
        if (busy || warp_pending || status_bits != 4) $fatal(1, "Invalid descriptor changed ownership");

        clear();
        command(OP_CPASYNC, 0, 10, descriptor(65536, 4, 32'h0101, 32'h00040080));
        take(1);
        global_ready = 1;
        @(negedge clk); global_ready = 0;
        global_error = 1; global_response_valid = 1;
        @(negedge clk); global_response_valid = 0; global_error = 0;
        command(OP_CPWAIT, 2, 0, {4{32'd1}});
        if (shared_access_ready || matrix_access_ready || status_bits != 32'h00100010)
            $fatal(1, "Failed copy was readable or error was not propagated");
        command(OP_CPRELEASE, 2, 0);
        command(OP_CPWAIT, 0, 0, {4{32'd1}});
        command(OP_CPRELEASE, 0, 0);
        clear();
        dut.generation[0] = 32'hffffffff;
        command(OP_CPASYNC, 0, 2, descriptor(4096, 4, 32'h0101, 32'h00040080));
        take(0);
        if (busy || status_bits != 32'h20) $fatal(1, "Generation wrapped");

        // Reset cancels all phases; the external endpoint must reset with it.
        clear();
        command(OP_CPASYNC, 0, 2, descriptor(4096, 4, 32'h0101, 32'h00040080));
        clear();
        command(OP_CPASYNC, 0, 2, descriptor(4096, 4, 32'h0101, 32'h00040080));
        global_ready = 1;
        @(negedge clk); global_ready = 0;
        clear();
        command(OP_CPASYNC, 0, 2, descriptor(4096, 4, 32'h0101, 32'h00040080));
        global_ready = 1;
        @(negedge clk); global_ready = 0; global_response_valid = 1; global_data = 42;
        @(negedge clk); global_response_valid = 0;
        clear();
        $display("PASS async copy descriptors, snapshots, tickets, leases, access guards, failures and reset");
        $finish;
    end
    initial begin #200000; $fatal(1, "Async unit timeout"); end
endmodule
