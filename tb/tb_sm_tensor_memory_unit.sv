`timescale 1ns/1ps
module tb_sm_tensor_memory_unit;
    import snackwell_pkg::*;
    logic clk = 0, reset = 1, command_valid = 0;
    wire command_ready;
    logic [5:0] opcode = OP_TSTATUS;
    logic [1:0] warp_id = 0;
    logic [3:0] active_mask = 15, rd_addr = 0, rs1_addr = 0, rs2_addr = 0;
    logic [13:0] immediate = 0;
    logic [127:0] source_a = 0, source_b = 0;
    logic array_available = 1, memory_idle = 1;
    wire busy, memory_lock, matrix_guard_valid, array_nvfp4_mode;
    wire [3:0] warp_pending;
    wire memory_request_valid, memory_response_ready;
    logic memory_request_ready = 0, memory_response_valid = 0;
    wire [127:0] memory_addresses;
    logic [127:0] memory_data = 0;
    wire array_request_valid, array_result_ready;
    logic array_request_ready = 0, array_result_valid = 0;
    wire [511:0] a_tile, b_tile, c_tile;
    logic [511:0] array_result = 0;
    wire row_valid;
    logic row_ready = 0;
    wire [1:0] row_warp_id;
    wire [3:0] row_rd_addr, row_write_mask;
    wire [127:0] row_data;
    sm_tensor_memory_unit dut (.*);
    always #5 clk = ~clk;

    task automatic reset_unit;
        @(negedge clk); reset = 1; command_valid = 0;
        memory_request_ready = 0; memory_response_valid = 0;
        array_request_ready = 0; array_result_valid = 0; row_ready = 0;
        repeat (2) @(negedge clk);
        reset = 0;
        if (busy || row_valid || memory_request_valid || array_request_valid || warp_pending)
            $fatal(1, "Reset retained tensor work or leases");
    endtask

    task automatic candidate(input logic [5:0] op, input logic [1:0] warp,
                             input logic [13:0] imm, input logic [127:0] a = 0,
                             input logic [127:0] b = 0);
        @(negedge clk);
        opcode = op; warp_id = warp; immediate = imm;
        source_a = a; source_b = b; command_valid = 1;
        #1;
    endtask

    task automatic command(input logic [5:0] op, input logic [1:0] warp,
                           input logic [13:0] imm, input logic [127:0] a = 0,
                           input logic [127:0] b = 0);
        candidate(op, warp, imm, a, b);
        if (!command_ready) $fatal(1, "Unexpected stall for opcode %h", op);
        @(negedge clk); command_valid = 0;
    endtask

    task automatic blocked(input logic [5:0] op, input logic [1:0] warp,
                           input logic [13:0] imm, input logic [127:0] a = 0);
        candidate(op, warp, imm, a);
        repeat (3) begin
            if (command_ready) $fatal(1, "Unsafe command accepted: %h", op);
            @(negedge clk);
        end
        command_valid = 0;
    endtask

    task automatic take_row(input logic [127:0] expected, input logic [1:0] warp);
        repeat (4) begin
            @(negedge clk);
            if (!row_valid || row_data !== expected || row_warp_id != warp || row_write_mask != 15)
                $fatal(1, "Writeback changed under backpressure");
        end
        row_ready = 1;
        @(negedge clk); row_ready = 0;
        if (row_valid) $fatal(1, "Duplicate writeback");
    endtask

    // Delayed memory model returns address-tagged words, not matrix results.
    // Arithmetic is covered by the array and full-SM numerical tests.
    task automatic feed_operands;
        integer base;
        reg [127:0] expected;
        for (integer row = 0; row < 8; row = row + 1) begin
            base = row < 4 ? row * 32 : 256 + (row - 4) * 16;
            for (integer lane = 0; lane < 4; lane = lane + 1)
                expected[lane*32 +: 32] = base + lane * 4;
            repeat (3) begin
                @(negedge clk);
                if (!memory_request_valid || memory_addresses !== expected)
                    $fatal(1, "Descriptor address changed or request lost");
            end
            memory_request_ready = 1;
            @(negedge clk); memory_request_ready = 0;
            repeat (row % 3 + 1) @(negedge clk);
            if (!memory_response_ready || memory_request_valid)
                $fatal(1, "Incorrect memory response ownership");
            memory_data = expected; memory_response_valid = 1;
            @(negedge clk); memory_response_valid = 0; memory_data = 'x;
        end
        for (integer r = 0; r < 4; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1)
                if (a_tile[(r*4+c)*32 +: 32] !== 32'(r*32+c*4) ||
                    b_tile[(r*4+c)*32 +: 32] !== 32'(256+r*16+c*4))
                    $fatal(1, "Operand snapshot corrupted");
        repeat (3) begin
            @(negedge clk);
            if (!array_request_valid || memory_lock) $fatal(1, "Array request not held");
        end
        array_request_ready = 1;
        @(negedge clk); array_request_ready = 0;
    endtask

    task automatic feed_nvfp4;
        reg [127:0] expected;
        integer base;
        for (integer word_index = 0; word_index < 6; word_index = word_index + 1) begin
            base = word_index < 3 ? word_index * 4 : 128 + (word_index - 3) * 4;
            for (integer lane = 0; lane < 4; lane = lane + 1)
                expected[lane*32 +: 32] = 32'(base + lane * 12);
            repeat (3) begin
                @(negedge clk);
                if (!memory_request_valid || memory_addresses !== expected)
                    $fatal(1, "NVFP4 gather changed under backpressure");
            end
            memory_request_ready = 1;
            @(negedge clk); memory_request_ready = 0;
            repeat (2) @(negedge clk);
            memory_data = (word_index == 2 || word_index == 5) ? {4{32'h38}} : {4{32'h22222222}};
            memory_response_valid = 1;
            @(negedge clk); memory_response_valid = 0;
        end
    endtask

    initial begin
        reset_unit();
        command(OP_TALLOC, 0, 0);
        command(OP_TALLOC, 1, 1);
        blocked(OP_TALLOC, 2, 0);
        for (integer row = 0; row < 4; row = row + 1)
            command(OP_TST, 0, 14'(row*2), {4{32'h3f800000}});
        command(OP_TMMA_ACC, 0, 0, {4{32'h00020000}}, {4{32'h00010100}});
        command(OP_TCOMMIT, 0, 10); // Warps 0 and 2 must both release.
        if (!row_valid || row_data !== {4{32'd1}}) $fatal(1, "Wrong first ticket");
        // Change descriptors while the command output is stalled: accepted
        // work must continue independently using only its captured operands.
        source_a = '1; source_b = '1;
        feed_operands();
        if (c_tile !== {16{32'h3f800000}}) $fatal(1, "Initial accumulator lost");
        take_row({4{32'd1}}, 0);
        blocked(OP_TWAIT, 2, 0, {4{32'd1}});
        array_result = {16{32'h40400000}};
        array_result_valid = 1;
        @(negedge clk); array_result_valid = 0;
        command(OP_TWAIT, 2, 0, {4{32'd1}});
        command(OP_TLD, 2, 4);
        take_row({4{32'h40400000}}, 2);
        blocked(OP_TFREE, 0, 0);
        command(OP_TRELEASE, 2, 0);
        blocked(OP_TFREE, 0, 0); // Owner has not waited/released yet.
        command(OP_TWAIT, 0, 0, {4{32'd1}});
        command(OP_TRELEASE, 0, 0);
        command(OP_TFREE, 0, 0);
        command(OP_TWAIT, 2, 0, {4{32'd1}}); // Released generation cannot revive.
        command(OP_TSTATUS, 2, 0);
        take_row({4{32'h10}}, 2);

        // Reset during a memory transaction, then during array execution.
        command(OP_TALLOC, 0, 0);
        command(OP_TMMA, 0, 0, {4{32'h00020000}}, {4{32'h00010100}});
        memory_request_ready = 1;
        @(negedge clk); memory_request_ready = 0;
        reset_unit();
        command(OP_TALLOC, 0, 0);
        command(OP_TMMA, 0, 0, {4{32'h00020000}}, {4{32'h00010100}});
        feed_operands();
        reset_unit();

        command(OP_TALLOC, 0, 0);
        command(OP_TMMA_NVFP4, 0, 0, {4{32'h0000c000}}, {4{32'h0000c080}});
        feed_nvfp4();
        // Four internal array passes belong to one generation. Model the
        // result port independently; full-SM tests check actual arithmetic.
        for (integer pass = 0; pass < 4; pass = pass + 1) begin
            wait (array_request_valid);
            repeat (3) begin
                @(negedge clk);
                if (!array_request_valid || !array_nvfp4_mode || memory_lock ||
                    a_tile !== {16{32'h382}} || b_tile !== {16{32'h382}} ||
                    c_tile !== {16{32'(pass)}} || dut.completed[0])
                    $fatal(1, "NVFP4 pass lost snapshot, accumulator or completion ordering");
            end
            array_request_ready = 1;
            @(negedge clk); array_request_ready = 0;
            repeat (4) @(negedge clk);
            array_result = {16{32'(pass+1)}};
            array_result_valid = 1;
            @(negedge clk); array_result_valid = 0;
        end
        command(OP_TCOMMIT, 0, 2);
        take_row({4{32'd1}}, 0);
        command(OP_TWAIT, 0, 0, {4{32'd1}});
        command(OP_TLD, 0, 0);
        take_row({4{32'd4}}, 0);
        command(OP_TRELEASE, 0, 0);
        command(OP_TFREE, 0, 0);

        // Reset the packed path after gathering and during its first pass.
        command(OP_TALLOC, 0, 0);
        command(OP_TMMA_NVFP4, 0, 0, {4{32'h0000c000}}, {4{32'h0000c080}});
        feed_nvfp4();
        wait (array_request_valid);
        @(negedge clk); array_request_ready = 1;
        @(negedge clk); array_request_ready = 0;
        reset_unit();
        command(OP_TSTATUS, 0, 0);
        take_row(0, 0);
        $display("PASS tensor ownership, completion, backpressure and reset");
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "Tensor-memory unit timeout");
    end
endmodule
