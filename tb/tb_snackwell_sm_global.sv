`timescale 1ns/1ps
`include "tb/support/global_memory_model.sv"
module tb_snackwell_sm_global;
    import snackwell_pkg::*;
    logic clk = 0, reset = 1, run_enable = 0, launch_enable = 0;
    logic debug_write_enable = 0;
    wire [3:0] halted_mask;
    wire global_request_valid, global_request_ready, global_request_write;
    wire [31:0] global_request_address, global_request_data;
    wire global_response_valid, global_response_ready, global_response_error;
    wire [31:0] global_response_data;
    integer reads, writes, responses, request_stalls;
    snackwell_sm dut (
        .clk(clk), .reset(reset), .run_enable(run_enable),
        .launch_enable(launch_enable), .launch_warp_id(2'd2),
        .launch_pc(32'd0), .launch_active_mask(4'b0101),
        .debug_write_enable(debug_write_enable), .debug_warp_id(2'd0),
        .debug_lane_id(2'd0), .debug_reg_addr(4'd15), .debug_write_data(32'd123),
        .halted_mask(halted_mask),
        .global_request_valid(global_request_valid), .global_request_ready(global_request_ready),
        .global_request_write(global_request_write), .global_request_address(global_request_address),
        .global_request_data(global_request_data), .global_response_valid(global_response_valid),
        .global_response_ready(global_response_ready), .global_response_error(global_response_error),
        .global_response_data(global_response_data)
    );
    global_memory_model ram (
        .clk(clk), .reset(reset), .request_valid(global_request_valid),
        .request_ready(global_request_ready), .request_write(global_request_write),
        .request_address(global_request_address), .request_data(global_request_data),
        .response_valid(global_response_valid), .response_ready(global_response_ready),
        .response_error(global_response_error), .response_data(global_response_data),
        .hold_requests(1'b0), .hold_responses(1'b0), .latency(2),
        .reads(reads), .writes(writes), .responses(responses), .request_stalls(request_stalls)
    );
    always #5 clk = ~clk;

    initial begin
        ram.memory[1024] = 32'h3fa00000; // 1.25f
        ram.memory[1025] = 32'hdeadbeef;
        repeat (3) @(negedge clk);
        reset = 0;
        dut.core.imem.memory[0] = {OP_LD_GLOBAL, 4'd2, 4'd0, 4'd0, 14'd4096};
        dut.core.imem.memory[1] = {OP_ADDI, 4'd3, 4'd0, 4'd0, 14'd9};
        dut.core.imem.memory[2] = {OP_FADD, 4'd4, 4'd2, 4'd2, 14'd0};
        dut.core.imem.memory[3] = {OP_ST_GLOBAL, 4'd0, 4'd0, 4'd4, 14'd4100};
        dut.core.imem.memory[4] = {OP_EXIT, 26'd0};
        @(negedge clk); launch_enable = 1;
        @(negedge clk); launch_enable = 0; run_enable = 1;
        // Hold an earlier integer result in the register-file writeback slot.
        wait (dut.core.wb_valid && dut.core.wb_rd_addr == 3);
        @(negedge clk); debug_write_enable = 1;
        repeat (40) @(negedge clk);
        repeat (10) begin
            if (!dut.core.global_completion_valid || dut.core.global_completion_ready ||
                !dut.core.wb_valid || dut.core.wb_rd_addr != 3 ||
                dut.core.global_warp != 2 || dut.core.global_mask != 5 ||
                dut.core.global_data !== {32'd0, 32'h3fa00000, 32'd0, 32'h3fa00000} ||
                halted_mask[2]) $fatal(1, "Global completion escaped a blocked writeback slot");
            @(negedge clk);
        end
        debug_write_enable = 0;
        wait (halted_mask[2]);
        @(negedge clk);
        if (!dut.core.execution_idle || reads != 1 || writes != 2 || responses != 3 ||
            ram.memory[1025] !== 32'h40200000) $fatal(1, "Global result did not drain after writeback resumed");
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            if (dut.core.regfile.registers[2][lane][2] !== (lane % 2 == 0 ? 32'h3fa00000 : 32'd0) ||
                dut.core.regfile.registers[2][lane][3] !== (lane % 2 == 0 ? 32'd9 : 32'd0) ||
                dut.core.regfile.registers[2][lane][4] !== (lane % 2 == 0 ? 32'h40200000 : 32'd0))
                $fatal(1, "Wrong warp/lane received a global result");
        end
        $display("PASS global SM writeback backpressure, dependent FP, warp/mask tags and halt drain");
        $finish;
    end
    initial begin #100000; $fatal(1, "Global SM timeout"); end
endmodule
