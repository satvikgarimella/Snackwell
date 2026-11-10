`timescale 1ns/1ps
`include "tb/support/global_memory_model.sv"
// Program and shared data enter through the public host bus. The independent
// external RAM is initialized by the test host; the SM performs all arithmetic.
module tb_snackwell_global_memory;
    import snackwell_pkg::*;
    logic clk = 0, reset = 1;
    logic request_valid = 0, request_write = 0, response_ready = 0;
    logic [31:0] request_address = 0, request_data = 0;
    wire request_ready, response_valid, response_error, busy, done, fault;
    wire [31:0] response_data;
    wire global_request_valid, global_request_ready, global_request_write;
    wire [31:0] global_request_address, global_request_data;
    wire global_response_valid, global_response_ready, global_response_error;
    wire [31:0] global_response_data;
    logic hold_requests = 0, hold_responses = 0;
    integer latency = 2, reads, writes, responses, request_stalls;
    integer max_cycles = 20000, warp_selection = 1, lane_selection = 65535, expected_fault = 0;
    logic [31:0] image [0:1023], shared_after [0:1023];
    logic [31:0] register_after [0:255];
    logic [31:0] value, status, cycles, errors, copy_errors;
    string program_path, data_path, global_path, output_path, global_output_path, metrics_path, trace_path;
    string register_path;
    integer trace_file = 0, metrics_file;
    integer independent_issues = 0, dependent_stalls = 0;
    integer copy_reads = 0, copy_writes = 0, overlap_cycles = 0;
    wire arithmetic_active = |dut.core.tensor_execute.tensor.pe_arithmetic_busy;
    wire [15:0] nvfp4_mac, fp32_tensor_multiply;
    for (genvar r = 0; r < 4; r = r + 1) begin : trace_rows
        for (genvar c = 0; c < 4; c = c + 1) begin : trace_columns
            assign nvfp4_mac[r*4+c] = dut.core.tensor_execute.tensor.rows[r].columns[c].pe.accept &&
                dut.core.tensor_execute.tensor.rows[r].columns[c].pe.a_valid &&
                dut.core.tensor_execute.tensor.rows[r].columns[c].pe.b_valid &&
                dut.core.tensor_execute.tensor.rows[r].columns[c].pe.nvfp4_mode;
            assign fp32_tensor_multiply[r*4+c] = dut.core.tensor_execute.tensor.rows[r].columns[c].pe.multiplier.start;
        end
    end
    snackwell_device dut (.*);
    global_memory_model ram (
        .clk(clk), .reset(reset), .request_valid(global_request_valid),
        .request_ready(global_request_ready), .request_write(global_request_write),
        .request_address(global_request_address), .request_data(global_request_data),
        .response_valid(global_response_valid), .response_ready(global_response_ready),
        .response_error(global_response_error), .response_data(global_response_data),
        .hold_requests(hold_requests), .hold_responses(hold_responses), .latency(latency),
        .reads(reads), .writes(writes), .responses(responses), .request_stalls(request_stalls)
    );
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!reset && dut.core.run_enable) begin
            if (dut.core.global_busy && dut.core.instruction_execute_accept && dut.core.is_integer_op)
                independent_issues = independent_issues + 1;
            if (dut.core.global_busy && dut.core.dependency_stall)
                dependent_stalls = dependent_stalls + 1;
            if (dut.core.copy_global_valid && dut.core.copy_global_ready) copy_reads = copy_reads + 1;
            if (dut.core.copy_shared_valid && dut.core.copy_shared_ready) copy_writes = copy_writes + 1;
            if (dut.core.copy_busy && arithmetic_active) overlap_cycles = overlap_cycles + 1;
            if (trace_file != 0)
                $fwrite(trace_file, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    dut.cycles, dut.core.instruction_execute_accept, dut.core.selected_warp_id,
                    dut.core.opcode, dut.core.dependency_stall, dut.core.resource_stall,
                    dut.core.global_busy, global_request_valid && global_request_ready,
                    global_request_write, global_request_address,
                    global_response_valid && global_response_ready, global_response_error,
                    dut.core.tmem_array_valid && dut.core.tmem_array_ready,
                    dut.core.copy_busy, dut.core.copy_global_valid && dut.core.copy_global_ready,
                    dut.core.copy_shared_valid && dut.core.copy_shared_ready, arithmetic_active,
                    dut.core.copy_start, dut.core.async_copy.job_slot, dut.core.selected_pc,
                    nvfp4_mac, fp32_tensor_multiply);
        end
    end

    task automatic transaction(input bit write_op, input logic [31:0] address, data,
                               input bit expect_error, output logic [31:0] result);
        integer waits;
        @(negedge clk);
        request_valid = 1; request_write = write_op; request_address = address; request_data = data;
        waits = 0;
        while (!request_ready && waits < 50) begin @(negedge clk); waits = waits + 1; end
        if (!request_ready) $fatal(1, "Host request timeout");
        @(negedge clk); request_valid = 0;
        waits = 0;
        while (!response_valid && waits < 50) begin @(negedge clk); waits = waits + 1; end
        if (!response_valid || response_error !== expect_error)
            $fatal(1, "Host response error for address %h", address);
        result = response_data;
        response_ready = 1;
        @(negedge clk); response_ready = 0;
    endtask
    task automatic put(input logic [31:0] address, data);
        logic [31:0] ignored;
        transaction(1, address, data, 0, ignored);
    endtask
    task automatic complete(input bit expect_fault);
        integer waits;
        waits = 0;
        while (busy && waits < max_cycles + 100) begin @(negedge clk); waits = waits + 1; end
        if (busy || fault !== expect_fault || done !== !expect_fault)
            $fatal(1, "Incorrect device completion busy=%b fault=%b done=%b", busy, fault, done);
        if (ram.pending || global_request_valid || global_response_valid || reads + writes != responses)
            $fatal(1, "Device finished with undrained external requests");
    endtask
    task automatic reset_link;
        @(negedge clk); reset = 1; request_valid = 0; response_ready = 0;
        repeat (3) @(negedge clk);
        reset = 0;
        hold_requests = 0; hold_responses = 0;
    endtask

    task automatic control_tests;
        // Abort with a presented request, an accepted request, and a watchdog.
        for (integer mode = 0; mode < 3; mode = mode + 1) begin
            reset_link();
            ram.memory[1024] = 0;
            put(32'h1000, {OP_ADDI, 4'd1, 4'd0, 4'd0, 14'd37});
            put(32'h1004, {OP_ST_GLOBAL, 4'd0, 4'd0, 4'd1, 14'd4096});
            put(32'h1008, {OP_EXIT, 26'd0});
            put(32'h10, 1);
            put(32'h18, mode == 2 ? 20 : 1000);
            hold_requests = mode == 0;
            hold_responses = mode != 0;
            put(32'h08, 1);
            wait (global_request_valid);
            if (mode != 0) wait (ram.pending);
            if (mode != 2) put(32'h08, 2);
            else repeat (30) @(negedge clk);
            repeat (10) begin
                @(negedge clk);
                if (!busy || !fault || done) $fatal(1, "Abort discarded outstanding transaction");
            end
            transaction(1, 32'h08, 1, 1, value); // Relaunch must wait for draining.
            hold_requests = 0; hold_responses = 0;
            complete(1);
            if (writes != 4 || ram.memory[1024] !== 37)
                $fatal(1, "Abort failed to finish exactly the captured store vector");

            // Relaunch uses the preserved global data and fresh core state.
            put(32'h1000, {OP_LD_GLOBAL, 4'd2, 4'd0, 4'd0, 14'd4096});
            put(32'h1004, {OP_ST_SHARED, 4'd0, 4'd0, 4'd2, 14'd0});
            put(32'h18, 1000);
            put(32'h08, 1);
            complete(0);
            transaction(0, 32'h2000, 0, 0, value);
            if (value !== 37) $fatal(1, "Stale result after abort/relaunch");
        end

        reset_link();
        hold_responses = 1;
        put(32'h08, 1);
        wait (ram.pending);
        reset_link(); // Simultaneously cancel both ends of the external link.
        if (global_request_valid || global_response_valid || ram.pending || busy)
            $fatal(1, "Link reset retained a request");
        put(32'h10, 1);
        put(32'h08, 1);
        complete(0);
        transaction(0, 32'h2000, 0, 0, value);
        if (value !== 37) $fatal(1, "Link reset lost external data");
        $display("PASS global device abort, watchdog, drain, reset and relaunch");
    endtask

    task automatic async_control_tests;
        for (integer mode = 0; mode < 3; mode = mode + 1) begin
            reset_link();
            for (integer i = 0; i < 8; i = i + 1) ram.memory[4096+i] = 100+i;
            put(32'h1000, {OP_TID, 4'd12, 22'd0});
            put(32'h1004, {OP_ADD, 4'd12, 4'd12, 4'd12, 14'd0});
            put(32'h1008, {OP_ADD, 4'd12, 4'd12, 4'd12, 14'd0});
            put(32'h100c, {OP_LD_SHARED, 4'd6, 4'd12, 4'd0, 14'd768});
            put(32'h1010, {OP_CPASYNC, 4'd8, 4'd6, 4'd0, 14'd2});
            put(32'h1014, {OP_CPWAIT, 4'd0, 4'd8, 4'd0, 14'd0});
            put(32'h1018, {OP_CPRELEASE, 26'd0});
            put(32'h101c, {OP_EXIT, 26'd0});
            put(32'h2300, 32'h4000);
            put(32'h2304, 16);
            put(32'h2308, 32'h0402);
            put(32'h230c, 32'h00010080);
            for (integer i = 0; i < 8; i = i + 1) put(32'h2080+i*4, 0);
            put(32'h10, 1);
            put(32'h18, mode == 2 ? 35 : 1000);
            hold_requests = mode == 0;
            hold_responses = mode != 0;
            put(32'h08, 1);
            wait (global_request_valid);
            if (mode != 0) wait (ram.pending);
            if (mode != 2) put(32'h08, 2);
            else repeat (45) @(negedge clk);
            repeat (10) begin
                @(negedge clk);
                if (!busy || !fault || done) $fatal(1, "Abort cancelled a live async copy");
            end
            transaction(1, 32'h08, 1, 1, value);
            hold_requests = 0; hold_responses = 0;
            complete(1);
            if (reads != 8 || writes != 0) $fatal(1, "Async abort did not drain exactly one rectangle");
            for (integer i = 0; i < 8; i = i + 1) begin
                transaction(0, 32'h2080+i*4, 0, 0, value);
                if (value !== 32'(100+i)) $fatal(1, "Async abort lost a shared write");
            end
            put(32'h18, 1000);
            put(32'h08, 1);
            complete(0);
            transaction(0, 32'h24, 0, 0, value);
            if (value != 0) $fatal(1, "Async relaunch retained errors/tickets");
        end
        hold_responses = 1;
        put(32'h08, 1);
        wait (ram.pending);
        reset_link();
        if (ram.pending || global_response_valid || global_request_valid || busy)
            $fatal(1, "Async link reset retained a transaction");
        put(32'h10, 1);
        put(32'h08, 1);
        complete(0);
        $display("PASS async copy abort, watchdog, shared-write drain, link reset and relaunch");
    endtask

    initial begin
        for (integer i = 0; i < 16384; i = i + 1) ram.memory[i] = 32'hdeadbeef;
        reset_link();
        if ($value$plusargs("PROGRAM=%s", program_path)) begin
            if (!$value$plusargs("DATA=%s", data_path) || !$value$plusargs("GLOBAL=%s", global_path) ||
                !$value$plusargs("OUTPUT=%s", output_path) ||
                !$value$plusargs("GLOBAL_OUTPUT=%s", global_output_path) ||
                !$value$plusargs("METRICS=%s", metrics_path)) $fatal(1, "Missing global runner path");
            if ($value$plusargs("WARP_MASK=%d", warp_selection)) begin end
            if ($value$plusargs("ACTIVE_MASKS=%d", lane_selection)) begin end
            if ($value$plusargs("MAX_CYCLES=%d", max_cycles)) begin end
            if ($value$plusargs("LATENCY=%d", latency)) begin end
            if ($value$plusargs("EXPECT_FAULT=%d", expected_fault)) begin end
            if ($value$plusargs("TRACE=%s", trace_path)) begin
                trace_file = $fopen(trace_path, "w");
                if (trace_file == 0) $fatal(1, "Cannot open global trace");
                $fwrite(trace_file, "cycle,accepted,warp,opcode,dependency_stall,resource_stall,global_busy,request,write,address,response,error,matrix_job,copy_busy,copy_read,copy_write,arithmetic_active,copy_command,copy_slot,pc,nvfp4_mac,fp32_tensor_multiply\n");
            end
            $readmemh(program_path, image);
            for (integer i = 0; i < 1024; i = i + 1) put(32'h1000 + i*4, image[i]);
            $readmemh(data_path, image);
            for (integer i = 0; i < 1024; i = i + 1) put(32'h2000 + i*4, image[i]);
            $readmemh(global_path, ram.memory);
            put(32'h10, warp_selection);
            put(32'h14, lane_selection);
            put(32'h18, max_cycles);
            put(32'h08, 1);
            complete(expected_fault != 0);
            if (trace_file != 0) $fclose(trace_file);
            transaction(0, 32'h1c, 0, 0, cycles);
            transaction(0, 32'h20, 0, 0, errors);
            transaction(0, 32'h24, 0, 0, copy_errors);
            for (integer i = 0; i < 1024; i = i + 1)
                transaction(0, 32'h2000 + i*4, 0, 0, shared_after[i]);
            $writememh(output_path, shared_after);
            $writememh(global_output_path, ram.memory);
            if ($value$plusargs("REGISTERS=%s", register_path)) begin
                for (integer i = 0; i < 256; i = i + 1)
                    register_after[i] = dut.core.regfile.registers[i/64][(i/16)%4][i%16];
                $writememh(register_path, register_after);
            end
            metrics_file = $fopen(metrics_path, "w");
            if (metrics_file == 0) $fatal(1, "Cannot open metrics");
            $fdisplay(metrics_file, "%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                cycles, reads, writes, responses, request_stalls, independent_issues, dependent_stalls, errors,
                copy_errors, copy_reads, copy_writes, overlap_cycles);
            $fclose(metrics_file);
            $display("PASS global-memory kernel: %0d cycles, %0d reads, %0d writes", cycles, reads, writes);
        end else begin control_tests(); async_control_tests(); end
        $finish;
    end
    initial begin #100000000; $fatal(1, "Global device timeout"); end
endmodule
