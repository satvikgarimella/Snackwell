`timescale 1ns/1ps
// No hierarchical reads/writes: the simulated host uses only the device bus.
module tb_snackwell_device;
    import snackwell_pkg::*;
    logic clk = 0, reset = 1;
    logic request_valid = 0, request_write = 0, response_ready = 0;
    logic [31:0] request_address = 0, request_data = 0;
    wire request_ready, response_valid, response_error, busy, done, fault;
    wire [31:0] response_data;
    logic [31:0] value, status, cycles, image [0:1023], output_image [0:1023];
    string program_path, data_path, output_path;
    integer warp_selection = 15, lane_selection = 65535, entry = 0;
    integer max_cycles = 10000;
    integer cycles_file;
    string cycles_path;

    snackwell_device dut (.*);
    always #5 clk = ~clk;

    task automatic transaction(
        input bit write_request, input logic [31:0] address, data,
        input bit expected_error, input integer hold_cycles,
        output logic [31:0] result
    );
        integer waits;
        logic held_error;
        @(negedge clk);
        request_valid = 1;
        request_write = write_request;
        request_address = address;
        request_data = data;
        waits = 0;
        while (!request_ready && waits < 50) begin
            @(negedge clk);
            waits = waits + 1;
        end
        if (!request_ready) $fatal(1, "Host request timeout");
        @(negedge clk);
        request_valid = 0;
        waits = 0;
        while (!response_valid && waits < 50) begin
            @(negedge clk);
            waits = waits + 1;
        end
        if (!response_valid) $fatal(1, "Host response timeout");
        result = response_data;
        held_error = response_error;
        if (response_error !== expected_error)
            $fatal(1, "Address %h error=%b expected=%b", address, response_error, expected_error);
        // Change all request fields while the response is stalled. Neither
        // memory responses nor program reads may follow the new address.
        request_address = 32'hffffffff;
        request_data = 32'hbadbadba;
        request_write = !write_request;
        request_valid = 1;
        repeat (hold_cycles) begin
            @(negedge clk);
            if (!response_valid || response_data !== result || response_error !== held_error || request_ready)
                $fatal(1, "Response changed under backpressure");
        end
        request_valid = 0;
        response_ready = 1;
        @(negedge clk);
        response_ready = 0;
    endtask

    task automatic write_word(input logic [31:0] address, data);
        logic [31:0] ignored;
        transaction(1, address, data, 0, 0, ignored);
    endtask

    task automatic wait_for_completion(input bit expect_fault);
        integer attempts;
        attempts = 0;
        status = 1;
        while (status[0] && attempts < max_cycles + 100) begin
            transaction(0, 32'h04, 0, 0, 0, status);
            attempts = attempts + 1;
        end
        if (status[0] || status[2] !== expect_fault || status[1] !== !expect_fault)
            $fatal(1, "Incorrect completion status %h", status);
    endtask

    task automatic control_tests;
        // Both memory windows include their last word, and invalid addresses
        // must not wrap into a valid word or silently truncate configuration.
        transaction(0, 0, 0, 0, 3, value);
        if (value !== 32'h534e4b31) $fatal(1, "Device ID mismatch");
        write_word(32'h1ffc, {OP_EXIT, 26'd0});
        transaction(0, 32'h1ffc, 0, 0, 5, value);
        if (value !== {OP_EXIT, 26'd0}) $fatal(1, "Program readback mismatch");
        for (integer bank = 0; bank < 4; bank = bank + 1) begin
            write_word(32'h2ff0 + bank*4, 32'habcd0000 + bank);
            transaction(0, 32'h2ff0 + bank*4, 0, 0, 5, value);
            if (value !== 32'habcd0000 + bank) $fatal(1, "Bank readback mismatch");
        end
        transaction(1, 32'h1002, 0, 1, 2, value);
        transaction(1, 32'h2001, 0, 1, 2, value);
        transaction(1, 32'h3000, 0, 1, 0, value);
        transaction(0, 32'hffff2000, 0, 1, 0, value);
        transaction(1, 32'h00, 0, 1, 0, value);
        transaction(1, 32'h08, 3, 1, 0, value);
        transaction(1, 32'h0c, 2, 1, 0, value);
        transaction(1, 32'h0c, 4096, 1, 0, value);
        transaction(1, 32'h10, 0, 1, 0, value);
        transaction(1, 32'h10, 16, 1, 0, value);
        transaction(1, 32'h14, 65536, 1, 0, value);
        transaction(1, 32'h18, 0, 1, 0, value);

        // A branch to itself exercises the watchdog and rejects mutation while
        // the SM runs. Host access resumes after an abort has reset execution.
        write_word(32'h1000, {OP_BEQ, 4'd0, 4'd0, 4'd0, 14'h3fff});
        write_word(32'h10, 1);
        write_word(32'h18, 100);
        write_word(32'h08, 1);
        transaction(1, 32'h1000, {OP_EXIT, 26'd0}, 1, 0, value);
        transaction(1, 32'h2000, 1, 1, 0, value);
        transaction(0, 32'h2000, 0, 1, 0, value);
        transaction(1, 32'h10, 2, 1, 0, value);
        transaction(1, 32'h08, 1, 1, 0, value);
        wait_for_completion(1);
        transaction(0, 32'h1c, 0, 0, 0, value);
        if (value !== 100) $fatal(1, "Watchdog cycle mismatch: %d", value);
        // Explicit abort and external reset cannot leave an old success flag.
        write_word(32'h18, 1000);
        write_word(32'h08, 1);
        write_word(32'h08, 2);
        wait_for_completion(1);

        // Completion requires execution to drain, and each launch resets
        // registers without erasing the previously loaded data/program.
        write_word(32'h1000, {OP_ADDI, 4'd1, 4'd1, 4'd0, 14'd1});
        write_word(32'h1004, {OP_ST_SHARED, 4'd0, 4'd0, 4'd1, 14'd0});
        write_word(32'h1008, {OP_EXIT, 26'd0});
        write_word(32'h14, 1);
        repeat (2) begin
            write_word(32'h08, 1);
            wait_for_completion(0);
            transaction(0, 32'h2000, 0, 0, 3, value);
            if (value !== 1) $fatal(1, "Launch did not reset registers or drain stores");
            transaction(0, 32'h2ffc, 0, 0, 0, value);
            if (value !== 32'habcd0003) $fatal(1, "Launch erased shared memory");
        end
        // Nonzero entry, nonzero warp, and no active lanes still terminate.
        write_word(32'h0c, 4092);
        write_word(32'h10, 8);
        write_word(32'h14, 0);
        write_word(32'h18, 1);
        write_word(32'h08, 1);
        wait_for_completion(0);
        transaction(0, 32'h1c, 0, 0, 0, value);
        if (value !== 1) $fatal(1, "Completion at the cycle limit was not accepted");
        @(negedge clk); reset = 1;
        repeat (2) @(negedge clk);
        reset = 0;
        transaction(0, 32'h04, 0, 0, 0, value);
        if (value !== 0) $fatal(1, "Reset left stale device status");
        transaction(0, 32'h2ffc, 0, 0, 0, value);
        if (value !== 32'habcd0003) $fatal(1, "Reset erased shared memory");

        // Reset while a read response is stalled discards the response and
        // permits a fresh transaction, without clearing the stored word.
        @(negedge clk);
        request_address = 32'h2ffc;
        request_write = 0;
        request_valid = 1;
        @(negedge clk); request_valid = 0;
        repeat (6) @(negedge clk);
        if (!response_valid) $fatal(1, "Missing response before reset");
        reset = 1;
        repeat (2) @(negedge clk);
        if (response_valid) $fatal(1, "Reset did not cancel outstanding response");
        reset = 0;
        transaction(0, 32'h2ffc, 0, 0, 0, value);
        if (value !== 32'habcd0003) $fatal(1, "Read failed after transaction reset");
        $display("PASS device control, memory, backpressure, watchdog, abort and relaunch");
    endtask

    initial begin
        repeat (3) @(negedge clk);
        reset = 0;
        if ($value$plusargs("DEVICE_PROGRAM=%s", program_path)) begin
            if (!$value$plusargs("DEVICE_DATA=%s", data_path) ||
                !$value$plusargs("DEVICE_OUTPUT=%s", output_path) ||
                !$value$plusargs("DEVICE_CYCLES=%s", cycles_path))
                $fatal(1, "Device workload requires data/output/cycles paths");
            if ($value$plusargs("WARP_MASK=%d", warp_selection)) begin end
            if ($value$plusargs("ACTIVE_MASKS=%d", lane_selection)) begin end
            if ($value$plusargs("ENTRY=%d", entry)) begin end
            if ($value$plusargs("MAX_CYCLES=%d", max_cycles)) begin end
            $readmemh(program_path, image);
            for (integer i = 0; i < 1024; i = i + 1) write_word(32'h1000 + i*4, image[i]);
            for (integer i = 0; i < 1024; i = i + 1) begin
                transaction(0, 32'h1000 + i*4, 0, 0, i % 3, value);
                if (value !== image[i]) $fatal(1, "Program word %d mismatch", i);
            end
            $readmemh(data_path, image);
            for (integer i = 0; i < 1024; i = i + 1) write_word(32'h2000 + i*4, image[i]);
            write_word(32'h0c, entry);
            write_word(32'h10, warp_selection);
            write_word(32'h14, lane_selection);
            write_word(32'h18, max_cycles);
            write_word(32'h08, 1);
            wait_for_completion(0);
            transaction(0, 32'h1c, 0, 0, 0, cycles);
            for (integer i = 0; i < 1024; i = i + 1)
                transaction(0, 32'h2000 + i*4, 0, 0, i % 4, output_image[i]);
            $writememh(output_path, output_image);
            cycles_file = $fopen(cycles_path, "w");
            if (cycles_file == 0) $fatal(1, "Cannot write cycles");
            $fdisplay(cycles_file, "%0d", cycles);
            $fclose(cycles_file);
            $display("PASS device kernel completed in %0d cycles", cycles);
        end else control_tests();
        $finish;
    end

    initial begin
        #100000000;
        $fatal(1, "Device test timeout");
    end
endmodule
