`timescale 1ns/1ps
module tb_tensor_core;
    logic clk = 0, reset = 1, enable = 1;
    logic load_valid = 0, load_matrix_b = 0, start_valid = 0, result_ready = 0;
    logic [3:0] load_index = 0;
    logic [31:0] load_data = 0;
    wire load_ready, start_ready, busy, result_valid, result_last;
    wire [3:0] result_index;
    wire [31:0] result_data;
    tensor_core dut (.*);
    always #5 clk = ~clk;

    reg [31:0] source_a[0:15], source_b[0:15], expected[0:15];
    reg [31:0] held_data, word_value;
    reg held_last;
    integer fd, scan, word_scan, case_id, cases = 0, cycles, index;
    string vector_path;

    task automatic load_word(input logic matrix_b, input integer address, input logic [31:0] value);
        @(negedge clk);
        if (!load_ready) $fatal(1, "Tile loader not ready");
        load_valid = 1; load_matrix_b = matrix_b; load_index = address[3:0]; load_data = value;
        @(negedge clk);
        load_valid = 0;
    endtask

    task automatic load_tile;
        begin
            // A duplicate word must not let an incomplete tile launch.
            load_word(0, 0, source_a[0]);
            load_word(0, 0, source_a[0]);
            if (cases == 0)
                repeat (16) begin
                    load_word(0, 0, source_a[0]);
                    load_word(1, 0, source_b[0]);
                end
            @(negedge clk); start_valid = 1;
            @(negedge clk); start_valid = 0;
            if (busy || start_ready) $fatal(1, "Incomplete tile was launched");
            // Permuted writes exercise row-major addressing and both banks.
            for (integer i = 0; i < 16; i = i + 1) begin
                index = (i*5 + case_id) % 16;
                load_word(1, index, source_b[index]);
                load_word(0, index, source_a[index]);
            end
            #1;
            if (!start_ready) $fatal(1, "Fully loaded tiles cannot launch");
        end
    endtask

    task automatic run_and_drain;
        begin
            // Load wins a simultaneous load/start. Update one valid word and
            // require start to handshake only after that update is captured.
            @(negedge clk);
            load_valid = 1; load_matrix_b = 0; load_index = 0; load_data = source_a[0];
            start_valid = 1; enable = 0;
            #1;
            if (start_ready) $fatal(1, "Ambiguous simultaneous load/start accepted");
            @(negedge clk);
            if (busy) $fatal(1, "Load did not take priority");
            load_valid = 0;
            @(negedge clk);
            start_valid = 0;
            if (!busy || load_ready || start_ready) $fatal(1, "Tensor start handshake failed");
            // Writes and starts presented during compute must be rejected.
            load_valid = 1; load_data = 32'hdeadbeef; start_valid = 1;
            repeat (3) @(negedge clk);
            enable = 1;
            cycles = 0;
            while (!result_valid && cycles < 6000) begin
                @(negedge clk);
                if (load_ready || start_ready) $fatal(1, "Accepted work while tensor core busy");
                cycles = cycles + 1;
            end
            if (!result_valid) $fatal(1, "Tensor timeout on case %0d", case_id);
            load_valid = 0; start_valid = 0;
            enable = 0; // Completed results can still drain while paused.
            for (integer i = 0; i < 16; i = i + 1) begin
                if (!result_valid || result_index !== i[3:0] || result_last !== (i == 15))
                    $fatal(1, "Incorrect result stream framing");
                if (expected[i][30:23] == 255 && expected[i][22:0] != 0) begin
                    if (result_data[30:23] !== 8'hff || (|result_data[22:0]) !== 1'b1)
                        $fatal(1, "Expected NaN from tensor output");
                end else if (result_data !== expected[i])
                    $fatal(1, "Tensor case %0d word %0d got=%h expected=%h", case_id, i, result_data, expected[i]);
                held_data = result_data;
                held_last = result_last;
                repeat (1 + (i + case_id) % 4) begin
                    @(negedge clk);
                    if (!result_valid || result_data !== held_data || result_index !== i[3:0] ||
                        result_last !== held_last || load_ready || start_ready || !busy)
                        $fatal(1, "Backpressure corrupted result stream");
                end
                result_ready = 1;
                @(negedge clk);
                result_ready = 0;
            end
            if (busy || result_valid || !load_ready || start_ready || result_index != 0)
                $fatal(1, "Tensor job failed to retire or stale tiles remained launchable");
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
                if (i < 16) source_a[i] = word_value;
                else if (i < 32) source_b[i-16] = word_value;
                else expected[i-32] = word_value;
            end
            load_tile();
            run_and_drain();
            cases = cases + 1;
            scan = $fscanf(fd, "%h", case_id);
        end
        if (!$feof(fd) || cases == 0) $fatal(1, "Malformed/empty vector file");
        $fclose(fd);
        $display("PASS: %0d word-loaded matrix jobs with permuted/duplicate writes and ordered results", cases);
        $display("PASS: partial-load rejection, busy rejection, simultaneous load/start, and per-word backpressure");

        // Reset during a held result must drop the entire old result stream.
        load_tile();
        @(negedge clk); start_valid = 1;
        @(negedge clk); start_valid = 0;
        cycles = 0;
        while (!result_valid && cycles < 6000) begin
            @(negedge clk); cycles = cycles + 1;
        end
        if (!result_valid) $fatal(1, "Reset setup timeout");
        result_ready = 1;
        repeat (3) @(negedge clk);
        result_ready = 0; reset = 1;
        repeat (3) @(negedge clk);
        reset = 0;
        #1;
        if (result_valid || busy || start_ready || !load_ready || result_index != 0)
            $fatal(1, "Reset left stale tensor state");
        load_tile();
        run_and_drain();
        $display("PASS: reset discards partial result streams and allows a fresh job");
        $display("ALL TENSOR CORE TESTS PASSED");
        $finish;
    end
    initial begin
        #15000000;
        $fatal(1, "Tensor core global timeout");
    end
endmodule
