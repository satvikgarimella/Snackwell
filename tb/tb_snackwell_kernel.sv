`timescale 1ns/1ps
// Simulation host: load flat memory, launch selected warps through the public
// interface, and export results. It does not calculate kernel answers.
module tb_snackwell_kernel;
    logic clk = 0, reset = 1, run_enable = 0, launch_enable = 0;
    logic [1:0] launch_warp_id = 0;
    logic [31:0] launch_pc = 0;
    logic [3:0] launch_active_mask = 0;
    wire [3:0] halted_mask;
    snackwell_sm dut (
        .clk(clk), .reset(reset), .run_enable(run_enable),
        .launch_enable(launch_enable), .launch_warp_id(launch_warp_id),
        .launch_pc(launch_pc), .launch_active_mask(launch_active_mask),
        .debug_write_enable(1'b0), .debug_warp_id(2'd0), .debug_lane_id(2'd0),
        .debug_reg_addr(4'd0), .debug_write_data(32'd0), .halted_mask(halted_mask)
    );
    always #5 clk = ~clk;
    string program_path, data_path, result_path, register_path, trace_path;
    reg [31:0] memory_image [0:1023];
    reg [31:0] registers_image [0:255];
    reg [15:0] active_masks = 16'hffff;
    reg [3:0] warp_mask = 4'hf;
    integer entry_pc = 0, max_cycles = 10000, cycles, handle;
    integer trace_handle = 0, trace_cycles = 0;
    wire [15:0] pe_mac_accept, pe_arithmetic_active;

    // Observe actual handshakes and PE states without changing the DUT.
    for (genvar r = 0; r < 4; r = r + 1) begin : monitor_rows
        for (genvar c = 0; c < 4; c = c + 1) begin : monitor_columns
            assign pe_mac_accept[r*4+c] =
                dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.accept &&
                dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.a_valid &&
                dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.b_valid;
            assign pe_arithmetic_active[r*4+c] =
                dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.state ==
                    dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.MULTIPLY ||
                dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.state ==
                    dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.ADD_START ||
                dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.state ==
                    dut.core.tensor_execute.tensor.array.rows[r].columns[c].pe.ADD_WAIT;
        end
    end

    // Sample before the rising edge updates state: each row describes work
    // accepted on that edge. Initialization and warp launch are not timed.
    always @(posedge clk) begin
        if (!reset && run_enable) begin
            trace_cycles = trace_cycles + 1;
            if (trace_handle != 0)
                $fwrite(trace_handle,
                    "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    trace_cycles, dut.core.issue_valid, dut.core.instruction_execute_accept,
                    dut.core.selected_warp_id, dut.core.selected_pc, dut.core.opcode,
                    dut.core.selected_active_mask, dut.core.dependency_stall, dut.core.resource_stall,
                    dut.core.issue_accept && dut.core.at_reconvergence_point,
                    dut.core.fp_inflight, dut.core.fp_start, dut.core.fp_take, dut.core.wb_fire,
                    dut.core.tensor_execute.state, dut.core.tensor_execute.tensor.array_result_valid,
                    dut.core.tensor_execute.tensor.array.step_index,
                    dut.core.tensor_execute.tensor.array.pe_start,
                    dut.core.tensor_execute.tensor.array.step_advance,
                    pe_mac_accept, pe_arithmetic_active, dut.core.tensor_take,
                    dut.core.wb_fire && dut.core.wb_is_tensor,
                    dut.core.shared_inflight, dut.core.shared_mem.servicing,
                    dut.core.shared_start, dut.core.shared_response_valid,
                    dut.core.shared_response_ready, dut.core.shared_bank_conflict,
                    dut.core.tensor_start);
        end
    end

    initial begin
        if (!$value$plusargs("PROGRAM=%s", program_path) ||
            !$value$plusargs("DATA=%s", data_path) || !$value$plusargs("RESULT=%s", result_path))
            $fatal(1, "Kernel runner requires +PROGRAM, +DATA, and +RESULT");
        if ($value$plusargs("ACTIVE_MASKS=%h", active_masks)) begin end
        if ($value$plusargs("WARP_MASK=%h", warp_mask)) begin end
        if ($value$plusargs("ENTRY=%d", entry_pc)) begin end
        if ($value$plusargs("MAX_CYCLES=%d", max_cycles)) begin end
        if (max_cycles < 1 || max_cycles > 1000000) $fatal(1, "Invalid cycle limit");
        if (warp_mask == 0) $fatal(1, "At least one warp must be launched");
        if ($value$plusargs("TRACE=%s", trace_path)) begin
            trace_handle = $fopen(trace_path, "w");
            if (trace_handle == 0) $fatal(1, "Cannot open kernel trace: %s", trace_path);
            $fwrite(trace_handle, "cycle,issue_valid,instruction_accepted,warp,pc,opcode,active_mask,dependency_stall,resource_stall,reconvergence_accepted,fp_inflight,fp_start,fp_take,wb_commit,tensor_state,array_result_valid,array_step,array_step_start,array_step_advance,pe_mac_accept,pe_arithmetic_active,tensor_row_take,tensor_row_commit,shared_inflight,shared_servicing,shared_start,shared_response_valid,shared_response_ready,shared_bank_conflict,tensor_start\n");
        end
        handle = $fopen(data_path, "r");
        if (handle == 0) $fatal(1, "Cannot open kernel input: %s", data_path);
        $fclose(handle);
        $readmemh(data_path, memory_image);
        repeat (3) @(negedge clk);
        reset = 0;
        for (integer row = 0; row < 256; row = row + 1) begin
            dut.core.shared_mem.bank0[row] = memory_image[row*4];
            dut.core.shared_mem.bank1[row] = memory_image[row*4+1];
            dut.core.shared_mem.bank2[row] = memory_image[row*4+2];
            dut.core.shared_mem.bank3[row] = memory_image[row*4+3];
        end
        for (integer warp = 0; warp < 4; warp = warp + 1) begin
            if (warp_mask[warp]) begin
                @(negedge clk);
                launch_enable = 1; launch_warp_id = warp;
                launch_active_mask = active_masks[warp*4 +: 4]; launch_pc = entry_pc;
                @(negedge clk);
                launch_enable = 0;
            end
        end
        run_enable = 1;
        cycles = 0;
        while ((halted_mask & warp_mask) != warp_mask && cycles < max_cycles) begin
            @(negedge clk);
            cycles = cycles + 1;
        end
        if ((halted_mask & warp_mask) !== warp_mask) $fatal(1, "Kernel timeout after %0d cycles", cycles);
        if (trace_cycles != cycles) $fatal(1, "Trace and execution cycle counts disagree");
        if (dut.core.shared_inflight || dut.core.fp_inflight || dut.core.tensor_busy || dut.core.wb_valid)
            $fatal(1, "Kernel halted before all work completed");
        run_enable = 0;
        if (trace_handle != 0) $fclose(trace_handle);
        for (integer row = 0; row < 256; row = row + 1) begin
            memory_image[row*4]   = dut.core.shared_mem.bank0[row];
            memory_image[row*4+1] = dut.core.shared_mem.bank1[row];
            memory_image[row*4+2] = dut.core.shared_mem.bank2[row];
            memory_image[row*4+3] = dut.core.shared_mem.bank3[row];
        end
        $writememh(result_path, memory_image);
        if ($value$plusargs("REGISTERS=%s", register_path)) begin
            for (integer warp = 0; warp < 4; warp = warp + 1)
                for (integer lane = 0; lane < 4; lane = lane + 1)
                    for (integer rd = 0; rd < 16; rd = rd + 1)
                        registers_image[(warp*4+lane)*16+rd] = dut.core.regfile.registers[warp][lane][rd];
            $writememh(register_path, registers_image);
        end
        $display("PASS kernel completed in %0d cycles", cycles);
        $finish;
    end
endmodule
