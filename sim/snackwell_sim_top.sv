// Simulation-only memory access. The SM executes every instruction normally;
// these functions replace testbench file I/O, not execution or arithmetic.
`include "tb/support/global_memory_model.sv"
module snackwell_sim_top (
    input logic clk, reset, run_enable, launch_enable,
    input logic global_enable, resident_enable,
    input integer global_latency,
    input logic [1:0] launch_warp_id,
    input logic [31:0] launch_pc,
    input logic [3:0] launch_active_mask,
    output logic [3:0] halted_mask,
    output logic pending,
    output integer global_reads, global_writes, global_responses, global_stalls,
    output integer copy_reads, copy_writes, overlap_cycles, narrow_macs, wide_multiplies,
    output logic [11:0] global_errors,
    output logic [31:0] copy_errors
);
    wire req_valid, req_ready, req_write, rsp_valid, rsp_ready, rsp_error;
    wire [31:0] req_address, req_data, rsp_data;
    // Resident experiments use 1 MiB of simulated external RAM. The original
    // global transport retains its 64 KiB address limit. Neither is on-chip RAM.
    global_memory_model #(.WORDS(262144)) ram (
        .clk(clk), .reset(reset), .request_valid(req_valid && global_enable),
        .request_ready(req_ready), .request_write(req_write),
        .request_address(!resident_enable && req_address >= 65536 ? 32'hffffffff : req_address),
        .request_data(req_data), .response_valid(rsp_valid), .response_ready(rsp_ready),
        .response_error(rsp_error), .response_data(rsp_data),
        .hold_requests(1'b0), .hold_responses(1'b0), .latency(global_latency),
        .reads(global_reads), .writes(global_writes), .responses(global_responses),
        .request_stalls(global_stalls)
    );
    snackwell_sm dut (
        .global_request_valid(req_valid), .global_request_ready(global_enable ? req_ready : 1'b1),
        .global_request_write(req_write), .global_request_address(req_address), .global_request_data(req_data),
        .global_response_valid(global_enable ? rsp_valid : 1'b1), .global_response_ready(rsp_ready),
        .global_response_error(global_enable ? rsp_error : 1'b1), .global_response_data(rsp_data),

        .clk(clk), .reset(reset), .run_enable(run_enable),
        .launch_enable(launch_enable), .launch_warp_id(launch_warp_id),
        .launch_pc(launch_pc), .launch_active_mask(launch_active_mask),
        .debug_write_enable(1'b0), .debug_warp_id(2'd0), .debug_lane_id(2'd0),
        .debug_reg_addr(4'd0), .debug_write_data(32'd0), .halted_mask(halted_mask)
    );
    assign pending = dut.core.copy_busy || dut.core.copy_row_valid || dut.core.copy_warp_pending != 0 ||
                     dut.core.global_busy || dut.core.shared_inflight || dut.core.fp_inflight ||
                     dut.core.tensor_busy || dut.core.wb_valid || dut.core.tmem_busy ||
                     dut.core.tmem_row_valid || dut.core.tmem_warp_pending != 0 ||
                     (global_enable && (ram.pending || rsp_valid || req_valid));
    assign global_errors = dut.core.global_status;
    assign copy_errors = dut.core.copy_status;
    wire [15:0] narrow_accept, wide_start;
    for (genvar r = 0; r < 4; r = r + 1) begin : trace_rows
        for (genvar c = 0; c < 4; c = c + 1) begin : trace_columns
            assign narrow_accept[r*4+c] = dut.core.tensor_execute.tensor.rows[r].columns[c].pe.accept &&
                dut.core.tensor_execute.tensor.rows[r].columns[c].pe.a_valid &&
                dut.core.tensor_execute.tensor.rows[r].columns[c].pe.b_valid &&
                dut.core.tensor_execute.tensor.rows[r].columns[c].pe.nvfp4_mode;
            assign wide_start[r*4+c] = dut.core.tensor_execute.tensor.rows[r].columns[c].pe.multiplier.start;
        end
    end
    always @(posedge clk) begin
        if (reset) begin
            copy_reads <= 0; copy_writes <= 0; overlap_cycles <= 0;
            narrow_macs <= 0; wide_multiplies <= 0;
        end else if (run_enable) begin
            if (dut.core.copy_global_valid && dut.core.copy_global_ready) copy_reads <= copy_reads + 1;
            if (dut.core.copy_shared_valid && dut.core.copy_shared_ready) copy_writes <= copy_writes + 1;
            if (dut.core.copy_busy && |dut.core.tensor_execute.tensor.pe_arithmetic_busy)
                overlap_cycles <= overlap_cycles + 1;
            narrow_macs <= narrow_macs + $countones(narrow_accept);
            wide_multiplies <= wide_multiplies + $countones(wide_start);
        end
    end

    function void load_global_word(input int index, input logic [31:0] data);
        /* verilator public */
        ram.memory[index] = data;
    endfunction

    function logic [31:0] read_global_word(input int index);
        /* verilator public */
        return ram.memory[index];
    endfunction

    function void load_word(input bit program_word, input int index, input logic [31:0] data);
        /* verilator public */
        if (program_word) dut.core.imem.memory[index] = data;
        else case (index % 4)
            0: dut.core.shared_mem.bank0[index/4] = data;
            1: dut.core.shared_mem.bank1[index/4] = data;
            2: dut.core.shared_mem.bank2[index/4] = data;
            3: dut.core.shared_mem.bank3[index/4] = data;
        endcase
    endfunction

    function logic [31:0] read_word(input int index);
        /* verilator public */
        case (index % 4)
            0: return dut.core.shared_mem.bank0[index/4];
            1: return dut.core.shared_mem.bank1[index/4];
            2: return dut.core.shared_mem.bank2[index/4];
            3: return dut.core.shared_mem.bank3[index/4];
        endcase
    endfunction

    function logic [31:0] read_register(input int index);
        /* verilator public */
        return dut.core.regfile.registers[index/64][(index/16)%4][index%16];
    endfunction
endmodule
