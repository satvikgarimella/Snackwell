// Simulation-only memory access. The SM executes every instruction normally;
// these functions replace testbench file I/O, not execution or arithmetic.
module snackwell_sim_top (
    input logic clk, reset, run_enable, launch_enable,
    input logic [1:0] launch_warp_id,
    input logic [31:0] launch_pc,
    input logic [3:0] launch_active_mask,
    output logic [3:0] halted_mask,
    output logic pending
);
    snackwell_sm dut (
        .clk(clk), .reset(reset), .run_enable(run_enable),
        .launch_enable(launch_enable), .launch_warp_id(launch_warp_id),
        .launch_pc(launch_pc), .launch_active_mask(launch_active_mask),
        .debug_write_enable(1'b0), .debug_warp_id(2'd0), .debug_lane_id(2'd0),
        .debug_reg_addr(4'd0), .debug_write_data(32'd0), .halted_mask(halted_mask)
    );
    assign pending = dut.core.shared_inflight || dut.core.fp_inflight ||
                     dut.core.tensor_busy || dut.core.wb_valid;

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
