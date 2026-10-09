`timescale 1ns/1ps
module tb_snackwell_sm_memory;
    import snackwell_pkg::*;
    logic clk = 0, reset = 1, run_enable = 0;
    logic launch_enable = 0;
    logic [1:0] launch_warp_id = 0;
    logic [31:0] launch_pc = 0;
    logic [3:0] launch_active_mask = 0;
    logic debug_write_enable = 0;
    logic [1:0] debug_warp_id = 0, debug_lane_id = 0;
    logic [3:0] debug_reg_addr = 0;
    logic [31:0] debug_write_data = 0;
    wire issue_valid, issue_accept, dependency_stall, resource_stall;
    wire [1:0] selected_warp_id;
    wire [31:0] selected_pc, instruction;
    wire [5:0] opcode;
    wire [3:0] halted_mask, selected_active_mask;
    snackwell_sm dut (
        .clk(clk), .reset(reset), .run_enable(run_enable),
        .launch_enable(launch_enable), .launch_warp_id(launch_warp_id),
        .launch_pc(launch_pc), .launch_active_mask(launch_active_mask),
        .debug_write_enable(debug_write_enable), .debug_warp_id(debug_warp_id),
        .debug_lane_id(debug_lane_id), .debug_reg_addr(debug_reg_addr),
        .debug_write_data(debug_write_data),
        .issue_valid(issue_valid), .issue_accept(issue_accept), .instruction(instruction),
        .selected_warp_id(selected_warp_id), .selected_pc(selected_pc), .opcode(opcode),
        .selected_active_mask(selected_active_mask), .halted_mask(halted_mask),
        .dependency_stall(dependency_stall), .resource_stall(resource_stall)
    );
    always #5 clk = ~clk;

    // Independent flat memory and per-load architectural expectations. The
    // reference advances in instruction acceptance order, not replay order.
    reg [31:0] memory_model [0:1023];
    reg [15:0] pending_load [0:3], completed_load [0:3];
    reg [127:0] expected_load [0:3][0:15];
    reg [3:0] expected_mask [0:3][0:15];
    reg memory_pending, model_is_load;
    integer model_warp, model_rd;
    integer requests, responses, load_writes, independent_issue, conflicts, raw_stalls;
    integer fp_collisions, tensor_collisions, mask_changes;
    integer total_requests = 0;
    integer w, d, word_index;
    reg [31:0] address;
    reg [15:0] access_mask;

    always @(posedge clk) begin
        if (reset) begin
            memory_pending = 0; model_is_load = 0; model_warp = 0; model_rd = 0;
            requests = 0; responses = 0; load_writes = 0;
            independent_issue = 0; conflicts = 0; raw_stalls = 0;
            fp_collisions = 0; tensor_collisions = 0; mask_changes = 0;
            for (integer i = 0; i < 1024; i = i + 1) memory_model[i] = 0;
            for (integer i = 0; i < 4; i = i + 1) begin
                pending_load[i] = 0; completed_load[i] = 0;
            end
        end else begin
            if (dut.core.shared_bank_conflict) conflicts = conflicts + 1;
            if (dependency_stall) raw_stalls = raw_stalls + 1;
            if (dut.core.shared_completion_valid && dut.core.fp_completion_valid)
                fp_collisions = fp_collisions + 1;
            if (dut.core.shared_completion_valid && dut.core.tensor_row_valid)
                tensor_collisions = tensor_collisions + 1;
            if (dut.core.instruction_execute_accept && dut.core.branch_divergent &&
                memory_pending && model_is_load) mask_changes = mask_changes + 1;
            if (dut.core.shared_take && (dut.core.fp_completion_valid || dut.core.tensor_row_valid))
                $fatal(1, "Shared load bypassed higher-priority completion");

            if (dut.core.wb_fire && pending_load[dut.core.wb_warp_id][dut.core.wb_rd_addr]) begin
                w = dut.core.wb_warp_id; d = dut.core.wb_rd_addr;
                if (!completed_load[w][d] || dut.core.wb_write_mask !== expected_mask[w][d] ||
                    dut.core.wb_result !== expected_load[w][d])
                    $fatal(1, "Incorrect/early memory writeback W%0d R%0d got=%h expected=%h mask=%b/%b",
                        w, d, dut.core.wb_result, expected_load[w][d], dut.core.wb_write_mask, expected_mask[w][d]);
                pending_load[w][d] = 0; completed_load[w][d] = 0;
                load_writes = load_writes + 1;
            end
            if (dut.core.shared_response_valid && dut.core.shared_response_ready) begin
                if (!memory_pending) $fatal(1, "Duplicate/unallocated memory completion");
                if (model_is_load) completed_load[model_warp][model_rd] = 1;
                memory_pending = 0; responses = responses + 1;
            end
            if (dut.core.instruction_execute_accept) begin
                w = selected_warp_id; d = instruction[25:22];
                access_mask = (dut.core.uses_rs1 ? (16'h1 << instruction[21:18]) : 0) |
                              (dut.core.uses_rs2 ? (16'h1 << instruction[17:14]) : 0) |
                              (dut.core.writes_rd ? (16'h1 << d) : 0);
                if (dut.core.is_tensor_op)
                    access_mask = (16'hf << instruction[21:18]) |
                                  (16'hf << instruction[17:14]) | (16'hf << d);
                if (|(access_mask & pending_load[w])) $fatal(1, "Issued across a load RAW/WAW hazard");
                if (opcode == OP_EXIT && (pending_load[w] != 0 || (memory_pending && model_warp == w)))
                    $fatal(1, "EXIT before this warp's memory completed");
                if (memory_pending && dut.core.is_integer_op) independent_issue = independent_issue + 1;
                if (opcode == OP_LD_SHARED || opcode == OP_ST_SHARED) begin
                    if (memory_pending) $fatal(1, "Younger memory instruction overtook pending request");
                    memory_pending = 1; model_warp = w; model_rd = d;
                    model_is_load = opcode == OP_LD_SHARED;
                    if (model_is_load) begin
                        pending_load[w][d] = 1;
                        expected_load[w][d] = 0;
                        expected_mask[w][d] = selected_active_mask;
                    end
                    for (integer lane = 0; lane < 4; lane = lane + 1) begin
                        address = dut.core.regfile.registers[w][lane][instruction[21:18]] +
                                  {{18{instruction[13]}}, instruction[13:0]};
                        word_index = (address >> 2) & 1023;
                        if (selected_active_mask[lane]) begin
                            if (model_is_load) expected_load[w][d][lane*32 +: 32] = memory_model[word_index];
                            else memory_model[word_index] = dut.core.regfile.registers[w][lane][instruction[17:14]];
                        end
                    end
                    requests = requests + 1; total_requests = total_requests + 1;
                end
            end
        end
    end

    function automatic [31:0] insn(input logic [5:0] op,
        input logic [3:0] rd, a, b, input integer imm = 0);
        insn = {op, rd, a, b, imm[13:0]};
    endfunction
    function automatic [31:0] fp_int(input integer n);
        integer highest;
        reg [31:0] shifted;
        begin
            highest = 0;
            for (integer b = 0; b < 24; b = b + 1)
                if ((n >> b) != 0) highest = b;
            shifted = n << (23 - highest);
            fp_int = n == 0 ? 0 : ((highest + 127) << 23) | (shifted & 32'h007fffff);
        end
    endfunction
    task automatic reset_core;
        @(negedge clk);
        reset = 1; run_enable = 0; launch_enable = 0; debug_write_enable = 0;
        repeat (3) @(negedge clk);
        reset = 0;
        for (integer i = 0; i < 16; i = i + 1) dut.core.imem.memory[i] = insn(OP_EXIT, 0, 0, 0);
    endtask
    task automatic write_reg(input integer warp, lane, rd, input reg [31:0] value);
        @(negedge clk);
        debug_write_enable = 1; debug_warp_id = warp; debug_lane_id = lane;
        debug_reg_addr = rd; debug_write_data = value;
        @(negedge clk);
        debug_write_enable = 0;
    endtask
    task automatic launch(input integer warp, input reg [3:0] mask, input integer pc = 0);
        @(negedge clk);
        launch_enable = 1; launch_warp_id = warp; launch_active_mask = mask; launch_pc = pc;
        @(negedge clk);
        launch_enable = 0;
    endtask
    task automatic expect_reg(input integer warp, lane, rd, input reg [31:0] value);
        if (dut.core.regfile.registers[warp][lane][rd] !== value)
            $fatal(1, "W%0d L%0d R%0d got=%h expected=%h", warp, lane, rd,
                dut.core.regfile.registers[warp][lane][rd], value);
    endtask
    task automatic check_memory;
        for (integer row = 0; row < 256; row = row + 1)
            if (dut.core.shared_mem.bank0[row] !== memory_model[row*4] ||
                dut.core.shared_mem.bank1[row] !== memory_model[row*4+1] ||
                dut.core.shared_mem.bank2[row] !== memory_model[row*4+2] ||
                dut.core.shared_mem.bank3[row] !== memory_model[row*4+3])
                $fatal(1, "Architectural memory mismatch row %0d", row);
    endtask
    task automatic run_to_halt(input reg [3:0] mask);
        integer cycles;
        @(negedge clk); run_enable = 1;
        cycles = 0;
        while ((halted_mask & mask) != mask && cycles < 5000) begin
            @(negedge clk); cycles = cycles + 1;
        end
        if ((halted_mask & mask) != mask) $fatal(1, "SM timeout pc=%h halted=%b", selected_pc, halted_mask);
        run_enable = 0;
        if (memory_pending || requests != responses) $fatal(1, "Unfinished memory after halt");
        for (integer warp = 0; warp < 4; warp = warp + 1)
            if (pending_load[warp] != 0) $fatal(1, "Unretired load after halt");
        check_memory();
    endtask
    task automatic seed_conflicting(input integer warp);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            write_reg(warp, lane, 1, warp*256 + lane*16);
            write_reg(warp, lane, 2, 100 + warp*10 + lane);
        end
    endtask

    initial begin
        // Every mask, four concurrent warps, different conflict groups, signed
        // offsets, memory ordering and dependent integer/store consumers.
        for (integer mask = 0; mask < 16; mask = mask + 1) begin
            reset_core();
            dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2, -16);
            dut.core.imem.memory[1] = insn(OP_LD_SHARED, 3, 1, 0, -16);
            dut.core.imem.memory[2] = insn(OP_ADD, 4, 3, 2);
            dut.core.imem.memory[3] = insn(OP_ST_SHARED, 0, 1, 4, 48);
            dut.core.imem.memory[4] = insn(OP_LD_SHARED, 5, 1, 0, 48);
            for (integer warp = 0; warp < 4; warp = warp + 1) begin
                for (integer lane = 0; lane < 4; lane = lane + 1) begin
                    write_reg(warp, lane, 1, warp*256 + lane*16 + 16 +
                        4*(warp == 0 ? 0 : warp == 1 ? lane%2 : warp == 2 ? (lane == 3 ? 1 : 0) : lane));
                    write_reg(warp, lane, 2, 1000 + warp*10 + lane);
                    for (integer rd = 3; rd <= 5; rd = rd + 1) write_reg(warp, lane, rd, 32'hdeadbeef);
                end
                launch(warp, mask);
            end
            run_to_halt(15);
            for (integer warp = 0; warp < 4; warp = warp + 1)
                for (integer lane = 0; lane < 4; lane = lane + 1) begin
                    expect_reg(warp, lane, 3, (mask & (1 << lane)) ? 1000 + warp*10 + lane : 32'hdeadbeef);
                    expect_reg(warp, lane, 4, (mask & (1 << lane)) ? 2*(1000 + warp*10 + lane) : 32'hdeadbeef);
                    expect_reg(warp, lane, 5, (mask & (1 << lane)) ? 2*(1000 + warp*10 + lane) : 32'hdeadbeef);
                end
            if (requests != 16 || load_writes != 8)
                $fatal(1, "Missing multiwarp/mask/hazard coverage");
        end
        $display("PASS: all active masks, four-warps, signed offsets, stores/loads and RAW dependencies");

        // Stores capture their addresses/data before younger instructions
        // overwrite those registers. Loads reserve their destination through WB.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
        dut.core.imem.memory[1] = insn(OP_ADD, 2, 0, 0);
        dut.core.imem.memory[2] = insn(OP_ADD, 1, 0, 0);
        dut.core.imem.memory[3] = insn(OP_LD_SHARED, 3, 6, 0);
        dut.core.imem.memory[4] = insn(OP_ADD, 6, 0, 0);
        dut.core.imem.memory[5] = insn(OP_ADD, 3, 0, 0);
        dut.core.imem.memory[6] = insn(OP_ADD, 4, 3, 0);
        seed_conflicting(0);
        for (integer lane = 0; lane < 4; lane = lane + 1) write_reg(0, lane, 6, lane*16);
        launch(0, 15); run_to_halt(1);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            expect_reg(0, lane, 3, 0); expect_reg(0, lane, 4, 0);
            if (dut.core.shared_mem.bank0[lane] !== 100 + lane) $fatal(1, "Lost captured store operands");
        end
        if (independent_issue == 0 || raw_stalls == 0) $fatal(1, "Missing independent issue/WAW coverage");
        $display("PASS: captured operands, independent issue during replay and load WAW ordering");

        // Same-word stores have deterministic lane order, and every lane gets
        // the final word on a subsequent load.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
        dut.core.imem.memory[1] = insn(OP_LD_SHARED, 3, 1, 0);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            write_reg(0, lane, 1, 4092); write_reg(0, lane, 2, 50 + lane);
        end
        launch(0, 15); run_to_halt(1);
        for (integer lane = 0; lane < 4; lane = lane + 1) expect_reg(0, lane, 3, 53);
        $display("PASS: same-address stores followed by repeated reads");

        // A later warp reads an older warp's store. The load also overwrites
        // its own address register, which must not disturb the captured addresses.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
        dut.core.imem.memory[8] = insn(OP_LD_SHARED, 1, 1, 0);
        dut.core.imem.memory[9] = insn(OP_ADD, 3, 1, 0);
        seed_conflicting(0);
        for (integer lane = 0; lane < 4; lane = lane + 1) write_reg(1, lane, 1, lane*16);
        launch(0, 15); launch(1, 15, 32); run_to_halt(3);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            expect_reg(1, lane, 1, 100 + lane); expect_reg(1, lane, 3, 100 + lane);
        end
        $display("PASS: cross-warp store/load ordering and aliased address/destination");

        // A branch changes the active mask before the load returns. Its result
        // still belongs to all originally active lanes; path stores use their
        // own masks and reconvergence restores the complete mask.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
        dut.core.imem.memory[1] = insn(OP_SSY, 0, 0, 0, 7); // join at 36
        dut.core.imem.memory[2] = insn(OP_LD_SHARED, 3, 1, 0);
        dut.core.imem.memory[3] = insn(OP_BEQ, 0, 6, 0, 2); // taken at 24
        dut.core.imem.memory[4] = insn(OP_ST_SHARED, 0, 1, 2, 64);
        dut.core.imem.memory[5] = insn(OP_BEQ, 0, 0, 0, 3); // join at 36
        dut.core.imem.memory[6] = insn(OP_ST_SHARED, 0, 1, 2, 64);
        dut.core.imem.memory[7] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[8] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[9] = insn(OP_LD_SHARED, 4, 1, 0, 64);
        seed_conflicting(0);
        for (integer lane = 0; lane < 4; lane = lane + 1) write_reg(0, lane, 6, lane%2);
        launch(0, 15); run_to_halt(1);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            expect_reg(0, lane, 3, 100 + lane); expect_reg(0, lane, 4, 100 + lane);
        end
        if (!dut.core.reconv_stack.empty || requests != 5 || mask_changes == 0)
            $fatal(1, "Memory paths did not exercise divergence/reconvergence during a load");
        $display("PASS: load mask survives divergence and masked stores reconverge");

        // EXIT directly after a store must wait even without a register result.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
        seed_conflicting(0); launch(0, 15); run_to_halt(1);
        if (requests != 1 || load_writes != 0 || conflicts == 0) $fatal(1, "Missing store drain coverage");
        $display("PASS: EXIT waits for every store lane");

        // Keep an older integer write in WB while a conflicting load finishes.
        // Debug writes backpressure WB; paused issue must not stop replay.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
        dut.core.imem.memory[1] = insn(OP_LD_SHARED, 3, 1, 0);
        dut.core.imem.memory[2] = insn(OP_ADD, 8, 0, 0);
        dut.core.imem.memory[3] = insn(OP_ADD, 4, 3, 0);
        seed_conflicting(0); launch(0, 15);
        @(negedge clk); run_enable = 1;
        @(posedge clk);
        while (!(dut.core.shared_start && opcode == OP_LD_SHARED)) @(posedge clk);
        repeat (2) @(negedge clk);
        run_enable = 0; debug_write_enable = 1;
        debug_warp_id = 1; debug_lane_id = 0; debug_reg_addr = 15; debug_write_data = 32'h12345678;
        repeat (12) @(negedge clk);
        if (!dut.core.shared_response_valid || !dut.core.shared_inflight || !dut.core.wb_valid || load_writes != 0)
            $fatal(1, "Memory result lost under writeback backpressure");
        for (integer lane = 0; lane < 4; lane = lane + 1) expect_reg(0, lane, 3, 0);
        debug_write_enable = 0;
        repeat (4) @(negedge clk);
        if (issue_accept || pending_load[0] != 0 || dut.core.shared_inflight)
            $fatal(1, "Paused memory did not drain correctly");
        for (integer lane = 0; lane < 4; lane = lane + 1) expect_reg(0, lane, 3, 100 + lane);
        run_to_halt(1);
        for (integer lane = 0; lane < 4; lane = lane + 1) expect_reg(0, lane, 4, 100 + lane);
        $display("PASS: pause/resume, held response and debug writeback backpressure");

        // Reset during partial store replay, partial load replay, and a load
        // response blocked behind an older WB. No stale writes may escape.
        for (integer phase = 0; phase < 3; phase = phase + 1) begin
            reset_core();
            dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
            dut.core.imem.memory[1] = insn(OP_LD_SHARED, 3, 1, 0);
            dut.core.imem.memory[2] = insn(OP_ADD, 8, 0, 0);
            seed_conflicting(0); launch(0, 15);
            @(negedge clk); run_enable = 1;
            @(posedge clk);
            while (!(dut.core.shared_start && opcode == (phase == 0 ? OP_ST_SHARED : OP_LD_SHARED))) @(posedge clk);
            repeat (2) @(negedge clk);
            if (phase == 2) begin
                run_enable = 0; debug_write_enable = 1; debug_warp_id = 1;
                repeat (10) @(negedge clk);
                if (!dut.core.shared_response_valid) $fatal(1, "Reset did not target pending response");
            end
            reset_core();
            repeat (12) @(negedge clk);
            if (dut.core.shared_inflight || dut.core.shared_response_valid || dut.core.wb_valid || requests != 0)
                $fatal(1, "Stale memory work after reset phase %0d", phase);
            check_memory();
            dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 1, 2);
            dut.core.imem.memory[1] = insn(OP_LD_SHARED, 3, 1, 0);
            seed_conflicting(0); launch(0, 15); run_to_halt(1);
            for (integer lane = 0; lane < 4; lane = lane + 1) expect_reg(0, lane, 3, 100 + lane);
        end
        $display("PASS: reset during store/load replay and held completion, followed by reuse");

        // Real FP, tensor, and load results all wait behind an older integer
        // write. Releasing the debug port must drain every producer correctly,
        // even with issue paused and different saved warp/lane tags.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[1] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[4] = insn(OP_FMUL, 3, 1, 2);
        dut.core.imem.memory[5] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[8] = insn(OP_LD_SHARED, 3, 1, 0);
        dut.core.imem.memory[9] = insn(OP_ADD, 8, 11, 12);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            for (integer row = 0; row < 4; row = row + 1) begin
                write_reg(0, lane, row, fp_int(row == lane ? 1 : 0));
                write_reg(0, lane, row+4, fp_int(row*4 + lane + 1));
                write_reg(0, lane, row+8, 32'hdeadbeef);
            end
            write_reg(1, lane, 1, 32'h00000001);
            write_reg(1, lane, 2, fp_int(1));
            write_reg(1, lane, 3, 32'hdeadbeef);
            write_reg(2, lane, 1, lane*16);
            write_reg(2, lane, 3, 32'hdeadbeef);
            write_reg(2, lane, 11, 7); write_reg(2, lane, 12, 9);
        end
        launch(0, 15); launch(1, 5, 16); launch(2, 11, 32);
        @(negedge clk); run_enable = 1;
        @(posedge clk);
        while (!(dut.core.instruction_execute_accept && selected_warp_id == 2 && opcode == OP_ADD))
            @(posedge clk);
        @(negedge clk);
        run_enable = 0; debug_write_enable = 1; debug_warp_id = 3;
        debug_reg_addr = 15; debug_lane_id = 0; debug_write_data = 123;
        repeat (2000) @(negedge clk);
        if (!dut.core.fp_completion_valid || !dut.core.tensor_row_valid ||
            !dut.core.shared_completion_valid || !dut.core.wb_valid)
            $fatal(1, "Did not hold all four writeback producers: fp=%b tensor=%b load=%b wb=%b requests=%0d responses=%0d",
                dut.core.fp_completion_valid, dut.core.tensor_row_valid, dut.core.shared_completion_valid,
                dut.core.wb_valid, requests, responses);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            expect_reg(1, lane, 3, 32'hdeadbeef); expect_reg(2, lane, 3, 32'hdeadbeef);
        end
        debug_write_enable = 0;
        repeat (50) @(negedge clk);
        if (dut.core.shared_inflight || pending_load[2] != 0 || dut.core.wb_valid ||
            dut.core.fp_inflight || dut.core.tensor_busy || issue_accept)
            $fatal(1, "Paused completions failed to drain");
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            expect_reg(1, lane, 3, (5 & (1 << lane)) ? 32'h00000001 : 32'hdeadbeef);
            expect_reg(2, lane, 3, (11 & (1 << lane)) ? 0 : 32'hdeadbeef);
            expect_reg(2, lane, 8, (11 & (1 << lane)) ? 16 : 0);
            for (integer row = 0; row < 4; row = row + 1)
                expect_reg(0, lane, row+8, fp_int(row*4 + lane + 1));
        end
        if (fp_collisions == 0 || tensor_collisions == 0)
            $fatal(1, "Missing FP/tensor/load collision coverage");
        run_to_halt(7);
        $display("PASS: simultaneous FP/tensor/load completion with occupied WB, masks and warp tags");

        // Conflicting loads fill A, TMUL waits for the entire input, and its
        // rows are stored with conflicts and loaded back into an FP consumer.
        reset_core();
        for (integer row = 0; row < 4; row = row + 1) begin
            dut.core.imem.memory[row] = insn(OP_ST_SHARED, 0, 12, row, row*64);
            dut.core.imem.memory[row+4] = insn(OP_LD_SHARED, row, 12, 0, row*64);
            dut.core.imem.memory[row+9] = insn(OP_ST_SHARED, 0, 12, row+8, 256+row*64);
            for (integer lane = 0; lane < 4; lane = lane + 1) begin
                write_reg(0, lane, row, fp_int(row*4 + lane + 1));
                write_reg(0, lane, row+4, fp_int(row == lane ? 1 : 0));
            end
        end
        dut.core.imem.memory[8] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[13] = insn(OP_LD_SHARED, 13, 12, 0, 256);
        dut.core.imem.memory[14] = insn(OP_FADD, 14, 13, 13);
        for (integer lane = 0; lane < 4; lane = lane + 1) write_reg(0, lane, 12, lane*16);
        launch(0, 15); run_to_halt(1);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            for (integer row = 0; row < 4; row = row + 1)
                expect_reg(0, lane, row+8, fp_int(row*4 + lane + 1));
            expect_reg(0, lane, 13, fp_int(lane + 1));
            expect_reg(0, lane, 14, fp_int(2*(lane + 1)));
        end
        if (requests != 13 || load_writes != 5 || raw_stalls == 0)
            $fatal(1, "Missing memory/tensor/FP dependency coverage");
        $display("PASS: conflicting loads -> TMUL -> conflicting stores -> load -> FADD");
        $display("ALL FULL-SM MEMORY TESTS PASSED (%0d memory instructions)", total_requests);
        $finish;
    end
    initial begin
        #4000000;
        $fatal(1, "Full-SM memory timeout");
    end
endmodule
