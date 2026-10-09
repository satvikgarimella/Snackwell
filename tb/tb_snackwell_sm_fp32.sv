`timescale 1ns/1ps
module tb_snackwell_sm_fp32;
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
    wire [31:0] selected_pc;
    wire [5:0] opcode;
    wire [3:0] halted_mask;
    snackwell_sm dut (
        .clk(clk), .reset(reset), .run_enable(run_enable),
        .launch_enable(launch_enable), .launch_warp_id(launch_warp_id),
        .launch_pc(launch_pc), .launch_active_mask(launch_active_mask),
        .debug_write_enable(debug_write_enable), .debug_warp_id(debug_warp_id),
        .debug_lane_id(debug_lane_id), .debug_reg_addr(debug_reg_addr),
        .debug_write_data(debug_write_data),
        .issue_valid(issue_valid), .issue_accept(issue_accept),
        .selected_warp_id(selected_warp_id), .selected_pc(selected_pc), .opcode(opcode),
        .halted_mask(halted_mask), .dependency_stall(dependency_stall),
        .resource_stall(resource_stall)
    );
    always #5 clk = ~clk;

    integer fp_issued, retired, raw_stalls, resource_stalls, integer_overlap, fp_collisions;
    reg [15:0] pending[0:3];
    reg [3:0] masks[0:3];
    integer cycles;
    integer collisions_total = 0, simultaneous_writes = 0;

    // Independently track architectural destinations allocated at issue.
    // Check that writes neither arrive early nor retire twice, and EXIT drains.
    always @(posedge clk) begin
        if (reset) begin
            fp_issued = 0; retired = 0; raw_stalls = 0; resource_stalls = 0;
            integer_overlap = 0; fp_collisions = 0;
            for (integer w = 0; w < 4; w = w + 1) pending[w] = 0;
        end else begin
            if (dependency_stall) raw_stalls = raw_stalls + 1;
            if (resource_stall) resource_stalls = resource_stalls + 1;
            if (dut.core.fp_start) fp_issued = fp_issued + 1;
            if (dut.core.fp_inflight && issue_accept && dut.core.is_integer_op)
                integer_overlap = integer_overlap + 1;
            if (dut.core.fp_completion_valid && issue_valid &&
                (dut.core.is_integer_op || dut.core.is_shared_load))
                fp_collisions = fp_collisions + 1;
            if (dut.core.fp_take && dut.core.wb_fire)
                simultaneous_writes = simultaneous_writes + 1;
            if (dut.core.wb_fire) begin
                if (!pending[dut.core.wb_warp_id][dut.core.wb_rd_addr])
                    $fatal(1, "Writeback without a pending destination");
                pending[dut.core.wb_warp_id][dut.core.wb_rd_addr] = 0;
                retired = retired + 1;
            end
            if (dut.core.instruction_execute_accept && dut.core.writes_rd) begin
                if (pending[selected_warp_id][dut.core.rd_addr])
                    $fatal(1, "Two instructions own the same destination");
                pending[selected_warp_id][dut.core.rd_addr] = 1;
            end
            if (dut.core.instruction_execute_accept && opcode == OP_EXIT &&
                pending[selected_warp_id] != 0)
                $fatal(1, "EXIT before all writes retired");
        end
    end

    function automatic [31:0] insn(input logic [5:0] op,
        input logic [3:0] rd, rs1, rs2, input integer imm = 0);
        insn = {op, rd, rs1, rs2, imm[13:0]};
    endfunction

    // Exact conversion for the small nonnegative integer expectations below.
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
        for (integer i = 0; i < 16; i = i + 1)
            dut.core.imem.memory[i] = insn(OP_EXIT, 0, 0, 0);
    endtask

    task automatic write_reg(input logic [1:0] w, lane,
        input logic [3:0] rd, input logic [31:0] value);
        @(negedge clk);
        debug_write_enable = 1;
        debug_warp_id = w; debug_lane_id = lane;
        debug_reg_addr = rd; debug_write_data = value;
        @(negedge clk);
        debug_write_enable = 0;
    endtask

    task automatic launch(input logic [1:0] w, input logic [3:0] mask,
        input logic [31:0] pc = 0);
        @(negedge clk);
        launch_enable = 1; launch_warp_id = w; launch_active_mask = mask; launch_pc = pc;
        @(negedge clk);
        launch_enable = 0;
    endtask

    task automatic expect_reg(input integer w, lane, rd, input logic [31:0] value);
        if (dut.core.regfile.registers[w][lane][rd] !== value)
            $fatal(1, "W%0d L%0d R%0d got=%h expected=%h", w, lane, rd,
                dut.core.regfile.registers[w][lane][rd], value);
    endtask

    task automatic run_to_halt(input logic [3:0] mask);
        @(negedge clk);
        run_enable = 1;
        cycles = 0;
        while ((halted_mask & mask) != mask && cycles < 4000) begin
            @(negedge clk);
            cycles = cycles + 1;
        end
        if ((halted_mask & mask) != mask)
            $fatal(1, "SM timeout halted=%b pc=%h blocked=%b", halted_mask,
                selected_pc, dut.core.blocked_mask);
        run_enable = 0;
        for (integer w = 0; w < 4; w = w + 1)
            if (mask[w] && pending[w] != 0) $fatal(1, "Pending writes after halt");
    endtask

    initial begin
        reset_core();
        // Four warps contend for the FP unit; every operation feeds its next
        // consumer, then passes through the integer and shared-memory paths.
        dut.core.imem.memory[0] = insn(OP_FMUL, 3, 1, 2);
        dut.core.imem.memory[1] = insn(OP_FADD, 4, 3, 2);
        dut.core.imem.memory[2] = insn(OP_FSUB, 5, 4, 1);
        dut.core.imem.memory[3] = insn(OP_ADD, 6, 5, 0);
        dut.core.imem.memory[4] = insn(OP_ST_SHARED, 0, 12, 6);
        dut.core.imem.memory[5] = insn(OP_LD_SHARED, 7, 12, 0);
        masks[0] = 15; masks[1] = 5; masks[2] = 10; masks[3] = 0;
        for (integer w = 0; w < 4; w = w + 1) begin
            for (integer lane = 0; lane < 4; lane = lane + 1) begin
                write_reg(w, lane, 1, fp_int(w + 1));
                write_reg(w, lane, 2, fp_int(lane + 1));
                write_reg(w, lane, 12, w*16 + lane*4);
                for (integer rd = 3; rd <= 7; rd = rd + 1)
                    write_reg(w, lane, rd, 32'hdeadbeef);
            end
            launch(w, masks[w]);
        end
        run_to_halt(15);
        for (integer w = 0; w < 4; w = w + 1)
            for (integer lane = 0; lane < 4; lane = lane + 1) begin
                expect_reg(w, lane, 3, masks[w][lane] ? fp_int((w+1)*(lane+1)) : 32'hdeadbeef);
                expect_reg(w, lane, 4, masks[w][lane] ? fp_int((w+2)*(lane+1)) : 32'hdeadbeef);
                for (integer rd = 5; rd <= 7; rd = rd + 1)
                    expect_reg(w, lane, rd, masks[w][lane] ? fp_int((w+2)*(lane+1)-(w+1)) : 32'hdeadbeef);
            end
        if (fp_issued != 12 || retired != 20 || raw_stalls == 0 || resource_stalls == 0)
            $fatal(1, "Missing multiwarp FP/dependency/resource coverage");
        $display("PASS: four-warp FMUL/FADD/FSUB, RAW dependencies, masks, integer and memory consumers");

        // A younger integer write to the FP destination must wait (WAW),
        // while an independent integer write can proceed during FP execution.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_FMUL, 3, 1, 2);
        dut.core.imem.memory[1] = insn(OP_ADD, 8, 10, 11);
        dut.core.imem.memory[2] = insn(OP_ADD, 3, 10, 11);
        dut.core.imem.memory[3] = insn(OP_ADD, 4, 3, 11);
        write_reg(0, 0, 1, 32'h40200000);
        write_reg(0, 0, 2, 32'h40800000);
        write_reg(0, 0, 10, 7);
        write_reg(0, 0, 11, 9);
        launch(0, 1);
        run_to_halt(1);
        expect_reg(0, 0, 3, 16); expect_reg(0, 0, 4, 25); expect_reg(0, 0, 8, 16);
        if (integer_overlap == 0 || raw_stalls == 0 || retired != 4)
            $fatal(1, "Missing WAW or concurrent integer coverage");
        $display("PASS: WAW ordering and independent integer issue during FP execution");

        // Sweep completion timing against a second warp's integer/load loop.
        // Both producers share one writeback slot; all loop iterations must
        // survive, including a cycle retiring an older write as FP completes.
        for (integer op = 0; op < 3; op = op + 1)
            for (integer padding = 0; padding < 4; padding = padding + 1) begin
                reset_core();
                for (integer pc = 0; pc < padding; pc = pc + 1)
                    dut.core.imem.memory[pc] = insn(OP_NOP, 0, 0, 0);
                dut.core.imem.memory[padding] = insn(op == 0 ? OP_FMUL : op == 1 ? OP_FADD : OP_FSUB, 3, 1, 2);
                dut.core.imem.memory[8] = insn(OP_ADD, 8, 8, 11);
                dut.core.imem.memory[9] = insn(OP_LD_SHARED, 10, 14, 0);
                dut.core.imem.memory[10] = insn(OP_SUB, 9, 9, 11);
                dut.core.imem.memory[11] = insn(OP_BEQ, 0, 9, 0, 1);
                dut.core.imem.memory[12] = insn(OP_BEQ, 0, 0, 0, -5);
                write_reg(0, 0, 1, fp_int(1));
                write_reg(0, 0, 2, 32'h00000001);
                write_reg(1, 0, 9, 64); write_reg(1, 0, 11, 1);
                dut.core.shared_mem.bank0[0] = 32'hcafebabe;
                launch(0, 1); launch(1, 1, 32);
                run_to_halt(3);
                expect_reg(0, 0, 3, op == 0 ? 32'h00000001 : fp_int(1));
                expect_reg(1, 0, 8, 64); expect_reg(1, 0, 9, 0);
                expect_reg(1, 0, 10, 32'hcafebabe);
                if (retired != 193 || integer_overlap == 0)
                    $fatal(1, "Lost integer/load/FP write during collision sweep");
                collisions_total = collisions_total + fp_collisions;
            end
        if (collisions_total == 0 || simultaneous_writes == 0)
            $fatal(1, "Writeback collision sweep did not exercise arbitration");
        $display("PASS: FP/integer/load writeback arbitration (%0d collisions, %0d concurrent retirements)",
            collisions_total, simultaneous_writes);

        // An instruction following the branch changes the active mask before
        // the older FP result returns. Its saved mask must still be used.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_SSY, 0, 0, 0, 6); // join at 28
        dut.core.imem.memory[1] = insn(OP_FMUL, 3, 6, 7);
        dut.core.imem.memory[2] = insn(OP_BEQ, 0, 1, 2, 2); // taken at 20
        dut.core.imem.memory[3] = insn(OP_FSUB, 4, 6, 7);   // deferred
        dut.core.imem.memory[4] = insn(OP_BEQ, 0, 0, 0, 2);
        dut.core.imem.memory[5] = insn(OP_FADD, 4, 6, 7);   // taken
        dut.core.imem.memory[6] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[7] = insn(OP_FADD, 5, 3, 4);   // join consumer
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            write_reg(0, lane, 1, lane % 2);
            write_reg(0, lane, 6, fp_int(4));
            write_reg(0, lane, 7, fp_int(2));
        end
        launch(0, 15);
        run_to_halt(1);
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            expect_reg(0, lane, 3, fp_int(8));
            expect_reg(0, lane, 4, fp_int(lane % 2 ? 2 : 6));
            expect_reg(0, lane, 5, fp_int(lane % 2 ? 10 : 14));
        end
        if (!dut.core.reconv_stack.empty || retired != 4)
            $fatal(1, "FP divergent paths did not reconverge");
        $display("PASS: captured masks survive divergence and FP executes through reconvergence");

        // Pause only stops new issue. Completions still drain, and a debug
        // register write temporarily backpressures the shared writeback port.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_FMUL, 3, 1, 2);
        dut.core.imem.memory[1] = insn(OP_ADD, 8, 0, 0);
        dut.core.imem.memory[2] = insn(OP_ADD, 4, 3, 0);
        write_reg(0, 0, 1, fp_int(3)); write_reg(0, 0, 2, fp_int(4));
        launch(0, 1);
        @(negedge clk); run_enable = 1;
        @(negedge clk);
        @(negedge clk); run_enable = 0;
        debug_write_enable = 1; debug_warp_id = 1; debug_lane_id = 0;
        debug_reg_addr = 15; debug_write_data = 32'h12345678;
        repeat (100) @(negedge clk);
        expect_reg(0, 0, 3, 0);
        if (!dut.core.wb_valid || !pending[0][3] || !dut.core.fp_result_pending)
            $fatal(1, "Lost FP completion while an older writeback was backpressured");
        debug_write_enable = 0;
        repeat (3) @(negedge clk);
        expect_reg(0, 0, 3, fp_int(12));
        if (issue_accept || pending[0] != 0) $fatal(1, "Pause failed to drain without issuing");
        run_to_halt(1);
        expect_reg(0, 0, 4, fp_int(12));
        $display("PASS: pause/resume and debug-port writeback backpressure");

        // EXIT must wait even when it does not read the pending FP result.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_FMUL, 3, 1, 2);
        write_reg(0, 0, 1, fp_int(2)); write_reg(0, 0, 2, fp_int(4));
        launch(0, 1);
        run_to_halt(1);
        expect_reg(0, 0, 3, fp_int(8));
        if (retired != 1) $fatal(1, "EXIT lost outstanding FP result");
        $display("PASS: EXIT drains outstanding FP writeback");

        // Reset during execution discards both the completion tag and result.
        reset_core();
        dut.core.imem.memory[0] = insn(OP_FMUL, 3, 1, 2);
        write_reg(0, 0, 1, 1); write_reg(0, 0, 2, fp_int(1));
        launch(0, 1);
        @(negedge clk); run_enable = 1;
        repeat (5) @(negedge clk);
        reset_core();
        repeat (100) @(negedge clk);
        if (dut.core.wb_valid || dut.core.fp_inflight || retired != 0)
            $fatal(1, "Stale FP completion after reset");
        expect_reg(0, 0, 3, 0);
        launch(0, 1);
        run_to_halt(1);
        $display("PASS: reset abort and relaunch");
        $display("ALL FULL-SM FP32 TESTS PASSED");
        $finish;
    end

    initial begin
        #1000000;
        $fatal(1, "Full-SM FP timeout");
    end
endmodule
