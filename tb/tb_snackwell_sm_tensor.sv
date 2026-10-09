`timescale 1ns/1ps
module tb_snackwell_sm_tensor;
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

    function automatic [31:0] insn(input logic [5:0] op, input logic [3:0] d, a, b,
        input logic [13:0] imm = 0);
        insn = {op, d, a, b, imm};
    endfunction
    function automatic [31:0] fp_int(input integer n);
        integer highest, shifted, magnitude;
        begin
            magnitude = n < 0 ? -n : n;
            highest = 0;
            for (integer b = 0; b < 23; b = b + 1)
                if ((magnitude >> b) != 0) highest = b;
            shifted = magnitude << (23 - highest);
            fp_int = n == 0 ? 0 : ((n < 0) ? 32'h80000000 : 0) |
                ((highest + 127) << 23) | (shifted & 32'h007fffff);
        end
    endfunction
    function automatic bit same_fp(input logic [31:0] got, expected);
        if (expected[30:23] == 255 && expected[22:0] != 0)
            same_fp = got[30:23] === 8'hff && (|got[22:0]) === 1'b1;
        else same_fp = got === expected;
    endfunction

    // Architectural monitor: tile reservations remain live across all four
    // writes. Check instruction operands against this independent pending set.
    reg [15:0] pending[0:3];
    reg model_tensor_busy;
    integer model_warp, model_base, model_rows, model_gather;
    reg [3:0] model_mask;
    reg [511:0] model_a, model_b;
    integer jobs, rows, raw_stalls, unit_stalls, int_overlap, fp_overlap, memory_overlap;
    integer fp_tensor_collisions, tensor_fast_collisions, simultaneous_writes;
    integer all_fp_collisions = 0, all_fast_collisions = 0;
    integer d, a, b, w;
    reg reads_a, reads_b, writes_d, valid_tensor;
    reg [15:0] accesses;
    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < 4; i = i + 1) pending[i] = 0;
            model_tensor_busy = 0; model_rows = 0; model_gather = 0;
            jobs = 0; rows = 0; raw_stalls = 0; unit_stalls = 0;
            int_overlap = 0; fp_overlap = 0; memory_overlap = 0;
            fp_tensor_collisions = 0; tensor_fast_collisions = 0; simultaneous_writes = 0;
        end else begin
            if (dependency_stall) raw_stalls = raw_stalls + 1;
            if (resource_stall) unit_stalls = unit_stalls + 1;
            if (dut.core.fp_completion_valid && dut.core.tensor_row_valid)
                fp_tensor_collisions = fp_tensor_collisions + 1;
            if (dut.core.tensor_row_valid && issue_valid &&
                (opcode == OP_ADD || opcode == OP_LD_SHARED))
                tensor_fast_collisions = tensor_fast_collisions + 1;
            if (dut.core.tensor_take && dut.core.wb_fire)
                simultaneous_writes = simultaneous_writes + 1;
            if (model_gather != 0) begin
                if (issue_accept) $fatal(1, "Instruction issued while tensor owns register reads");
                if (dut.core.rs1_data !== model_a[(4-model_gather)*128 +: 128] ||
                    dut.core.rs2_data !== model_b[(4-model_gather)*128 +: 128])
                    $fatal(1, "Tensor gather lost its issue-time operands/warp context");
                model_gather = model_gather - 1;
            end
            if (dut.core.wb_fire) begin
                if (!pending[dut.core.wb_warp_id][dut.core.wb_rd_addr])
                    $fatal(1, "Unallocated or duplicate writeback");
                if (dut.core.wb_is_tensor) begin
                    if (!model_tensor_busy || dut.core.wb_warp_id != model_warp ||
                        dut.core.wb_rd_addr != model_base + model_rows ||
                        dut.core.wb_write_mask !== model_mask ||
                        dut.core.wb_tensor_last !== (model_rows == 3))
                        $fatal(1, "Incorrect tensor writeback tag/order/mask");
                    rows = rows + 1;
                    model_rows = model_rows + 1;
                    if (model_rows == 4) begin
                        pending[model_warp] = pending[model_warp] & ~(16'h000f << model_base);
                        model_tensor_busy = 0;
                    end
                end else pending[dut.core.wb_warp_id][dut.core.wb_rd_addr] = 0;
            end
            if (issue_accept && !dut.core.at_reconvergence_point) begin
                d = instruction[25:22]; a = instruction[21:18]; b = instruction[17:14];
                w = selected_warp_id;
                reads_a = 0; reads_b = 0; writes_d = 0;
                valid_tensor = opcode == OP_TMUL && d <= 12 && a <= 12 && b <= 12 &&
                               instruction[13:0] == 0;
                case (opcode)
                    OP_ADD, OP_SUB, OP_AND, OP_OR, OP_XOR, OP_FMUL, OP_FADD, OP_FSUB:
                        begin reads_a = 1; reads_b = 1; writes_d = 1; end
                    OP_LD_SHARED: begin reads_a = 1; writes_d = 1; end
                    OP_ST_SHARED, OP_BEQ: begin reads_a = 1; reads_b = 1; end
                    default: begin end
                endcase
                accesses = (reads_a ? (16'h1 << a) : 0) |
                           (reads_b ? (16'h1 << b) : 0) | (writes_d ? (16'h1 << d) : 0);
                if (valid_tensor)
                    accesses = (16'hf << a) | (16'hf << b) | (16'hf << d);
                if (|(accesses & pending[w])) $fatal(1, "Issued across a RAW/WAW tile dependency");
                if (opcode == OP_EXIT && pending[w] != 0) $fatal(1, "EXIT before tile completion");
                if (model_tensor_busy && opcode == OP_ADD) int_overlap = int_overlap + 1;
                if (model_tensor_busy && (opcode == OP_FMUL || opcode == OP_FADD || opcode == OP_FSUB))
                    fp_overlap = fp_overlap + 1;
                if (model_tensor_busy && (opcode == OP_LD_SHARED || opcode == OP_ST_SHARED))
                    memory_overlap = memory_overlap + 1;
                if (writes_d) pending[w][d] = 1;
                if (valid_tensor) begin
                    if (model_tensor_busy) $fatal(1, "Overlapping tensor jobs");
                    model_tensor_busy = 1; model_warp = w; model_base = d;
                    model_rows = 0; model_gather = 4; model_mask = selected_active_mask;
                    jobs = jobs + 1;
                    pending[w] = pending[w] | (16'hf << d);
                    for (integer r = 0; r < 4; r = r + 1)
                        for (integer c = 0; c < 4; c = c + 1) begin
                            model_a[(r*4+c)*32 +: 32] = dut.core.regfile.registers[w][c][a+r];
                            model_b[(r*4+c)*32 +: 32] = dut.core.regfile.registers[w][c][b+r];
                        end
                end
            end
        end
    end

    task automatic reset_core;
        @(negedge clk);
        reset = 1; run_enable = 0; launch_enable = 0; debug_write_enable = 0;
        repeat (3) @(negedge clk);
        reset = 0;
        for (integer i = 0; i < 16; i = i + 1)
            dut.core.imem.memory[i] = insn(OP_EXIT, 0, 0, 0);
    endtask
    task automatic write_reg(input logic [1:0] warp, lane,
        input logic [3:0] rd, input logic [31:0] value);
        @(negedge clk);
        debug_write_enable = 1; debug_warp_id = warp; debug_lane_id = lane;
        debug_reg_addr = rd; debug_write_data = value;
        @(negedge clk); debug_write_enable = 0;
    endtask
    task automatic launch(input logic [1:0] warp, input logic [3:0] mask,
        input logic [31:0] pc = 0);
        @(negedge clk);
        launch_enable = 1; launch_warp_id = warp; launch_active_mask = mask; launch_pc = pc;
        @(negedge clk); launch_enable = 0;
    endtask
    task automatic expect_reg(input integer warp, lane, rd, input logic [31:0] value);
        if (dut.core.regfile.registers[warp][lane][rd] !== value)
            $fatal(1, "W%0d L%0d R%0d got=%h expected=%h", warp, lane, rd,
                dut.core.regfile.registers[warp][lane][rd], value);
    endtask
    task automatic run_to_halt(input logic [3:0] mask);
        integer cycles;
        @(negedge clk); run_enable = 1;
        cycles = 0;
        while ((halted_mask & mask) != mask && cycles < 15000) begin
            @(negedge clk); cycles = cycles + 1;
        end
        if ((halted_mask & mask) != mask)
            $fatal(1, "SM timeout halted=%b pc=%h blocked=%b pending0=%h", halted_mask,
                selected_pc, dut.core.blocked_mask, pending[0]);
        run_enable = 0;
        for (integer warp = 0; warp < 4; warp = warp + 1)
            if (mask[warp] && pending[warp] != 0) $fatal(1, "Outstanding writes after EXIT");
    endtask
    task automatic simple_tiles(input integer warp, scale = 1);
        for (integer r = 0; r < 4; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1) begin
                write_reg(warp, c, r, fp_int((r*4+c+1)*scale));
                write_reg(warp, c, 4+r, fp_int(r == c));
            end
    endtask
    task automatic expect_simple(input integer warp, base = 8, scale = 1,
        input logic [3:0] mask = 15);
        for (integer r = 0; r < 4; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1)
                expect_reg(warp, c, base+r, mask[c] ? fp_int((r*4+c+1)*scale) : 0);
    endtask
    task automatic wait_busy;
        integer cycles;
        @(negedge clk); run_enable = 1;
        cycles = 0;
        while (!dut.core.tensor_busy && cycles < 100) begin
            @(negedge clk); cycles = cycles + 1;
        end
        if (!dut.core.tensor_busy) $fatal(1, "Tensor instruction did not start");
    endtask

    reg [31:0] source_a[0:15], source_b[0:15], expected[0:15];
    reg [31:0] expected_regs[0:3][0:15];
    reg [31:0] word_value;
    integer fd, scan, word_scan, case_id, cases = 0, vector_warp, destination;
    integer cycles;
    reg [3:0] masks[0:3];
    string vector_path;
    initial begin
        if (!$value$plusargs("TENSOR_VECTORS=%s", vector_path)) $fatal(1, "Missing tensor vectors");
        fd = $fopen(vector_path, "r");
        if (!fd) $fatal(1, "Cannot open tensor vectors");
        scan = $fscanf(fd, "%h", case_id);
        while (scan == 1) begin
            for (integer i = 0; i < 48; i = i + 1) begin
                word_scan = $fscanf(fd, "%h", word_value);
                if (word_scan != 1) $fatal(1, "Malformed tensor vector");
                if (i < 16) source_a[i] = word_value;
                else if (i < 32) source_b[i-16] = word_value;
                else expected[i-32] = word_value;
            end
            reset_core();
            vector_warp = case_id % 4;
            case (case_id % 6)
                0: destination = 8;
                1: destination = 0; // exact A alias
                2: destination = 4; // exact B alias
                3: destination = 2; // partial overlap with both A and B
                4: destination = 6;
                default: destination = 12;
            endcase
            for (integer r = 0; r < 4; r = r + 1)
                for (integer c = 0; c < 4; c = c + 1) begin
                    write_reg(vector_warp, c, r, source_a[r*4+c]);
                    write_reg(vector_warp, c, 4+r, source_b[r*4+c]);
                end
            for (integer c = 0; c < 4; c = c + 1)
                for (integer r = 0; r < 16; r = r + 1)
                    expected_regs[c][r] = dut.core.regfile.registers[vector_warp][c][r];
            for (integer r = 0; r < 4; r = r + 1)
                for (integer c = 0; c < 4; c = c + 1)
                    expected_regs[c][destination+r] = expected[r*4+c];
            dut.core.imem.memory[0] = insn(OP_TMUL, destination, 0, 4);
            launch(vector_warp, 15);
            run_to_halt(4'b1 << vector_warp);
            for (integer c = 0; c < 4; c = c + 1)
                for (integer r = 0; r < 16; r = r + 1)
                    if (!same_fp(dut.core.regfile.registers[vector_warp][c][r], expected_regs[c][r]))
                        $fatal(1, "SM matrix case %0d R%0d L%0d got=%h expected=%h", case_id, r, c,
                            dut.core.regfile.registers[vector_warp][c][r], expected_regs[c][r]);
            if (jobs != 1 || rows != 4 || dut.core.tensor_busy)
                $fatal(1, "Incorrect matrix retirement count");
            cases = cases + 1;
            scan = $fscanf(fd, "%h", case_id);
        end
        if (!$feof(fd) || cases == 0) $fatal(1, "Malformed/empty vector file");
        $fclose(fd);
        $display("PASS: %0d full-SM matrix vectors, 2960 outputs, in-place and overlapping destinations", cases);

        // Every row of either input or output must wait for an older FP writer.
        for (integer target = 0; target < 12; target = target + 1) begin
            reset_core();
            simple_tiles(0);
            for (integer c = 0; c < 4; c = c + 1) write_reg(0, c, 14, fp_int(1));
            dut.core.imem.memory[0] = insn(OP_FMUL, target, target, 14);
            dut.core.imem.memory[1] = insn(OP_TMUL, 8, 0, 4);
            launch(0, 15);
            run_to_halt(1);
            expect_simple(0);
            if (raw_stalls == 0 || jobs != 1 || rows != 4)
                $fatal(1, "Missed incoming tile dependency on R%0d", target);
        end
        $display("PASS: incoming FP RAW/WAW dependencies on every A/B/C row");

        // Younger consumers and writers must wait for the entire destination,
        // including when the first row has already reached the register file.
        for (integer target = 8; target < 12; target = target + 1) begin
            for (integer waw = 0; waw < 2; waw = waw + 1) begin
                reset_core();
                simple_tiles(0);
                dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
                dut.core.imem.memory[1] = waw ? insn(OP_ADD, target, 15, 15)
                                             : insn(OP_ADD, 12, 15, target);
                launch(0, 15);
                run_to_halt(1);
                for (integer r = 0; r < 4; r = r + 1)
                    for (integer c = 0; c < 4; c = c + 1)
                        expect_reg(0, c, 8+r, (waw && target == 8+r) ? 0 : fp_int(r*4+c+1));
                if (!waw)
                    for (integer c = 0; c < 4; c = c + 1)
                        expect_reg(0, c, 12, fp_int((target-8)*4+c+1));
                if (raw_stalls == 0) $fatal(1, "Missing outgoing tile dependency");
            end
        end
        $display("PASS: whole-tile RAW/WAW protection through the last row commit");

        // The RF read-port lock prevents a younger write from changing an
        // unread source row; after gathering, source registers can be reused.
        reset_core();
        simple_tiles(0);
        dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[1] = insn(OP_ADD, 3, 15, 15);
        dut.core.imem.memory[2] = insn(OP_FADD, 12, 8, 15);
        launch(0, 15);
        run_to_halt(1);
        expect_simple(0);
        for (integer c = 0; c < 4; c = c + 1) begin
            expect_reg(0, c, 3, 0);
            expect_reg(0, c, 12, fp_int(c+1));
        end
        if (int_overlap == 0) $fatal(1, "Independent source reuse failed to overlap");
        $display("PASS: source snapshots, gather interlock, and tensor-to-FP consumer");

        // Non-aligned legal bases and overlapping A/B inputs are valid.
        reset_core();
        simple_tiles(0);
        for (integer c = 0; c < 4; c = c + 1)
            for (integer r = 3; r >= 0; r = r - 1) begin
                write_reg(0, c, 6+r, fp_int(r == c));
                write_reg(0, c, 1+r, fp_int(r*4+c+1));
            end
        dut.core.imem.memory[0] = insn(OP_TMUL, 12, 1, 6);
        launch(0, 15);
        run_to_halt(1);
        expect_simple(0, 12);
        // Square the identity using exactly the same A and B tile.
        reset_core();
        simple_tiles(0);
        dut.core.imem.memory[0] = insn(OP_TMUL, 4, 4, 4);
        launch(0, 15);
        run_to_halt(1);
        for (integer r = 0; r < 4; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1) expect_reg(0, c, 4+r, fp_int(r == c));
        $display("PASS: unaligned bases, highest legal destination, and identical input tiles");

        // Partial input overlap is also legal: A=R0..R3, B=R2..R5, C=R1..R4.
        reset_core();
        for (integer r = 0; r < 6; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1) write_reg(0, c, r, fp_int(1));
        dut.core.imem.memory[0] = insn(OP_TMUL, 1, 0, 2);
        launch(0, 15);
        run_to_halt(1);
        for (integer r = 0; r < 6; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1)
                expect_reg(0, c, r, fp_int(r == 0 || r == 5 ? 1 : 4));
        $display("PASS: partially overlapping A, B, and C register tiles");

        // Dense B makes every active output depend on inactive A lanes too.
        // Check every mask and retain nonzero sentinels in inactive outputs.
        for (integer mask = 0; mask < 16; mask = mask + 1) begin
            reset_core();
            simple_tiles(0);
            for (integer r = 0; r < 4; r = r + 1)
                for (integer c = 0; c < 4; c = c + 1) begin
                    write_reg(0, c, 4+r, fp_int(1));
                    write_reg(0, c, 8+r, 32'hdeadbeef);
                end
            dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
            launch(0, mask);
            run_to_halt(1);
            for (integer r = 0; r < 4; r = r + 1)
                for (integer c = 0; c < 4; c = c + 1)
                    expect_reg(0, c, 8+r, mask[c] ? fp_int(16*r+10) : 32'hdeadbeef);
            if (jobs != 1 || rows != 4) $fatal(1, "Masked tensor failed to retire");
        end
        $display("PASS: all 16 output masks with dense inputs and inactive-column preservation");

        reset_core();
        dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[1] = insn(OP_ADD, 14, 15, 15);
        dut.core.imem.memory[2] = insn(OP_FADD, 12, 1, 15);
        dut.core.imem.memory[3] = insn(OP_ST_SHARED, 0, 13, 11);
        dut.core.imem.memory[4] = insn(OP_LD_SHARED, 14, 13, 0);
        dut.core.imem.memory[5] = insn(OP_FADD, 12, 10, 15);
        dut.core.imem.memory[6] = insn(OP_TMUL, 0, 8, 4);
        masks[0] = 15; masks[1] = 5; masks[2] = 10; masks[3] = 0;
        for (integer warp = 0; warp < 4; warp = warp + 1) begin
            simple_tiles(warp, warp+1);
            for (integer c = 0; c < 4; c = c + 1)
                write_reg(warp, c, 13, warp*64 + c*4);
            launch(warp, masks[warp]);
        end
        run_to_halt(15);
        for (integer warp = 0; warp < 4; warp = warp + 1) begin
            expect_simple(warp, 8, warp+1, masks[warp]);
            expect_simple(warp, 0, warp+1);
            for (integer c = 0; c < 4; c = c + 1) begin
                expect_reg(warp, c, 12, masks[warp][c] ? fp_int((9+c)*(warp+1)) : 0);
                expect_reg(warp, c, 14, masks[warp][c] ? fp_int((13+c)*(warp+1)) : 0);
            end
        end
        if (jobs != 8 || rows != 32 || raw_stalls == 0 || unit_stalls == 0 ||
            int_overlap == 0 || fp_overlap == 0)
            $fatal(1, "Missing multiwarp tensor/mask/overlap coverage");
        $display("PASS: four-warp contention, masks including zero, chained tensors, integer/FP/shared memory");

        // Branch changes masks while an earlier tensor job is in flight.
        // The join itself contains TMUL and must execute only after restoring
        // the full warp mask, not on either reconvergence micro-op.
        reset_core();
        simple_tiles(0);
        dut.core.imem.memory[0] = insn(OP_SSY, 0, 0, 0, 10); // join at 44
        dut.core.imem.memory[1] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[2] = insn(OP_BEQ, 0, 12, 15, 4); // taken at 28
        dut.core.imem.memory[3] = insn(OP_FADD, 0, 0, 14);
        dut.core.imem.memory[4] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[5] = insn(OP_BEQ, 0, 15, 15, 5);
        dut.core.imem.memory[6] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[7] = insn(OP_FSUB, 0, 0, 14);
        dut.core.imem.memory[8] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[9] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[10] = insn(OP_NOP, 0, 0, 0);
        dut.core.imem.memory[11] = insn(OP_TMUL, 0, 8, 4);
        for (integer c = 0; c < 4; c = c + 1) begin
            write_reg(0, c, 12, c % 2);
            write_reg(0, c, 14, fp_int(1));
        end
        launch(0, 15);
        run_to_halt(1);
        for (integer r = 0; r < 4; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1) begin
                expect_reg(0, c, r, fp_int(r*4+c+1 + (r == 0 ? (c % 2 ? 1 : -1) : 0)));
                expect_reg(0, c, 8+r, fp_int(r*4+c+1 + (r == 0 ? (c % 2 ? 1 : -1) : 0)));
            end
        if (jobs != 4 || rows != 16 || !dut.core.reconv_stack.empty)
            $fatal(1, "Tensor divergence/reconvergence executed incorrectly");
        $display("PASS: captured masks, divergent tensor paths, and tensor at reconvergence PC");

        // Pause immediately after acceptance: gathering and computation must
        // continue. Hold a row in WB and the next row at the producer using
        // debug-port backpressure; neither may lose data or release registers.
        reset_core();
        simple_tiles(0);
        dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
        dut.core.imem.memory[1] = insn(OP_ADD, 12, 8, 15);
        launch(0, 15);
        wait_busy();
        run_enable = 0;
        debug_write_enable = 1; debug_warp_id = 1; debug_lane_id = 0;
        debug_reg_addr = 15; debug_write_data = 32'h12345678;
        repeat (3000) @(negedge clk);
        if (!dut.core.wb_valid || !dut.core.wb_is_tensor || !dut.core.tensor_row_valid ||
            pending[0] !== 16'h0f00 || rows != 0 || issue_accept)
            $fatal(1, "Lost tensor completion/reservation during paused debug backpressure");
        for (integer r = 8; r < 12; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1) expect_reg(0, c, r, 0);
        // Release one committed row at a time, reapplying backpressure before
        // each next commit. All four reservations must survive the early rows.
        for (integer row = 0; row < 4; row = row + 1) begin
            debug_write_enable = 0;
            @(negedge clk);
            debug_write_enable = 1;
            if (rows != row+1) $fatal(1, "Unexpected tensor row count under intermittent backpressure");
            repeat (8) @(negedge clk);
            if (row < 3 && pending[0] !== 16'h0f00)
                $fatal(1, "Tensor reservation released before last commit");
        end
        debug_write_enable = 0;
        expect_simple(0);
        if (pending[0] != 0 || dut.core.tensor_busy || issue_accept)
            $fatal(1, "Paused tensor failed to drain");
        run_to_halt(1);
        for (integer c = 0; c < 4; c = c + 1) expect_reg(0, c, 12, fp_int(c+1));
        $display("PASS: pause during gather, continued computation, held rows and per-row debug backpressure");

        // Hold an older integer/load result in WB so FP and tensor completions
        // both become pending. Release all three while another fast writer is
        // trying to issue; verify priority and every architectural result.
        for (integer load_case = 0; load_case < 2; load_case = load_case + 1) begin
            reset_core();
            simple_tiles(0);
            for (integer c = 0; c < 4; c = c + 1) begin
                write_reg(0, c, 13, c*4);
                write_reg(0, c, 14, 32'hcafebabe);
            end
            dut.core.imem.memory[0] = insn(OP_ST_SHARED, 0, 13, 14);
            dut.core.imem.memory[1] = insn(OP_TMUL, 8, 0, 4);
            dut.core.imem.memory[2] = insn(OP_FMUL, 12, 0, 1);
            dut.core.imem.memory[3] = load_case ? insn(OP_LD_SHARED, 14, 13, 0)
                                              : insn(OP_ADD, 14, 15, 15);
            dut.core.imem.memory[4] = insn(OP_ADD, 3, 15, 15);
            dut.core.imem.memory[5] = insn(OP_LD_SHARED, 2, 13, 0);
            launch(0, 15);
            wait_busy();
            cycles = 0;
            while (!(dut.core.wb_valid && !dut.core.wb_is_tensor && dut.core.wb_rd_addr == 14)
                && cycles < 100) begin
                @(negedge clk); cycles = cycles + 1;
            end
            if (cycles == 100 || !dut.core.fp_inflight) $fatal(1, "Completion collision setup failed");
            run_enable = 0;
            debug_write_enable = 1; debug_warp_id = 1; debug_lane_id = 0;
            debug_reg_addr = 15; debug_write_data = 0;
            repeat (3000) @(negedge clk);
            if (!dut.core.fp_completion_valid || !dut.core.tensor_row_valid ||
                !dut.core.wb_valid || dut.core.wb_rd_addr != 14 || rows != 0)
                $fatal(1, "Did not retain all three colliding producers");
            debug_write_enable = 0; run_enable = 1;
            run_to_halt(1);
            expect_simple(0);
            for (integer c = 0; c < 4; c = c + 1) begin
                expect_reg(0, c, 12, fp_int((c+1)*(c+5)));
                expect_reg(0, c, 14, load_case ? 32'hcafebabe : 0);
                expect_reg(0, c, 3, 0);
                expect_reg(0, c, 2, 32'hcafebabe);
            end
            if (jobs != 1 || rows != 4 || fp_tensor_collisions == 0 ||
                tensor_fast_collisions == 0 || simultaneous_writes == 0)
                $fatal(1, "Missing completion arbitration coverage");
            all_fp_collisions = all_fp_collisions + fp_tensor_collisions;
            all_fast_collisions = all_fast_collisions + tensor_fast_collisions;
        end
        $display("PASS: FP/tensor/integer/load arbitration (%0d FP and %0d fast-writer contention cycles)",
            all_fp_collisions, all_fast_collisions);

        // Malformed TMUL encodings are no-ops; they cannot wrap R15 into R0
        // or leave scoreboard reservations that prevent EXIT.
        reset_core();
        simple_tiles(0);
        dut.core.imem.memory[0] = insn(OP_TMUL, 13, 0, 4);
        dut.core.imem.memory[1] = insn(OP_TMUL, 8, 13, 4);
        dut.core.imem.memory[2] = insn(OP_TMUL, 8, 0, 13);
        dut.core.imem.memory[3] = insn(OP_TMUL, 8, 0, 4, 1);
        launch(0, 15);
        run_to_halt(1);
        if (jobs != 0 || rows != 0) $fatal(1, "Malformed tensor encoding performed work");
        expect_simple(0, 0);
        for (integer r = 4; r < 16; r = r + 1)
            for (integer c = 0; c < 4; c = c + 1)
                expect_reg(0, c, r, r < 8 ? fp_int(r-4 == c) : 0);
        $display("PASS: malformed tile bases and reserved immediate bits");

        // Reset at gather, arithmetic, and partial writeback; then relaunch a
        // fresh job on the same warp and ensure no old completion can appear.
        for (integer phase = 0; phase < 3; phase = phase + 1) begin
            reset_core();
            simple_tiles(0);
            dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
            launch(0, 15);
            wait_busy();
            if (phase == 1) repeat (60) @(negedge clk);
            if (phase == 2) begin
                cycles = 0;
                while (rows < 1 && cycles < 6000) begin
                    @(negedge clk); cycles = cycles + 1;
                end
                if (rows != 1) $fatal(1, "Reset partial-writeback setup failed");
                debug_write_enable = 1; debug_warp_id = 1; debug_lane_id = 0;
                debug_reg_addr = 15; debug_write_data = 0;
            end
            reset_core();
            repeat (100) @(negedge clk);
            if (dut.core.wb_valid || dut.core.tensor_busy || dut.core.tensor_row_valid || rows != 0)
                $fatal(1, "Stale tensor state after reset");
            simple_tiles(0, 2);
            dut.core.imem.memory[0] = insn(OP_TMUL, 8, 0, 4);
            launch(0, 15);
            run_to_halt(1);
            expect_simple(0, 8, 2);
            if (jobs != 1 || rows != 4) $fatal(1, "Reset/relaunch failed");
            // Relaunch a retired warp without global reset, consuming its
            // previous result. This exercises normal tag/reservation reuse.
            dut.core.imem.memory[0] = insn(OP_TMUL, 0, 8, 4);
            launch(0, 15);
            run_to_halt(1);
            expect_simple(0, 0, 2);
            if (jobs != 2 || rows != 8) $fatal(1, "Retired warp relaunch lost work");
        end
        $display("PASS: reset during gather/compute/drain and reset-free retired warp relaunch");
        $display("ALL FULL-SM TENSOR TESTS PASSED");
        $finish;
    end
    initial begin
        #25000000;
        $fatal(1, "Full-SM tensor global timeout");
    end
endmodule
