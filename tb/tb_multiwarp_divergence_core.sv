`timescale 1ns/1ps

module tb_multiwarp_divergence_core;

    import snackwell_pkg::*;

    logic clk;
    logic reset;
    logic run_enable;

    logic launch_enable;
    logic [1:0] launch_warp_id;
    logic [31:0] launch_pc;
    logic [3:0] launch_active_mask;

    logic debug_write_enable;
    logic [1:0] debug_warp_id;
    logic [1:0] debug_lane_id;
    logic [3:0] debug_reg_addr;
    logic [31:0] debug_write_data;

    logic issue_valid;
    logic issue_accept;
    logic [1:0] selected_warp_id;

    logic [31:0] selected_pc;
    logic [31:0] instruction;
    logic [5:0] opcode;

    logic [3:0] selected_active_mask;

    logic [3:0] valid_mask;
    logic [3:0] ready_mask;
    logic [3:0] halted_mask;
    logic [3:0] blocked_mask;

    logic dependency_stall;
    logic resource_stall;

    logic [127:0] rs1_data;
    logic [127:0] rs2_data;
    logic [127:0] alu_result;

    logic shared_bank_conflict;
    logic [3:0] shared_conflict_mask;

    logic branch_divergent;
    logic [3:0] branch_taken_mask;
    logic [3:0] branch_not_taken_mask;

    logic reconvergence_event;

    multiwarp_execution_core dut (

        .clk                     (clk),
        .reset                   (reset),
        .run_enable              (run_enable),
        .host_program_write(1'b0), .host_program_address(10'd0), .host_program_data(32'd0),
        .host_program_read_data(),
        .host_memory_valid(1'b0), .host_memory_ready(), .host_memory_write(1'b0),
        .host_memory_address(10'd0), .host_memory_data(32'd0),
        .host_memory_response_valid(), .host_memory_response_ready(1'b0),
        .host_memory_read_data(), .execution_idle(),


        .launch_enable           (launch_enable),
        .launch_warp_id          (launch_warp_id),
        .launch_pc               (launch_pc),
        .launch_active_mask      (launch_active_mask),

        .debug_write_enable      (debug_write_enable),
        .debug_warp_id           (debug_warp_id),
        .debug_lane_id           (debug_lane_id),
        .debug_reg_addr          (debug_reg_addr),
        .debug_write_data        (debug_write_data),

        .issue_valid             (issue_valid),
        .issue_accept            (issue_accept),
        .selected_warp_id        (selected_warp_id),

        .selected_pc             (selected_pc),
        .instruction             (instruction),
        .opcode                  (opcode),

        .selected_active_mask    (selected_active_mask),

        .valid_mask              (valid_mask),
        .ready_mask              (ready_mask),
        .halted_mask             (halted_mask),
        .blocked_mask            (blocked_mask),

        .dependency_stall        (dependency_stall),
        .resource_stall          (resource_stall),

        .rs1_data                (rs1_data),
        .rs2_data                (rs2_data),
        .alu_result              (alu_result),

        .shared_bank_conflict    (shared_bank_conflict),
        .shared_conflict_mask    (shared_conflict_mask),

        .branch_divergent        (branch_divergent),
        .branch_taken_mask       (branch_taken_mask),
        .branch_not_taken_mask   (branch_not_taken_mask),

        .reconvergence_event     (reconvergence_event)

    );

    always #5 clk = ~clk;

    initial begin

        $dumpfile(
            "tb_multiwarp_divergence_core.vcd"
        );

        $dumpvars(
            0,
            tb_multiwarp_divergence_core
        );

    end

    task write_reg (
        input logic [1:0] warp,
        input logic [1:0] lane,
        input logic [3:0] reg_id,
        input logic [31:0] value
    );

        begin

            debug_write_enable = 1'b1;

            debug_warp_id =
                warp;

            debug_lane_id =
                lane;

            debug_reg_addr =
                reg_id;

            debug_write_data =
                value;

            @(posedge clk);
            #1;

            debug_write_enable =
                1'b0;

        end

    endtask

    integer cycle_count;

    initial begin

        clk = 0;
        reset = 1;
        run_enable = 0;

        launch_enable = 0;
        launch_warp_id = 0;
        launch_pc = 0;
        launch_active_mask = 0;

        debug_write_enable = 0;
        debug_warp_id = 0;
        debug_lane_id = 0;
        debug_reg_addr = 0;
        debug_write_data = 0;

        cycle_count = 0;

        $display("");
        $display(
            "=== Snackwell Full Divergence Integration Test ==="
        );
        $display("");
        // RESET

        @(posedge clk);
        #1;

        reset = 0;

        // PROGRAM
        //
        // 0x00 SSY join=0x14
        //
        // SSY:
        // target = PC + 4 + imm*4
        //
        // 0 + 4 + (4*4) = 0x14
        //
        //
        // 0x04 BEQ R1,R2,target=0x10
        //
        // 4 + 4 + (2*4) = 0x10
        //
        //
        // 0x08 ADD R4,R6,R7
        //      deferred / not-taken path
        //
        // 0x0C BEQ R0,R0,join=0x14
        //
        // 12 + 4 + (1*4) = 0x14
        //
        //
        // 0x10 ADD R3,R6,R7
        //      taken path
        //
        //
        // 0x14 NOP
        //      reconvergence point
        //
        // 0x18 EXIT

        dut.imem.memory[0] = {
            OP_SSY,
            4'd0,
            4'd0,
            4'd0,
            14'd4
        };

        dut.imem.memory[1] = {
            OP_BEQ,
            4'd0,
            4'd1,
            4'd2,
            14'd2
        };

        dut.imem.memory[2] = {
            OP_ADD,
            4'd4,
            4'd6,
            4'd7,
            14'd0
        };

        // BEQ R0,R0,+1 = unconditional for active lanes

        dut.imem.memory[3] = {
            OP_BEQ,
            4'd0,
            4'd0,
            4'd0,
            14'd1
        };

        dut.imem.memory[4] = {
            OP_ADD,
            4'd3,
            4'd6,
            4'd7,
            14'd0
        };

        dut.imem.memory[5] = {
            OP_NOP,
            4'd0,
            4'd0,
            4'd0,
            14'd0
        };

        dut.imem.memory[6] = {
            OP_EXIT,
            4'd0,
            4'd0,
            4'd0,
            14'd0
        };

        // REGISTER SETUP
        //
        // R1/R2 cause:
        //
        // L0 equal
        // L1 unequal
        // L2 equal
        // L3 unequal
        //
        // => taken 0101
        // => deferred 1010
        //
        // R6 + R7 = lane-specific result

        // ---------------- LANE 0 ----------------

        write_reg(
            2'd0, 2'd0,
            4'd1, 32'd10
        );

        write_reg(
            2'd0, 2'd0,
            4'd2, 32'd10
        );

        write_reg(
            2'd0, 2'd0,
            4'd6, 32'd100
        );

        write_reg(
            2'd0, 2'd0,
            4'd7, 32'd1
        );

        // ---------------- LANE 1 ----------------

        write_reg(
            2'd0, 2'd1,
            4'd1, 32'd20
        );

        write_reg(
            2'd0, 2'd1,
            4'd2, 32'd99
        );

        write_reg(
            2'd0, 2'd1,
            4'd6, 32'd200
        );

        write_reg(
            2'd0, 2'd1,
            4'd7, 32'd2
        );

        // ---------------- LANE 2 ----------------

        write_reg(
            2'd0, 2'd2,
            4'd1, 32'd30
        );

        write_reg(
            2'd0, 2'd2,
            4'd2, 32'd30
        );

        write_reg(
            2'd0, 2'd2,
            4'd6, 32'd300
        );

        write_reg(
            2'd0, 2'd2,
            4'd7, 32'd3
        );

        // ---------------- LANE 3 ----------------

        write_reg(
            2'd0, 2'd3,
            4'd1, 32'd40
        );

        write_reg(
            2'd0, 2'd3,
            4'd2, 32'd999
        );

        write_reg(
            2'd0, 2'd3,
            4'd6, 32'd400
        );

        write_reg(
            2'd0, 2'd3,
            4'd7, 32'd4
        );
        // LAUNCH W0

        launch_enable = 1'b1;

        launch_warp_id = 2'd0;
        launch_pc = 32'd0;
        launch_active_mask = 4'b1111;

        @(posedge clk);
        #1;

        launch_enable = 1'b0;

        $display(
            "PASS: W0 launched with active mask 1111"
        );
        // RUN

        run_enable = 1'b1;
        // WATCH EXECUTION

        cycle_count = 0;

        while (
            halted_mask[0] !== 1'b1 &&
            cycle_count < 50
        ) begin

            #1;

            if (
                issue_valid &&
                issue_accept
            ) begin

                $display(
                    "cycle=%0d PC=%h opcode=%h mask=%b",
                    cycle_count,
                    selected_pc,
                    opcode,
                    selected_active_mask
                );

            end

            // First BEQ should diverge.

            if (
                selected_pc == 32'h00000004 &&
                opcode == OP_BEQ &&
                issue_accept
            ) begin

                if (!branch_divergent) begin

                    $display(
                        "FAIL: branch should diverge"
                    );

                    $fatal;

                end

                if (
                    branch_taken_mask !== 4'b0101
                ) begin

                    $display(
                        "FAIL: taken mask=%b expected=0101",
                        branch_taken_mask
                    );

                    $fatal;

                end

                if (
                    branch_not_taken_mask !== 4'b1010
                ) begin

                    $display(
                        "FAIL: deferred mask=%b expected=1010",
                        branch_not_taken_mask
                    );

                    $fatal;

                end

                $display(
                    "PASS: BEQ split warp into 0101 / 1010"
                );

            end

            @(posedge clk);
            #1;

            cycle_count =
                cycle_count + 1;

        end

        if (cycle_count >= 50) begin

            $display(
                "FAIL: timeout waiting for EXIT"
            );

            $fatal;

        end

        // Allow final writeback to settle.

        @(posedge clk);
        #1;

        // VERIFY TAKEN PATH
        //
        // R3 should only be written on lanes 0 and 2.

        if (
            dut.regfile.registers[0][0][3]
            !== 32'd101
        ) begin

            $display(
                "FAIL: L0 R3 expected 101"
            );

            $fatal;

        end

        if (
            dut.regfile.registers[0][1][3]
            !== 32'd0
        ) begin

            $display(
                "FAIL: L1 incorrectly executed taken path"
            );

            $fatal;

        end

        if (
            dut.regfile.registers[0][2][3]
            !== 32'd303
        ) begin

            $display(
                "FAIL: L2 R3 expected 303"
            );

            $fatal;

        end

        if (
            dut.regfile.registers[0][3][3]
            !== 32'd0
        ) begin

            $display(
                "FAIL: L3 incorrectly executed taken path"
            );

            $fatal;

        end

        $display(
            "PASS: taken path executed only on lanes 0 and 2"
        );

        // VERIFY DEFERRED PATH
        //
        // R4 should only be written on lanes 1 and 3.

        if (
            dut.regfile.registers[0][0][4]
            !== 32'd0
        ) begin

            $display(
                "FAIL: L0 incorrectly executed deferred path"
            );

            $fatal;

        end

        if (
            dut.regfile.registers[0][1][4]
            !== 32'd202
        ) begin

            $display(
                "FAIL: L1 R4 expected 202"
            );

            $fatal;

        end

        if (
            dut.regfile.registers[0][2][4]
            !== 32'd0
        ) begin

            $display(
                "FAIL: L2 incorrectly executed deferred path"
            );

            $fatal;

        end

        if (
            dut.regfile.registers[0][3][4]
            !== 32'd404
        ) begin

            $display(
                "FAIL: L3 R4 expected 404"
            );

            $fatal;

        end

        $display(
            "PASS: deferred path executed only on lanes 1 and 3"
        );
        // VERIFY RECONVERGENCE STACK EMPTIED

        if (
            dut.reconv_stack.stack_depth[0]
            !== 3'd0
        ) begin

            $display(
                "FAIL: reconvergence stack not empty"
            );

            $fatal;

        end

        $display(
            "PASS: divergent paths reconverged and stack popped"
        );
        // VERIFY WARP HALTED

        if (
            halted_mask !== 4'b0001
        ) begin

            $display(
                "FAIL: W0 did not reach EXIT"
            );

            $fatal;

        end

        $display(
            "PASS: reconverged warp reached EXIT"
        );

        $display("");
        $display(
            "ALL DIVERGENT SIMT EXECUTION TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule
