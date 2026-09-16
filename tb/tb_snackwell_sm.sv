`timescale 1ns/1ps

module tb_snackwell_sm;

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


    snackwell_sm dut (

        .clk                     (clk),
        .reset                   (reset),
        .run_enable              (run_enable),

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

        $dumpfile("tb_snackwell_sm.vcd");
        $dumpvars(0, tb_snackwell_sm);

    end


    task write_reg (
        input logic [1:0] warp,
        input logic [1:0] lane,
        input logic [3:0] reg_id,
        input logic [31:0] value
    );

        begin

            debug_write_enable = 1'b1;

            debug_warp_id      = warp;
            debug_lane_id      = lane;
            debug_reg_addr     = reg_id;
            debug_write_data   = value;

            @(posedge clk);
            #1;

            debug_write_enable = 1'b0;

        end

    endtask


    integer lane;


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


        $display("");
        $display("=== Snackwell SM Top-Level Test ===");
        $display("");


        // =================================================
        // RESET
        // =================================================

        @(posedge clk);
        #1;

        reset = 0;


        if (
            valid_mask !== 4'b0000 ||
            ready_mask !== 4'b0000 ||
            halted_mask !== 4'b0000
        ) begin

            $display(
                "FAIL: SM warp state incorrect after reset"
            );

            $fatal;

        end


        $display(
            "PASS: SM reset"
        );


        // =================================================
        // SIMPLE PROGRAM
        //
        // ADD R3, R1, R2
        // EXIT
        // =================================================

        dut.core.imem.memory[0] = {
            OP_ADD,
            4'd3,
            4'd1,
            4'd2,
            14'd0
        };


        dut.core.imem.memory[1] = {
            OP_EXIT,
            4'd0,
            4'd0,
            4'd0,
            14'd0
        };


        // =================================================
        // PRELOAD FOUR LANES
        // =================================================

        write_reg(
            2'd0, 2'd0,
            4'd1, 32'd10
        );

        write_reg(
            2'd0, 2'd0,
            4'd2, 32'd5
        );


        write_reg(
            2'd0, 2'd1,
            4'd1, 32'd20
        );

        write_reg(
            2'd0, 2'd1,
            4'd2, 32'd7
        );


        write_reg(
            2'd0, 2'd2,
            4'd1, 32'd30
        );

        write_reg(
            2'd0, 2'd2,
            4'd2, 32'd9
        );


        write_reg(
            2'd0, 2'd3,
            4'd1, 32'd40
        );

        write_reg(
            2'd0, 2'd3,
            4'd2, 32'd11
        );


        // =================================================
        // LAUNCH
        // =================================================

        launch_enable = 1'b1;

        launch_warp_id = 2'd0;
        launch_pc = 32'd0;
        launch_active_mask = 4'b1111;

        @(posedge clk);
        #1;

        launch_enable = 1'b0;


        if (
            valid_mask[0] !== 1'b1 ||
            ready_mask[0] !== 1'b1
        ) begin

            $display(
                "FAIL: W0 did not launch"
            );

            $fatal;

        end


        $display(
            "PASS: warp launched through snackwell_sm"
        );


        // =================================================
        // RUN
        // =================================================

        run_enable = 1'b1;


        // Wait for halt.

        while (halted_mask[0] !== 1'b1) begin

            @(posedge clk);
            #1;

        end


        // One extra cycle for final WB safety.

        @(posedge clk);
        #1;


        // =================================================
        // VERIFY SIMD RESULTS
        // =================================================

        if (
            dut.core.regfile.registers[0][0][3]
            !== 32'd15
        ) begin

            $display(
                "FAIL: lane0 result"
            );

            $fatal;

        end


        if (
            dut.core.regfile.registers[0][1][3]
            !== 32'd27
        ) begin

            $display(
                "FAIL: lane1 result"
            );

            $fatal;

        end


        if (
            dut.core.regfile.registers[0][2][3]
            !== 32'd39
        ) begin

            $display(
                "FAIL: lane2 result"
            );

            $fatal;

        end


        if (
            dut.core.regfile.registers[0][3][3]
            !== 32'd51
        ) begin

            $display(
                "FAIL: lane3 result"
            );

            $fatal;

        end


        $display(
            "PASS: 4-lane SIMD executed through complete SM"
        );


        if (halted_mask !== 4'b0001) begin

            $display(
                "FAIL: warp did not halt correctly"
            );

            $fatal;

        end


        $display(
            "PASS: EXIT propagated through complete SM"
        );


        $display("");
        $display(
            "ALL SNACKWELL SM TOP-LEVEL TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule