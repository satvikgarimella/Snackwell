`timescale 1ns/1ps

module tb_multiwarp_memory_core;

    import snackwell_pkg::*;

    logic clk;
    logic reset;
    logic run_enable;

    logic        launch_enable;
    logic [1:0]  launch_warp_id;
    logic [31:0] launch_pc;
    logic [3:0]  launch_active_mask;

    logic        debug_write_enable;
    logic [1:0]  debug_warp_id;
    logic [1:0]  debug_lane_id;
    logic [3:0]  debug_reg_addr;
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
    integer cycles;


    multiwarp_execution_core dut (

        .clk                  (clk),
        .reset                (reset),
        .run_enable           (run_enable),

        .launch_enable        (launch_enable),
        .launch_warp_id       (launch_warp_id),
        .launch_pc            (launch_pc),
        .launch_active_mask   (launch_active_mask),

        .debug_write_enable   (debug_write_enable),
        .debug_warp_id        (debug_warp_id),
        .debug_lane_id        (debug_lane_id),
        .debug_reg_addr       (debug_reg_addr),
        .debug_write_data     (debug_write_data),

        .issue_valid          (issue_valid),
        .issue_accept         (issue_accept),
        .selected_warp_id     (selected_warp_id),

        .selected_pc          (selected_pc),
        .instruction          (instruction),
        .opcode               (opcode),

        .selected_active_mask (selected_active_mask),

        .valid_mask           (valid_mask),
        .ready_mask           (ready_mask),
        .halted_mask          (halted_mask),
        .blocked_mask         (blocked_mask),

        .dependency_stall     (dependency_stall),
        .resource_stall       (resource_stall),

        .rs1_data             (rs1_data),
        .rs2_data             (rs2_data),
        .alu_result           (alu_result),

        .shared_bank_conflict (shared_bank_conflict),
        .shared_conflict_mask (shared_conflict_mask)

    );


    always #5 clk = ~clk;


    initial begin

        $dumpfile("tb_multiwarp_memory_core.vcd");
        $dumpvars(0, tb_multiwarp_memory_core);

    end


    task launch_warp (
        input logic [1:0] warp
    );

        begin

            launch_enable      = 1'b1;
            launch_warp_id     = warp;
            launch_pc          = 32'd0;
            launch_active_mask = 4'b0001;

            @(posedge clk);
            #1;

            launch_enable = 1'b0;

        end

    endtask


    task write_reg (
        input logic [1:0]  warp,
        input logic [1:0]  lane,
        input logic [3:0]  reg_id,
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
        $display(
            "=== Snackwell Multi-Warp Shared Memory Integration Test ==="
        );
        $display("");


        // ---------------------------------------------
        // RESET
        // ---------------------------------------------

        @(posedge clk);
        #1;

        reset = 0;


        // ---------------------------------------------
        // PROGRAM
        //
        // 0x00:
        // ST_SHARED [R1 + 0], R2
        //
        // 0x04:
        // LD_SHARED R3, [R1 + 0]
        //
        // 0x08:
        // EXIT
        // ---------------------------------------------

        dut.imem.memory[0] = {
            OP_ST_SHARED,
            4'd0,
            4'd1,
            4'd2,
            14'd0
        };

        dut.imem.memory[1] = {
            OP_LD_SHARED,
            4'd3,
            4'd1,
            4'd0,
            14'd0
        };

        dut.imem.memory[2] = {
            OP_EXIT,
            4'd0,
            4'd0,
            4'd0,
            14'd0
        };


        // ---------------------------------------------
        // LAUNCH W0 ONLY
        // ---------------------------------------------

        launch_warp(2'd0);


        // Lane 0:
        //
        // R1 = shared memory address 0x40
        // R2 = value 1234

        write_reg(
            2'd0,
            2'd0,
            4'd1,
            32'h00000040
        );

        write_reg(
            2'd0,
            2'd0,
            4'd2,
            32'd1234
        );


        if (blocked_mask !== 4'b0000) begin

            $display(
                "FAIL: warp blocked before run"
            );

            $fatal;

        end


        $display(
            "PASS: warp launched and memory operands loaded"
        );


        // ---------------------------------------------
        // RUN
        // ---------------------------------------------

        run_enable = 1'b1;


        // ---------------------------------------------
        // STORE
        // ---------------------------------------------

        #1;

        if (
            opcode !== OP_ST_SHARED ||
            !issue_accept
        ) begin

            $display(
                "FAIL: expected ST_SHARED at PC 0"
            );

            $fatal;

        end


        $display(
            "ISSUE: ST_SHARED [R1], R2"
        );


        @(posedge clk);
        #1;


        // ---------------------------------------------
        // LOAD
        // ---------------------------------------------

        // The older store must finish before the next memory request issues.
        cycles = 0;
        while (!(opcode == OP_LD_SHARED && issue_accept) && cycles < 30) begin
            @(posedge clk);
            #1;
            cycles = cycles + 1;
        end

        if (
            opcode !== OP_LD_SHARED ||
            selected_pc !== 32'h00000004 ||
            !issue_accept
        ) begin

            $display(
                "FAIL: expected LD_SHARED at PC 4"
            );

            $fatal;

        end


        $display(
            "ISSUE: LD_SHARED R3, [R1]"
        );


        @(posedge clk);
        #1;


        // Loads complete asynchronously and retire through the shared WB port.
        cycles = 0;
        while (dut.regfile.registers[0][0][3] !== 32'd1234 && cycles < 30) begin
            @(posedge clk);
            #1;
            cycles = cycles + 1;
        end


        // ---------------------------------------------
        // DIRECTLY VERIFY ARCHITECTURAL R3
        // ---------------------------------------------

        if (
            dut.regfile.registers[0][0][3]
            !== 32'd1234 
        ) begin

            $display(
                "FAIL: W0 L0 R3=%0d expected=1234",
                dut.regfile.registers[0][0][3]
            );

            $fatal;

        end


        $display(
            "PASS: LD_SHARED wrote 1234 into W0 L0 R3"
        );


        // ---------------------------------------------
        // VERIFY SHARED MEMORY ITSELF
        //
        // Address 0x40:
        //
        // bank = address[3:2] = 0
        // row  = address[11:4] = 4
        // ---------------------------------------------

        if (
            dut.shared_mem.bank0[4]
            !== 32'd1234
        ) begin

            $display(
                "FAIL: shared memory value incorrect"
            );

            $fatal;

        end


        $display(
            "PASS: store committed into banked shared memory"
        );


        // ---------------------------------------------
        // RUN TO EXIT
        // ---------------------------------------------

        cycles = 0;
        while (halted_mask !== 4'b0001 && cycles < 30) begin
            @(posedge clk);
            #1;
            cycles = cycles + 1;
        end

        if (halted_mask !== 4'b0001) $fatal(1, "Memory program did not halt");


        if (issue_valid !== 1'b0) begin

            $display(
                "FAIL: halted warp still issuing"
            );

            $fatal;

        end


        $display(
            "PASS: warp reached EXIT after memory operations"
        );


        $display("");
        $display(
            "ALL MULTI-WARP MEMORY INTEGRATION TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule
