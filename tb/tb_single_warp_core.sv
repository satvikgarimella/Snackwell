`timescale 1ns/1ps

module tb_single_warp_core;

    logic clk;
    logic reset;
    logic run_enable;

    logic        debug_write_enable;
    logic [1:0]  debug_lane_select;
    logic [3:0]  debug_write_addr;
    logic [31:0] debug_write_data;

    logic [31:0] pc;
    logic [31:0] instruction;
    logic [5:0]  opcode;

    logic halted;

    logic [127:0] debug_rs1_data;
    logic [127:0] debug_rs2_data;
    logic [127:0] debug_alu_result;
    // DUT

    single_warp_core dut (
        .clk                (clk),
        .reset              (reset),
        .run_enable         (run_enable),

        .debug_write_enable (debug_write_enable),
        .debug_lane_select  (debug_lane_select),
        .debug_write_addr   (debug_write_addr),
        .debug_write_data   (debug_write_data),

        .pc                 (pc),
        .instruction        (instruction),
        .opcode             (opcode),

        .halted             (halted),

        .debug_rs1_data     (debug_rs1_data),
        .debug_rs2_data     (debug_rs2_data),
        .debug_alu_result   (debug_alu_result)
    );

    // CLOCK
    //
    // 10 ns period

    always #5 clk = ~clk;
    // WAVEFORM

    initial begin

        $dumpfile("tb_single_warp_core.vcd");
        $dumpvars(0, tb_single_warp_core);

    end
    // PRELOAD REGISTER

    task preload_register (
        input logic [1:0]  lane,
        input logic [3:0]  reg_addr,
        input logic [31:0] value
    );

        begin

            debug_lane_select  = lane;
            debug_write_addr   = reg_addr;
            debug_write_data   = value;
            debug_write_enable = 1'b1;

            @(posedge clk);
            #1;

            debug_write_enable = 1'b0;

        end

    endtask

    // CHECK ALL FOUR LANES
    //
    // Reads from debug_rs1_data.

    task check_four_lane_values (
        input logic [31:0] lane0_expected,
        input logic [31:0] lane1_expected,
        input logic [31:0] lane2_expected,
        input logic [31:0] lane3_expected
    );

        begin

            #1;

            if (debug_rs1_data[31:0] !== lane0_expected) begin

                $display(
                    "FAIL lane 0: got=%h expected=%h",
                    debug_rs1_data[31:0],
                    lane0_expected
                );

                $fatal;

            end

            if (debug_rs1_data[63:32] !== lane1_expected) begin

                $display(
                    "FAIL lane 1: got=%h expected=%h",
                    debug_rs1_data[63:32],
                    lane1_expected
                );

                $fatal;

            end

            if (debug_rs1_data[95:64] !== lane2_expected) begin

                $display(
                    "FAIL lane 2: got=%h expected=%h",
                    debug_rs1_data[95:64],
                    lane2_expected
                );

                $fatal;

            end

            if (debug_rs1_data[127:96] !== lane3_expected) begin

                $display(
                    "FAIL lane 3: got=%h expected=%h",
                    debug_rs1_data[127:96],
                    lane3_expected
                );

                $fatal;

            end

        end

    endtask
    // MAIN TEST

    initial begin

        clk = 0;

        reset      = 1;
        run_enable = 0;

        debug_write_enable = 0;
        debug_lane_select  = 0;
        debug_write_addr   = 0;
        debug_write_data   = 0;

        $display("");
        $display("=== Snackwell Single Warp Program Test ===");
        $display("");
        // RESET

        @(posedge clk);
        #1;

        reset = 0;

        if (pc !== 32'h00000000) begin

            $display(
                "FAIL: PC after reset = %h",
                pc
            );

            $fatal;

        end

        $display("PASS: PC reset to 0");

        // PRELOAD REGISTERS
        //
        // run_enable remains ZERO.
        //
        // Therefore the program must NOT
        // execute while loading registers.

        // Lane 0
        //
        // R1 = 7
        // R2 = 8

        preload_register(
            2'd0,
            4'd1,
            32'd7
        );

        preload_register(
            2'd0,
            4'd2,
            32'd8
        );

        // Lane 1
        //
        // R1 = 7
        // R2 = 5

        preload_register(
            2'd1,
            4'd1,
            32'd7
        );

        preload_register(
            2'd1,
            4'd2,
            32'd5
        );

        // Lane 2
        //
        // R1 = 20
        // R2 = 30

        preload_register(
            2'd2,
            4'd1,
            32'd20
        );

        preload_register(
            2'd2,
            4'd2,
            32'd30
        );

        // Lane 3
        //
        // R1 = 100
        // R2 = 1

        preload_register(
            2'd3,
            4'd1,
            32'd100
        );

        preload_register(
            2'd3,
            4'd2,
            32'd1
        );
        // VERIFY PC DID NOT MOVE

        if (pc !== 32'h00000000) begin

            $display(
                "FAIL: PC moved during preload. PC=%h",
                pc
            );

            $fatal;

        end

        $display(
            "PASS: PC held at 0 during register preload"
        );

        $display(
            "PASS: initial registers loaded"
        );

        $display("");
        // START EXECUTION

        run_enable = 1;

        // INSTRUCTION 0
        //
        // PC = 0x00
        // ADD R3,R1,R2

        if (pc !== 32'h00000000) begin

            $display(
                "FAIL: expected PC=0, got %h",
                pc
            );

            $fatal;

        end

        $display(
            "Executing ADD at PC=%h",
            pc
        );

        @(posedge clk);
        #1;

        // INSTRUCTION 1
        //
        // PC = 0x04
        // SUB R4,R1,R2

        if (pc !== 32'h00000004) begin

            $display(
                "FAIL: expected PC=4, got %h",
                pc
            );

            $fatal;

        end

        $display(
            "Executing SUB at PC=%h",
            pc
        );

        @(posedge clk);
        #1;

        // INSTRUCTION 2
        //
        // PC = 0x08
        // XOR R5,R1,R2

        if (pc !== 32'h00000008) begin

            $display(
                "FAIL: expected PC=8, got %h",
                pc
            );

            $fatal;

        end

        $display(
            "Executing XOR at PC=%h",
            pc
        );

        @(posedge clk);
        #1;
        // EXIT

        if (pc !== 32'h0000000C) begin

            $display(
                "FAIL: expected PC=C, got %h",
                pc
            );

            $fatal;

        end

        if (!halted) begin

            $display(
                "FAIL: expected core to be halted"
            );

            $fatal;

        end

        $display("");
        $display("PASS: program reached EXIT");
        $display("");

        // VERIFY R3
        //
        // ADD RESULTS

        force dut.execution.rs1_addr = 4'd3;

        #1;

        check_four_lane_values(
            32'd15,
            32'd12,
            32'd50,
            32'd101
        );

        release dut.execution.rs1_addr;

        $display(
            "PASS: R3 ADD results correct"
        );

        // VERIFY R4
        //
        // SUB RESULTS

        force dut.execution.rs1_addr = 4'd4;

        #1;

        check_four_lane_values(
            32'hFFFFFFFF,
            32'd2,
            32'hFFFFFFF6,
            32'd99
        );

        release dut.execution.rs1_addr;

        $display(
            "PASS: R4 SUB results correct"
        );

        // VERIFY R5
        //
        // XOR RESULTS

        force dut.execution.rs1_addr = 4'd5;

        #1;

        check_four_lane_values(
            32'd15,
            32'd2,
            32'd10,
            32'd101
        );

        release dut.execution.rs1_addr;

        $display(
            "PASS: R5 XOR results correct"
        );
        // VERIFY EXIT HOLDS PC

        repeat (3) begin

            @(posedge clk);
            #1;

            if (pc !== 32'h0000000C) begin

                $display(
                    "FAIL: EXIT did not hold PC. PC=%h",
                    pc
                );

                $fatal;

            end

        end

        $display(
            "PASS: EXIT holds PC at 0x0000000C"
        );

        $display("");
        $display(
            "ALL SINGLE WARP PROGRAM TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule
