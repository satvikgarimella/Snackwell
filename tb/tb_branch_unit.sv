`timescale 1ns/1ps

module tb_branch_unit;

    logic [31:0] current_pc;

    logic [127:0] operand_a;
    logic [127:0] operand_b;

    logic [3:0] active_mask;

    logic [31:0] immediate;

    logic [3:0] taken_mask;
    logic [3:0] not_taken_mask;

    logic uniform_taken;
    logic uniform_not_taken;
    logic divergent;

    logic [31:0] branch_target;
    logic [31:0] fallthrough_pc;


    branch_unit dut (

        .current_pc        (current_pc),

        .operand_a         (operand_a),
        .operand_b         (operand_b),

        .active_mask       (active_mask),

        .immediate         (immediate),

        .taken_mask        (taken_mask),
        .not_taken_mask    (not_taken_mask),

        .uniform_taken     (uniform_taken),
        .uniform_not_taken (uniform_not_taken),
        .divergent         (divergent),

        .branch_target     (branch_target),
        .fallthrough_pc    (fallthrough_pc)

    );


    initial begin

        $dumpfile("tb_branch_unit.vcd");
        $dumpvars(0, tb_branch_unit);

    end


    initial begin

        current_pc = 32'h00000100;

        operand_a = 128'd0;
        operand_b = 128'd0;

        active_mask = 4'b1111;

        immediate = 32'd2;


        $display("");
        $display("=== Snackwell Branch Unit Test ===");
        $display("");


        // =================================================
        // PC CALCULATION
        //
        // PC = 0x100
        // fallthrough = 0x104
        //
        // imm = 2 instructions
        // displacement = 8 bytes
        //
        // target = 0x10C
        // =================================================

        #1;


        if (fallthrough_pc !== 32'h00000104) begin

            $display(
                "FAIL: fallthrough=%h expected=00000104",
                fallthrough_pc
            );

            $fatal;

        end


        if (branch_target !== 32'h0000010C) begin

            $display(
                "FAIL: target=%h expected=0000010C",
                branch_target
            );

            $fatal;

        end


        $display(
            "PASS: branch target calculation"
        );


        // =================================================
        // TEST 1
        // ALL LANES TAKEN
        // =================================================

        operand_a = {
            32'd40,
            32'd30,
            32'd20,
            32'd10
        };

        operand_b = {
            32'd40,
            32'd30,
            32'd20,
            32'd10
        };

        active_mask = 4'b1111;

        #1;


        if (taken_mask !== 4'b1111) begin

            $display(
                "FAIL: taken_mask=%b expected=1111",
                taken_mask
            );

            $fatal;

        end


        if (!uniform_taken) begin

            $display(
                "FAIL: expected uniform taken"
            );

            $fatal;

        end


        if (divergent) begin

            $display(
                "FAIL: uniform branch marked divergent"
            );

            $fatal;

        end


        $display(
            "PASS: uniform taken branch"
        );


        // =================================================
        // TEST 2
        // NO LANES TAKEN
        // =================================================

        operand_b = {
            32'd41,
            32'd31,
            32'd21,
            32'd11
        };

        #1;


        if (taken_mask !== 4'b0000) begin

            $display(
                "FAIL: expected no taken lanes"
            );

            $fatal;

        end


        if (!uniform_not_taken) begin

            $display(
                "FAIL: expected uniform not taken"
            );

            $fatal;

        end


        if (divergent) begin

            $display(
                "FAIL: not-taken branch marked divergent"
            );

            $fatal;

        end


        $display(
            "PASS: uniform not-taken branch"
        );


        // =================================================
        // TEST 3
        // DIVERGENCE
        //
        // L0 taken
        // L1 not
        // L2 taken
        // L3 not
        //
        // taken = 0101
        // =================================================

        operand_a = {
            32'd40,
            32'd30,
            32'd20,
            32'd10
        };

        operand_b = {
            32'd99, // L3 false
            32'd30, // L2 true
            32'd99, // L1 false
            32'd10  // L0 true
        };

        #1;


        if (taken_mask !== 4'b0101) begin

            $display(
                "FAIL: taken_mask=%b expected=0101",
                taken_mask
            );

            $fatal;

        end


        if (not_taken_mask !== 4'b1010) begin

            $display(
                "FAIL: not_taken_mask=%b expected=1010",
                not_taken_mask
            );

            $fatal;

        end


        if (!divergent) begin

            $display(
                "FAIL: divergent branch not detected"
            );

            $fatal;

        end


        $display(
            "PASS: divergent branch detected"
        );


        // =================================================
        // TEST 4
        // ACTIVE MASK
        //
        // Only L0/L2 active.
        // Both take branch.
        //
        // This is UNIFORM among ACTIVE lanes.
        // =================================================

        active_mask = 4'b0101;

        operand_a = {
            32'd0,
            32'd30,
            32'd0,
            32'd10
        };

        operand_b = {
            32'd999,
            32'd30,
            32'd999,
            32'd10
        };

        #1;


        if (taken_mask !== 4'b0101) begin

            $display(
                "FAIL: masked taken=%b expected=0101",
                taken_mask
            );

            $fatal;

        end


        if (!uniform_taken) begin

            $display(
                "FAIL: active lanes should be uniformly taken"
            );

            $fatal;

        end


        if (divergent) begin

            $display(
                "FAIL: inactive lanes caused divergence"
            );

            $fatal;

        end


        $display(
            "PASS: inactive lanes ignored correctly"
        );


        // =================================================
        // TEST 5
        // NEGATIVE BRANCH OFFSET
        //
        // PC = 0x100
        // imm = -2
        //
        // PC + 4 - 8 = 0xFC
        // =================================================

        current_pc = 32'h00000100;
        immediate  = 32'hFFFFFFFE;

        #1;


        if (branch_target !== 32'h000000FC) begin

            $display(
                "FAIL: negative target=%h expected=000000FC",
                branch_target
            );

            $fatal;

        end


        $display(
            "PASS: negative branch displacement"
        );


        $display("");
        $display(
            "ALL BRANCH UNIT TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule