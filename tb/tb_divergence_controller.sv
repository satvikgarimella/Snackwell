`timescale 1ns/1ps

module tb_divergence_controller;

    logic [31:0] current_pc;
    logic [3:0] current_active_mask;

    logic branch_valid;

    logic [3:0] taken_mask;
    logic [3:0] not_taken_mask;

    logic uniform_taken;
    logic uniform_not_taken;
    logic divergent;

    logic [31:0] branch_target;
    logic [31:0] fallthrough_pc;

    logic reconvergence_target_valid;
    logic [31:0] reconvergence_target_pc;

    logic stack_empty;

    logic [31:0] stack_deferred_pc;
    logic [3:0] stack_deferred_mask;

    logic [31:0] stack_reconverge_pc;
    logic [3:0] stack_reconverge_mask;

    logic stack_deferred_pending;

    logic pc_update_enable;
    logic [31:0] next_pc;

    logic mask_update_enable;
    logic [3:0] next_active_mask;

    logic stack_push_enable;

    logic [31:0] stack_push_deferred_pc;
    logic [3:0] stack_push_deferred_mask;

    logic [31:0] stack_push_reconverge_pc;
    logic [3:0] stack_push_reconverge_mask;

    logic stack_pop_enable;

    logic stack_update_pending_enable;
    logic stack_update_deferred_pending;

    logic reconvergence_event;
    logic switch_to_deferred_path;
    logic restore_reconverged_mask;

    logic missing_reconvergence_target;

    divergence_controller dut (

        .current_pc                     (current_pc),
        .current_active_mask            (current_active_mask),

        .branch_valid                   (branch_valid),

        .taken_mask                     (taken_mask),
        .not_taken_mask                 (not_taken_mask),

        .uniform_taken                  (uniform_taken),
        .uniform_not_taken              (uniform_not_taken),
        .divergent                      (divergent),

        .branch_target                  (branch_target),
        .fallthrough_pc                 (fallthrough_pc),

        .reconvergence_target_valid     (reconvergence_target_valid),
        .reconvergence_target_pc        (reconvergence_target_pc),

        .stack_empty                    (stack_empty),

        .stack_deferred_pc              (stack_deferred_pc),
        .stack_deferred_mask            (stack_deferred_mask),

        .stack_reconverge_pc            (stack_reconverge_pc),
        .stack_reconverge_mask          (stack_reconverge_mask),

        .stack_deferred_pending         (stack_deferred_pending),

        .pc_update_enable               (pc_update_enable),
        .next_pc                        (next_pc),

        .mask_update_enable             (mask_update_enable),
        .next_active_mask               (next_active_mask),

        .stack_push_enable              (stack_push_enable),

        .stack_push_deferred_pc         (stack_push_deferred_pc),
        .stack_push_deferred_mask       (stack_push_deferred_mask),

        .stack_push_reconverge_pc       (stack_push_reconverge_pc),
        .stack_push_reconverge_mask     (stack_push_reconverge_mask),

        .stack_pop_enable               (stack_pop_enable),

        .stack_update_pending_enable    (stack_update_pending_enable),
        .stack_update_deferred_pending  (stack_update_deferred_pending),

        .reconvergence_event            (reconvergence_event),
        .switch_to_deferred_path        (switch_to_deferred_path),
        .restore_reconverged_mask       (restore_reconverged_mask),

        .missing_reconvergence_target   (missing_reconvergence_target)

    );

    initial begin

        $dumpfile("tb_divergence_controller.vcd");
        $dumpvars(0, tb_divergence_controller);

    end

    initial begin

        current_pc = 32'h100;
        current_active_mask = 4'b1111;

        branch_valid = 0;

        taken_mask = 0;
        not_taken_mask = 0;

        uniform_taken = 0;
        uniform_not_taken = 0;
        divergent = 0;

        branch_target = 32'h200;
        fallthrough_pc = 32'h104;

        reconvergence_target_valid = 0;
        reconvergence_target_pc = 0;

        stack_empty = 1;

        stack_deferred_pc = 0;
        stack_deferred_mask = 0;

        stack_reconverge_pc = 0;
        stack_reconverge_mask = 0;

        stack_deferred_pending = 0;

        $display("");
        $display("=== Snackwell Divergence Controller Test ===");
        $display("");

        // TEST 1
        // UNIFORM TAKEN

        branch_valid = 1;

        uniform_taken = 1;
        uniform_not_taken = 0;
        divergent = 0;

        #1;

        if (
            !pc_update_enable ||
            next_pc !== 32'h200
        ) begin

            $display(
                "FAIL: uniform taken branch"
            );

            $fatal;

        end

        if (mask_update_enable) begin

            $display(
                "FAIL: uniform branch changed active mask"
            );

            $fatal;

        end

        if (stack_push_enable) begin

            $display(
                "FAIL: uniform branch pushed stack"
            );

            $fatal;

        end

        $display(
            "PASS: uniform taken branch"
        );

        // TEST 2
        // UNIFORM NOT TAKEN

        uniform_taken = 0;
        uniform_not_taken = 1;

        #1;

        if (
            !pc_update_enable ||
            next_pc !== 32'h104
        ) begin

            $display(
                "FAIL: uniform not-taken branch"
            );

            $fatal;

        end

        if (stack_push_enable) begin

            $display(
                "FAIL: not-taken branch pushed stack"
            );

            $fatal;

        end

        $display(
            "PASS: uniform not-taken branch"
        );

        // TEST 3
        // DIVERGENT BRANCH
        //
        // taken     = 0101
        // not taken = 1010
        //
        // execute taken first

        uniform_taken = 0;
        uniform_not_taken = 0;
        divergent = 1;

        taken_mask = 4'b0101;
        not_taken_mask = 4'b1010;

        reconvergence_target_valid = 1;
        reconvergence_target_pc = 32'h300;

        #1;

        if (
            !pc_update_enable ||
            next_pc !== 32'h200
        ) begin

            $display(
                "FAIL: divergent branch PC"
            );

            $fatal;

        end

        if (
            !mask_update_enable ||
            next_active_mask !== 4'b0101
        ) begin

            $display(
                "FAIL: divergent taken mask"
            );

            $fatal;

        end

        if (!stack_push_enable) begin

            $display(
                "FAIL: divergent branch did not push stack"
            );

            $fatal;

        end

        if (
            stack_push_deferred_pc !== 32'h104 ||
            stack_push_deferred_mask !== 4'b1010
        ) begin

            $display(
                "FAIL: deferred path incorrectly saved"
            );

            $fatal;

        end

        if (
            stack_push_reconverge_pc !== 32'h300 ||
            stack_push_reconverge_mask !== 4'b1111
        ) begin

            $display(
                "FAIL: reconvergence state incorrectly saved"
            );

            $fatal;

        end

        $display(
            "PASS: divergent branch creates two SIMT paths"
        );

        // TEST 4
        // DIVERGENCE WITHOUT SSY

        reconvergence_target_valid = 0;

        #1;

        if (!missing_reconvergence_target) begin

            $display(
                "FAIL: missing SSY target not detected"
            );

            $fatal;

        end

        if (
            pc_update_enable ||
            mask_update_enable ||
            stack_push_enable
        ) begin

            $display(
                "FAIL: invalid divergence modified state"
            );

            $fatal;

        end

        $display(
            "PASS: divergent branch requires reconvergence target"
        );

        // TEST 5
        // FIRST ARRIVAL AT RECONVERGENCE
        //
        // switch to deferred path

        branch_valid = 0;
        divergent = 0;

        current_pc = 32'h300;

        stack_empty = 0;

        stack_reconverge_pc = 32'h300;

        stack_deferred_pc = 32'h104;
        stack_deferred_mask = 4'b1010;

        stack_reconverge_mask = 4'b1111;

        stack_deferred_pending = 1;

        #1;

        if (!reconvergence_event) begin

            $display(
                "FAIL: reconvergence event not detected"
            );

            $fatal;

        end

        if (!switch_to_deferred_path) begin

            $display(
                "FAIL: did not switch to deferred path"
            );

            $fatal;

        end

        if (
            !pc_update_enable ||
            next_pc !== 32'h104
        ) begin

            $display(
                "FAIL: deferred PC incorrect"
            );

            $fatal;

        end

        if (
            !mask_update_enable ||
            next_active_mask !== 4'b1010
        ) begin

            $display(
                "FAIL: deferred active mask incorrect"
            );

            $fatal;

        end

        if (
            !stack_update_pending_enable ||
            stack_update_deferred_pending !== 1'b0
        ) begin

            $display(
                "FAIL: deferred pending state not updated"
            );

            $fatal;

        end

        if (stack_pop_enable) begin

            $display(
                "FAIL: stack popped too early"
            );

            $fatal;

        end

        $display(
            "PASS: first reconvergence arrival switches paths"
        );

        // TEST 6
        // SECOND ARRIVAL
        //
        // restore full mask + pop

        stack_deferred_pending = 0;

        #1;

        if (!reconvergence_event) begin

            $display(
                "FAIL: second reconvergence not detected"
            );

            $fatal;

        end

        if (!restore_reconverged_mask) begin

            $display(
                "FAIL: original active mask not restored"
            );

            $fatal;

        end

        if (
            !mask_update_enable ||
            next_active_mask !== 4'b1111
        ) begin

            $display(
                "FAIL: reconverged mask incorrect"
            );

            $fatal;

        end

        if (!stack_pop_enable) begin

            $display(
                "FAIL: reconvergence stack not popped"
            );

            $fatal;

        end

        if (pc_update_enable) begin

            $display(
                "FAIL: second reconvergence unexpectedly redirected PC"
            );

            $fatal;

        end

        $display(
            "PASS: second reconvergence arrival restores warp"
        );

        // TEST 7
        // RECONVERGENCE HAS PRIORITY OVER BRANCH

        branch_valid = 1;

        uniform_taken = 1;
        divergent = 0;

        stack_deferred_pending = 1;

        #1;

        if (
            !switch_to_deferred_path ||
            next_pc !== 32'h104
        ) begin

            $display(
                "FAIL: reconvergence priority incorrect"
            );

            $fatal;

        end

        $display(
            "PASS: reconvergence takes priority over fetched branch"
        );

        $display("");
        $display(
            "ALL DIVERGENCE CONTROLLER TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule
