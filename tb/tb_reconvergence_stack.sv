`timescale 1ns/1ps

module tb_reconvergence_stack;

    logic clk;
    logic reset;

    logic [1:0] warp_id;

    logic push_enable;

    logic [31:0] push_deferred_pc;
    logic [3:0]  push_deferred_mask;

    logic [31:0] push_reconverge_pc;
    logic [3:0]  push_reconverge_mask;

    logic pop_enable;

    logic update_pending_enable;
    logic update_deferred_pending;

    logic [31:0] top_deferred_pc;
    logic [3:0]  top_deferred_mask;

    logic [31:0] top_reconverge_pc;
    logic [3:0]  top_reconverge_mask;

    logic top_deferred_pending;

    logic empty;
    logic full;
    logic [2:0] depth;

    logic overflow_attempt;
    logic underflow_attempt;
    // DUT

    reconvergence_stack dut (

        .clk                     (clk),
        .reset                   (reset),

        .warp_id                 (warp_id),

        .push_enable             (push_enable),

        .push_deferred_pc        (push_deferred_pc),
        .push_deferred_mask      (push_deferred_mask),

        .push_reconverge_pc      (push_reconverge_pc),
        .push_reconverge_mask    (push_reconverge_mask),

        .pop_enable              (pop_enable),

        .update_pending_enable   (update_pending_enable),
        .update_deferred_pending (update_deferred_pending),

        .top_deferred_pc         (top_deferred_pc),
        .top_deferred_mask       (top_deferred_mask),

        .top_reconverge_pc       (top_reconverge_pc),
        .top_reconverge_mask     (top_reconverge_mask),

        .top_deferred_pending    (top_deferred_pending),

        .empty                   (empty),
        .full                    (full),
        .depth                   (depth),

        .overflow_attempt        (overflow_attempt),
        .underflow_attempt       (underflow_attempt)

    );
    // CLOCK

    always #5 clk = ~clk;
    // WAVEFORM

    initial begin

        $dumpfile("tb_reconvergence_stack.vcd");
        $dumpvars(0, tb_reconvergence_stack);

    end
    // PUSH HELPER

    task push_entry (
        input logic [1:0]  warp,
        input logic [31:0] deferred_pc,
        input logic [3:0]  deferred_mask,
        input logic [31:0] reconverge_pc,
        input logic [3:0]  reconverge_mask
    );

        begin

            warp_id = warp;

            push_deferred_pc     = deferred_pc;
            push_deferred_mask   = deferred_mask;

            push_reconverge_pc   = reconverge_pc;
            push_reconverge_mask = reconverge_mask;

            push_enable = 1'b1;

            @(posedge clk);
            #1;

            push_enable = 1'b0;

        end

    endtask
    // POP HELPER

    task pop_entry (
        input logic [1:0] warp
    );

        begin

            warp_id = warp;

            pop_enable = 1'b1;

            @(posedge clk);
            #1;

            pop_enable = 1'b0;

        end

    endtask
    // MAIN TEST

    initial begin

        clk = 0;
        reset = 1;

        warp_id = 0;

        push_enable = 0;

        push_deferred_pc = 0;
        push_deferred_mask = 0;

        push_reconverge_pc = 0;
        push_reconverge_mask = 0;

        pop_enable = 0;

        update_pending_enable = 0;
        update_deferred_pending = 0;

        $display("");
        $display(
            "=== Snackwell Reconvergence Stack Test ==="
        );
        $display("");
        // RESET

        @(posedge clk);
        #1;

        reset = 0;

        warp_id = 2'd0;

        #1;

        if (!empty) begin

            $display(
                "FAIL: W0 stack not empty after reset"
            );

            $fatal;

        end

        if (depth !== 3'd0) begin

            $display(
                "FAIL: W0 depth=%0d expected=0",
                depth
            );

            $fatal;

        end

        $display(
            "PASS: reset cleared stack"
        );

        // TEST 1
        // PUSH FIRST DIVERGENCE FOR W0

        push_entry(
            2'd0,

            32'h00000120,
            4'b1010,

            32'h00000140,
            4'b1111
        );

        warp_id = 2'd0;

        #1;

        if (depth !== 3'd1)
            $fatal;

        if (top_deferred_pc !== 32'h00000120) begin

            $display(
                "FAIL: deferred PC incorrect"
            );

            $fatal;

        end

        if (top_deferred_mask !== 4'b1010) begin

            $display(
                "FAIL: deferred mask incorrect"
            );

            $fatal;

        end

        if (top_reconverge_pc !== 32'h00000140) begin

            $display(
                "FAIL: reconvergence PC incorrect"
            );

            $fatal;

        end

        if (top_reconverge_mask !== 4'b1111) begin

            $display(
                "FAIL: reconvergence mask incorrect"
            );

            $fatal;

        end

        if (!top_deferred_pending) begin

            $display(
                "FAIL: deferred path should be pending"
            );

            $fatal;

        end

        $display(
            "PASS: divergence entry pushed"
        );

        // TEST 2
        //
        // TAKEN PATH FINISHED.
        //
        // Switch to deferred path and mark it consumed.

        update_deferred_pending = 1'b0;
        update_pending_enable   = 1'b1;

        @(posedge clk);
        #1;

        update_pending_enable = 1'b0;

        if (top_deferred_pending !== 1'b0) begin

            $display(
                "FAIL: deferred pending bit did not clear"
            );

            $fatal;

        end

        if (depth !== 3'd1) begin

            $display(
                "FAIL: updating top changed stack depth"
            );

            $fatal;

        end

        $display(
            "PASS: deferred path state updated"
        );

        // TEST 3
        // POP AFTER BOTH PATHS COMPLETE

        pop_entry(2'd0);

        #1;

        if (!empty) begin

            $display(
                "FAIL: stack not empty after pop"
            );

            $fatal;

        end

        $display(
            "PASS: reconvergence entry popped"
        );

        // TEST 4
        // PER-WARP INDEPENDENCE

        push_entry(
            2'd0,
            32'h00000200,
            4'b0011,
            32'h00000240,
            4'b1111
        );

        push_entry(
            2'd1,
            32'h00000300,
            4'b1100,
            32'h00000340,
            4'b1111
        );

        warp_id = 2'd0;
        #1;

        if (
            top_deferred_pc !==
            32'h00000200
        ) begin

            $display(
                "FAIL: W0 stack corrupted"
            );

            $fatal;

        end

        warp_id = 2'd1;
        #1;

        if (
            top_deferred_pc !==
            32'h00000300
        ) begin

            $display(
                "FAIL: W1 stack corrupted"
            );

            $fatal;

        end

        $display(
            "PASS: reconvergence stacks independent per warp"
        );

        // TEST 5
        // NESTED DIVERGENCE
        //
        // W2 gets four nested entries.

        push_entry(
            2'd2,
            32'h00000400,
            4'b1000,
            32'h00000440,
            4'b1111
        );

        push_entry(
            2'd2,
            32'h00000500,
            4'b0100,
            32'h00000540,
            4'b1111
        );

        push_entry(
            2'd2,
            32'h00000600,
            4'b0010,
            32'h00000640,
            4'b1111
        );

        push_entry(
            2'd2,
            32'h00000700,
            4'b0001,
            32'h00000740,
            4'b1111
        );

        warp_id = 2'd2;
        #1;

        if (!full) begin

            $display(
                "FAIL: W2 stack should be full"
            );

            $fatal;

        end

        if (depth !== 3'd4) begin

            $display(
                "FAIL: W2 depth=%0d expected=4",
                depth
            );

            $fatal;

        end

        if (
            top_deferred_pc !==
            32'h00000700
        ) begin

            $display(
                "FAIL: top of nested stack incorrect"
            );

            $fatal;

        end

        $display(
            "PASS: four-level nested divergence supported"
        );

        // TEST 6
        // OVERFLOW DETECTION

        push_enable = 1'b1;

        #1;

        if (!overflow_attempt) begin

            $display(
                "FAIL: stack overflow not detected"
            );

            $fatal;

        end

        @(posedge clk);
        #1;

        push_enable = 1'b0;

        if (depth !== 3'd4) begin

            $display(
                "FAIL: overflow changed stack depth"
            );

            $fatal;

        end

        $display(
            "PASS: stack overflow safely rejected"
        );

        // TEST 7
        // LIFO ORDER

        pop_entry(2'd2);

        warp_id = 2'd2;
        #1;

        if (
            top_deferred_pc !==
            32'h00000600
        ) begin

            $display(
                "FAIL: LIFO order incorrect after pop"
            );

            $fatal;

        end

        pop_entry(2'd2);

        if (
            top_deferred_pc !==
            32'h00000500
        ) begin

            $display(
                "FAIL: second LIFO pop incorrect"
            );

            $fatal;

        end

        $display(
            "PASS: nested entries pop in LIFO order"
        );

        // TEST 8
        // UNDERFLOW DETECTION

        // Empty W3 has never been touched.

        warp_id = 2'd3;
        pop_enable = 1'b1;

        #1;

        if (!underflow_attempt) begin

            $display(
                "FAIL: underflow not detected"
            );

            $fatal;

        end

        @(posedge clk);
        #1;

        pop_enable = 1'b0;

        if (depth !== 3'd0) begin

            $display(
                "FAIL: underflow changed stack depth"
            );

            $fatal;

        end

        $display(
            "PASS: stack underflow safely rejected"
        );

        $display("");
        $display(
            "ALL RECONVERGENCE STACK TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule
