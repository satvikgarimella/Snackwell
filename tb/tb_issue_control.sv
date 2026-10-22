`timescale 1ns/1ps

module tb_issue_control;

    logic clk;
    logic reset;

    logic [3:0] valid_mask;
    logic [3:0] ready_mask;
    logic [3:0] halted_mask;

    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;
    logic [3:0] rd_addr;

    logic uses_rs1;
    logic uses_rs2;
    logic writes_rd;

    logic execution_unit_ready;

    logic       clear_busy_enable;
    logic [1:0] clear_busy_warp_id;
    logic [3:0] clear_busy_reg;

    logic       issue_valid;
    logic       issue_accept;
    logic [1:0] selected_warp_id;

    logic dependency_stall;
    logic resource_stall;

    logic [3:0] blocked_mask;

    logic rs1_busy;
    logic rs2_busy;

    logic [15:0] selected_warp_busy_mask;
    logic [3:0] eligible_mask;
    // DUT

    issue_control dut (

        .clk                     (clk),
        .reset                   (reset),

        .valid_mask              (valid_mask),
        .ready_mask              (ready_mask),
        .halted_mask             (halted_mask),

        .rs1_addr                (rs1_addr),
        .rs2_addr                (rs2_addr),
        .rd_addr                 (rd_addr),

        .uses_rs1                (uses_rs1),
        .uses_rs2                (uses_rs2),
        .writes_rd               (writes_rd),
        .additional_dependency   (1'b0),

        .execution_unit_ready    (execution_unit_ready),

        .clear_busy_enable       (clear_busy_enable),
        .clear_busy_warp_id      (clear_busy_warp_id),
        .clear_busy_reg          (clear_busy_reg),

        .issue_valid             (issue_valid),
        .issue_accept            (issue_accept),
        .selected_warp_id        (selected_warp_id),

        .dependency_stall        (dependency_stall),
        .resource_stall          (resource_stall),

        .blocked_mask            (blocked_mask),

        .rs1_busy                (rs1_busy),
        .rs2_busy                (rs2_busy),

        .selected_warp_busy_mask (selected_warp_busy_mask),
        .eligible_mask           (eligible_mask)

    );
    // CLOCK

    always #5 clk = ~clk;
    // WAVEFORM

    initial begin

        $dumpfile("tb_issue_control.vcd");
        $dumpvars(0, tb_issue_control);

    end
    // CLEAR SCOREBOARD ENTRY

    task clear_reg (
        input logic [1:0] warp,
        input logic [3:0] reg_id
    );

        begin

            clear_busy_warp_id = warp;
            clear_busy_reg     = reg_id;
            clear_busy_enable  = 1'b1;

            @(posedge clk);
            #1;

            clear_busy_enable = 1'b0;

        end

    endtask
    // MAIN TEST

    initial begin

        clk   = 1'b0;
        reset = 1'b1;

        valid_mask  = 4'b1111;
        ready_mask  = 4'b1111;
        halted_mask = 4'b0000;

        rs1_addr = 4'd1;
        rs2_addr = 4'd2;
        rd_addr  = 4'd3;

        uses_rs1  = 1'b1;
        uses_rs2  = 1'b1;
        writes_rd = 1'b1;

        execution_unit_ready = 1'b1;

        clear_busy_enable  = 1'b0;
        clear_busy_warp_id = 2'd0;
        clear_busy_reg     = 4'd0;

        $display("");
        $display(
            "=== Snackwell Issue Control Integration Test ==="
        );
        $display("");
        // RESET

        @(posedge clk);
        #1;

        reset = 1'b0;

        if (blocked_mask !== 4'b0000) begin

            $display(
                "FAIL: blocked_mask after reset = %b",
                blocked_mask
            );

            $fatal;

        end

        $display("PASS: issue control reset");

        // TEST 1
        //
        // W0:
        // ADD R3,R1,R2
        //
        // R3 becomes busy ONLY for W0.

        #1;

        if (
            selected_warp_id !== 2'd0 ||
            !issue_accept
        ) begin

            $display(
                "FAIL: expected W0 initial issue"
            );

            $fatal;

        end

        if (rd_addr !== 4'd3) begin

            $display(
                "FAIL: W0 should allocate R3"
            );

            $fatal;

        end

        $display(
            "ISSUE: W0 ADD R3,R1,R2"
        );

        @(posedge clk);
        #1;

        if (selected_warp_id !== 2'd1) begin

            $display(
                "FAIL: expected scheduler to move to W1"
            );

            $fatal;

        end

        $display(
            "PASS: W0 issued and scheduler advanced"
        );

        // IMPORTANT
        //
        // W1/W2/W3 write R8 instead of R3.
        //
        // This preserves:
        //
        // W0.R3 = BUSY
        // W1.R3 = READY
        // W2.R3 = READY
        // W3.R3 = READY

        rd_addr = 4'd8;
        // W1

        if (!issue_accept) begin

            $display(
                "FAIL: expected W1 to issue"
            );

            $fatal;

        end

        $display(
            "ISSUE: W1 ADD R8,R1,R2"
        );

        @(posedge clk);
        #1;
        // W2

        if (selected_warp_id !== 2'd2) begin

            $display(
                "FAIL: expected W2"
            );

            $fatal;

        end

        if (!issue_accept) begin

            $display(
                "FAIL: expected W2 to issue"
            );

            $fatal;

        end

        $display(
            "ISSUE: W2 ADD R8,R1,R2"
        );

        @(posedge clk);
        #1;
        // W3

        if (selected_warp_id !== 2'd3) begin

            $display(
                "FAIL: expected W3"
            );

            $fatal;

        end

        if (!issue_accept) begin

            $display(
                "FAIL: expected W3 to issue"
            );

            $fatal;

        end

        $display(
            "ISSUE: W3 ADD R8,R1,R2"
        );

        @(posedge clk);
        #1;
        // SHOULD WRAP TO W0

        if (selected_warp_id !== 2'd0) begin

            $display(
                "FAIL: expected W0 after round robin"
            );

            $fatal;

        end

        $display(
            "PASS: other warps issued independent destinations"
        );

        // TEST 2
        //
        // W0:
        // ADD R4,R3,R2
        //
        // W0.R3 is busy.
        //
        // Other warps' R3 registers are NOT busy.

        rs1_addr = 4'd3;
        rs2_addr = 4'd2;
        rd_addr  = 4'd4;

        #1;

        if (!rs1_busy) begin

            $display(
                "FAIL: expected W0 R3 busy"
            );

            $fatal;

        end

        if (!dependency_stall) begin

            $display(
                "FAIL: expected W0 dependency stall"
            );

            $fatal;

        end

        if (issue_accept) begin

            $display(
                "FAIL: dependent W0 instruction issued"
            );

            $fatal;

        end

        $display(
            "PASS: W0 RAW dependency detected"
        );

        // CLOCK FAILED W0 ISSUE
        //
        // This registers W0 into blocked_mask.

        @(posedge clk);
        #1;

        if (blocked_mask[0] !== 1'b1) begin

            $display(
                "FAIL: W0 not marked blocked"
            );

            $fatal;

        end

        if (selected_warp_id === 2'd0) begin

            $display(
                "FAIL: scheduler did not skip blocked W0"
            );

            $fatal;

        end

        $display(
            "PASS: scheduler switched away from blocked W0"
        );

        // ALTERNATE WARP
        //
        // It also reads R3, but scoreboard state is
        // per-warp.
        //
        // W1.R3 is ready, so W1 can issue.

        if (selected_warp_id !== 2'd1) begin

            $display(
                "FAIL: expected W1 after blocking W0, got W%0d",
                selected_warp_id
            );

            $fatal;

        end

        if (rs1_busy) begin

            $display(
                "FAIL: W1 R3 should not be busy"
            );

            $fatal;

        end

        if (!issue_accept) begin

            $display(
                "FAIL: alternate warp failed to issue"
            );

            $fatal;

        end

        $display(
            "PASS: W1 issued while W0 was dependency blocked"
        );

        // COMPLETE W0.R3
        //
        // Simulates writeback from W0's original ADD.

        clear_reg(
            2'd0,
            4'd3
        );

        if (blocked_mask[0] !== 1'b0) begin

            $display(
                "FAIL: W0 still blocked after writeback"
            );

            $fatal;

        end

        $display(
            "PASS: W0 unblocked after dependency writeback"
        );
        // RUN UNTIL W0 RETURNS

        while (selected_warp_id !== 2'd0) begin

            if (!issue_accept) begin

                $display(
                    "FAIL: scheduler stuck before returning to W0"
                );

                $fatal;

            end

            @(posedge clk);
            #1;

        end
        // W0.R3 SHOULD NOW BE READY

        #1;

        if (rs1_busy) begin

            $display(
                "FAIL: W0 R3 still busy after clear"
            );

            $fatal;

        end

        if (!issue_accept) begin

            $display(
                "FAIL: W0 did not issue after dependency resolved"
            );

            $fatal;

        end

        $display(
            "PASS: W0 became schedulable after dependency resolved"
        );
        // RESOURCE STALL TEST

        execution_unit_ready = 1'b0;

        #1;

        if (!resource_stall) begin

            $display(
                "FAIL: resource stall not detected"
            );

            $fatal;

        end

        if (issue_accept) begin

            $display(
                "FAIL: issued while execution unit unavailable"
            );

            $fatal;

        end

        $display(
            "PASS: execution resource stall detected"
        );

        @(posedge clk);
        #1;

        if (blocked_mask == 4'b0000) begin

            $display(
                "FAIL: resource-stalled warp not blocked"
            );

            $fatal;

        end

        execution_unit_ready = 1'b1;

        $display("");
        $display(
            "ALL ISSUE CONTROL INTEGRATION TESTS PASSED"
        );
        $display("");

        // Resource recovery must not depend on this warp completing a write.
        valid_mask = 4'b0001;
        uses_rs1 = 0; uses_rs2 = 0; writes_rd = 0;
        clear_busy_enable = 0;
        repeat (3) begin
            @(posedge clk);
            #1;
        end
        if (!issue_valid || !issue_accept || selected_warp_id != 0)
            $fatal(1, "Resource-stalled warp did not retry without writeback");
        $display("PASS: resource stall recovery without a same-warp writeback");
        $finish;

    end

endmodule
