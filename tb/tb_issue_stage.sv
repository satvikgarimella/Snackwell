`timescale 1ns/1ps

module tb_issue_stage;

    logic       candidate_valid;
    logic [1:0] candidate_warp_id;

    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;
    logic [3:0] rd_addr;

    logic uses_rs1;
    logic uses_rs2;
    logic writes_rd;

    logic rs1_busy;
    logic rs2_busy;

    logic execution_unit_ready;

    logic issue_accept;

    logic dependency_stall;
    logic resource_stall;

    logic       mark_busy_enable;
    logic [1:0] mark_busy_warp_id;
    logic [3:0] mark_busy_reg;


    // =================================================
    // DUT
    // =================================================

    issue_stage dut (

        .candidate_valid      (candidate_valid),
        .candidate_warp_id    (candidate_warp_id),

        .rs1_addr             (rs1_addr),
        .rs2_addr             (rs2_addr),
        .rd_addr              (rd_addr),

        .uses_rs1             (uses_rs1),
        .uses_rs2             (uses_rs2),
        .writes_rd            (writes_rd),

        .rs1_busy             (rs1_busy),
        .rs2_busy             (rs2_busy),

        .execution_unit_ready (execution_unit_ready),

        .issue_accept         (issue_accept),

        .dependency_stall     (dependency_stall),
        .resource_stall       (resource_stall),

        .mark_busy_enable     (mark_busy_enable),
        .mark_busy_warp_id    (mark_busy_warp_id),
        .mark_busy_reg        (mark_busy_reg)

    );


    // =================================================
    // WAVEFORM
    // =================================================

    initial begin

        $dumpfile("tb_issue_stage.vcd");
        $dumpvars(0, tb_issue_stage);

    end


    // =================================================
    // EXPECT HELPER
    // =================================================

    task expect_result (
        input logic expected_accept,
        input logic expected_dependency_stall,
        input logic expected_resource_stall,
        input logic expected_mark_busy
    );

        begin

            #1;


            if (issue_accept !== expected_accept) begin

                $display(
                    "FAIL: issue_accept=%b expected=%b",
                    issue_accept,
                    expected_accept
                );

                $fatal;

            end


            if (
                dependency_stall !==
                expected_dependency_stall
            ) begin

                $display(
                    "FAIL: dependency_stall=%b expected=%b",
                    dependency_stall,
                    expected_dependency_stall
                );

                $fatal;

            end


            if (
                resource_stall !==
                expected_resource_stall
            ) begin

                $display(
                    "FAIL: resource_stall=%b expected=%b",
                    resource_stall,
                    expected_resource_stall
                );

                $fatal;

            end


            if (
                mark_busy_enable !==
                expected_mark_busy
            ) begin

                $display(
                    "FAIL: mark_busy_enable=%b expected=%b",
                    mark_busy_enable,
                    expected_mark_busy
                );

                $fatal;

            end


            $display(
                "PASS: accept=%b dep_stall=%b resource_stall=%b mark_busy=%b",
                issue_accept,
                dependency_stall,
                resource_stall,
                mark_busy_enable
            );

        end

    endtask


    // =================================================
    // MAIN TEST
    // =================================================

    initial begin

        candidate_valid   = 1'b0;
        candidate_warp_id = 2'd0;

        rs1_addr = 4'd0;
        rs2_addr = 4'd0;
        rd_addr  = 4'd0;

        uses_rs1 = 1'b0;
        uses_rs2 = 1'b0;
        writes_rd = 1'b0;

        rs1_busy = 1'b0;
        rs2_busy = 1'b0;

        execution_unit_ready = 1'b1;


        $display("");
        $display("=== Snackwell Issue Stage Test ===");
        $display("");


        // =================================================
        // TEST 1
        // NO CANDIDATE
        // =================================================

        expect_result(
            1'b0,
            1'b0,
            1'b0,
            1'b0
        );


        $display("PASS: invalid candidate does not issue");
        $display("");


        // =================================================
        // TEST 2
        // ADD R3,R1,R2
        //
        // Both sources ready.
        // =================================================

        candidate_valid   = 1'b1;
        candidate_warp_id = 2'd2;

        rs1_addr = 4'd1;
        rs2_addr = 4'd2;
        rd_addr  = 4'd3;

        uses_rs1  = 1'b1;
        uses_rs2  = 1'b1;
        writes_rd = 1'b1;

        rs1_busy = 1'b0;
        rs2_busy = 1'b0;

        execution_unit_ready = 1'b1;


        expect_result(
            1'b1,
            1'b0,
            1'b0,
            1'b1
        );


        if (
            mark_busy_warp_id !== 2'd2 ||
            mark_busy_reg !== 4'd3
        ) begin

            $display(
                "FAIL: incorrect scoreboard allocation"
            );

            $fatal;

        end


        $display(
            "PASS: ready ADD instruction accepted"
        );
        $display("");


        // =================================================
        // TEST 3
        // RAW HAZARD ON RS1
        // =================================================

        rs1_busy = 1'b1;
        rs2_busy = 1'b0;


        expect_result(
            1'b0,
            1'b1,
            1'b0,
            1'b0
        );


        $display(
            "PASS: RS1 dependency blocks issue"
        );
        $display("");


        // =================================================
        // TEST 4
        // RAW HAZARD ON RS2
        // =================================================

        rs1_busy = 1'b0;
        rs2_busy = 1'b1;


        expect_result(
            1'b0,
            1'b1,
            1'b0,
            1'b0
        );


        $display(
            "PASS: RS2 dependency blocks issue"
        );
        $display("");


        // =================================================
        // TEST 5
        // BOTH SOURCES BUSY
        // =================================================

        rs1_busy = 1'b1;
        rs2_busy = 1'b1;


        expect_result(
            1'b0,
            1'b1,
            1'b0,
            1'b0
        );


        $display(
            "PASS: multiple dependencies block issue"
        );
        $display("");


        // =================================================
        // TEST 6
        // UNUSED RS2 MUST NOT CAUSE A STALL
        //
        // Example:
        // ADDI R3,R1,10
        // =================================================

        uses_rs1 = 1'b1;
        uses_rs2 = 1'b0;
        writes_rd = 1'b1;

        rs1_busy = 1'b0;
        rs2_busy = 1'b1;


        expect_result(
            1'b1,
            1'b0,
            1'b0,
            1'b1
        );


        $display(
            "PASS: unused RS2 ignored"
        );
        $display("");


        // =================================================
        // TEST 7
        // UNUSED RS1 MUST NOT CAUSE A STALL
        // =================================================

        uses_rs1 = 1'b0;
        uses_rs2 = 1'b1;

        rs1_busy = 1'b1;
        rs2_busy = 1'b0;


        expect_result(
            1'b1,
            1'b0,
            1'b0,
            1'b1
        );


        $display(
            "PASS: unused RS1 ignored"
        );
        $display("");


        // =================================================
        // TEST 8
        // EXECUTION UNIT BUSY
        // =================================================

        uses_rs1 = 1'b1;
        uses_rs2 = 1'b1;
        writes_rd = 1'b1;

        rs1_busy = 1'b0;
        rs2_busy = 1'b0;

        execution_unit_ready = 1'b0;


        expect_result(
            1'b0,
            1'b0,
            1'b1,
            1'b0
        );


        $display(
            "PASS: execution resource stall detected"
        );
        $display("");


        // =================================================
        // TEST 9
        // DEPENDENCY TAKES PRECEDENCE
        // =================================================

        rs1_busy = 1'b1;
        execution_unit_ready = 1'b0;


        expect_result(
            1'b0,
            1'b1,
            1'b0,
            1'b0
        );


        $display(
            "PASS: dependency stall classified correctly"
        );
        $display("");


        // =================================================
        // TEST 10
        // EXIT-LIKE INSTRUCTION
        //
        // No sources, no destination.
        // It should issue without allocating scoreboard.
        // =================================================

        uses_rs1  = 1'b0;
        uses_rs2  = 1'b0;
        writes_rd = 1'b0;

        rs1_busy = 1'b1;
        rs2_busy = 1'b1;

        execution_unit_ready = 1'b1;


        expect_result(
            1'b1,
            1'b0,
            1'b0,
            1'b0
        );


        $display(
            "PASS: source-free instruction issues correctly"
        );
        $display("");


        // =================================================
        // TEST 11
        // Destination-less instruction
        //
        // It may issue but must not mark anything busy.
        // =================================================

        uses_rs1  = 1'b1;
        uses_rs2  = 1'b1;
        writes_rd = 1'b0;

        rs1_busy = 1'b0;
        rs2_busy = 1'b0;


        expect_result(
            1'b1,
            1'b0,
            1'b0,
            1'b0
        );


        $display(
            "PASS: destination-less instruction does not allocate scoreboard"
        );


        // =================================================

        $display("");
        $display(
            "ALL ISSUE STAGE TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule