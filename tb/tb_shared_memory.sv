`timescale 1ns/1ps

module tb_shared_memory;

    logic clk;
    logic reset;

    logic [127:0] lane_addresses;
    logic [127:0] store_data;

    logic [3:0] load_enable_mask;
    logic [3:0] store_enable_mask;

    logic [127:0] load_data;

    logic [3:0] conflict_mask;
    logic bank_conflict;


    shared_memory dut (

        .clk               (clk),
        .reset             (reset),

        .lane_addresses    (lane_addresses),

        .store_data        (store_data),

        .load_enable_mask  (load_enable_mask),
        .store_enable_mask (store_enable_mask),

        .load_data         (load_data),

        .conflict_mask     (conflict_mask),
        .bank_conflict     (bank_conflict)

    );


    always #5 clk = ~clk;


    initial begin

        $dumpfile("tb_shared_memory.vcd");
        $dumpvars(0, tb_shared_memory);

    end


    initial begin

        clk = 0;
        reset = 1;

        lane_addresses = 128'd0;
        store_data = 128'd0;

        load_enable_mask = 4'b0000;
        store_enable_mask = 4'b0000;


        $display("");
        $display("=== Snackwell Shared Memory Test ===");
        $display("");


        // =================================================
        // RESET
        // =================================================

        @(posedge clk);
        #1;

        reset = 0;


        // =================================================
        // TEST 1
        //
        // FOUR DIFFERENT BANKS
        //
        // L0 -> 0x000 -> bank 0
        // L1 -> 0x004 -> bank 1
        // L2 -> 0x008 -> bank 2
        // L3 -> 0x00C -> bank 3
        // =================================================

        lane_addresses = {
            32'h0000000C,
            32'h00000008,
            32'h00000004,
            32'h00000000
        };

        store_data = {
            32'd40,
            32'd30,
            32'd20,
            32'd10
        };

        store_enable_mask = 4'b1111;

        #1;


        if (bank_conflict) begin

            $display(
                "FAIL: unexpected bank conflict"
            );

            $fatal;

        end


        @(posedge clk);
        #1;

        store_enable_mask = 4'b0000;


        $display(
            "PASS: conflict-free 4-lane store"
        );


        // =================================================
        // LOAD BACK
        // =================================================

        load_enable_mask = 4'b1111;

        #1;


        if (load_data[31:0] !== 32'd10) begin

            $display(
                "FAIL: lane0 load=%0d expected=10",
                load_data[31:0]
            );

            $fatal;

        end


        if (load_data[63:32] !== 32'd20) begin

            $display(
                "FAIL: lane1 load=%0d expected=20",
                load_data[63:32]
            );

            $fatal;

        end


        if (load_data[95:64] !== 32'd30) begin

            $display(
                "FAIL: lane2 load=%0d expected=30",
                load_data[95:64]
            );

            $fatal;

        end


        if (load_data[127:96] !== 32'd40) begin

            $display(
                "FAIL: lane3 load=%0d expected=40",
                load_data[127:96]
            );

            $fatal;

        end


        $display(
            "PASS: conflict-free 4-lane load"
        );


        // =================================================
        // TEST 2
        //
        // BANK CONFLICT
        //
        // 0x000 -> bank0 row0
        // 0x010 -> bank0 row1
        //
        // L0 and L1 conflict.
        // =================================================

        lane_addresses = {
            32'h0000000C,
            32'h00000008,
            32'h00000010,
            32'h00000000
        };

        load_enable_mask = 4'b0011;

        #1;


        if (!bank_conflict) begin

            $display(
                "FAIL: expected bank conflict"
            );

            $fatal;

        end


        if (conflict_mask !== 4'b0011) begin

            $display(
                "FAIL: conflict_mask=%b expected=0011",
                conflict_mask
            );

            $fatal;

        end


        $display(
            "PASS: two-lane bank conflict detected"
        );


        // =================================================
        // TEST 3
        //
        // INACTIVE LANE MUST NOT CONFLICT
        // =================================================

        load_enable_mask = 4'b0001;

        #1;


        if (bank_conflict) begin

            $display(
                "FAIL: inactive lane caused conflict"
            );

            $fatal;

        end


        $display(
            "PASS: inactive lanes ignored by conflict logic"
        );


        // =================================================
        // TEST 4
        //
        // ALL LANES SAME BANK
        // =================================================

        lane_addresses = {
            32'h00000030,
            32'h00000020,
            32'h00000010,
            32'h00000000
        };

        load_enable_mask = 4'b1111;

        #1;


        if (!bank_conflict) begin

            $display(
                "FAIL: expected 4-way bank conflict"
            );

            $fatal;

        end


        if (conflict_mask !== 4'b1111) begin

            $display(
                "FAIL: conflict_mask=%b expected=1111",
                conflict_mask
            );

            $fatal;

        end


        $display(
            "PASS: 4-way bank conflict detected"
        );


        // =================================================
        // TEST 5
        //
        // MASKED STORE
        // =================================================

        load_enable_mask = 4'b0000;

        lane_addresses = {
            32'h0000004C,
            32'h00000048,
            32'h00000044,
            32'h00000040
        };

        store_data = {
            32'd400,
            32'd300,
            32'd200,
            32'd100
        };

        // Only lanes 0 and 2
        store_enable_mask = 4'b0101;

        #1;


        if (bank_conflict) begin

            $display(
                "FAIL: unexpected conflict on masked store"
            );

            $fatal;

        end


        @(posedge clk);
        #1;

        store_enable_mask = 4'b0000;

        load_enable_mask = 4'b1111;

        #1;


        if (load_data[31:0] !== 32'd100) begin

            $display(
                "FAIL: lane0 masked store"
            );

            $fatal;

        end


        if (load_data[63:32] !== 32'd0) begin

            $display(
                "FAIL: lane1 should remain zero"
            );

            $fatal;

        end


        if (load_data[95:64] !== 32'd300) begin

            $display(
                "FAIL: lane2 masked store"
            );

            $fatal;

        end


        if (load_data[127:96] !== 32'd0) begin

            $display(
                "FAIL: lane3 should remain zero"
            );

            $fatal;

        end


        $display(
            "PASS: active-mask controlled store"
        );


        // =================================================

        $display("");
        $display(
            "ALL SHARED MEMORY TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule