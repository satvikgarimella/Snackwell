`timescale 1ns/1ps

module tb_load_store_unit;

    logic [31:0] base_addr;
    logic [31:0] offset;

    logic [127:0] store_data;

    logic [3:0] active_mask;

    logic is_load;
    logic is_store;

    logic [127:0] lane_addresses;
    logic [127:0] store_write_data;

    logic [3:0] load_enable_mask;
    logic [3:0] store_enable_mask;


    load_store_unit dut (

        .base_addr         (base_addr),
        .offset            (offset),

        .store_data        (store_data),

        .active_mask       (active_mask),

        .is_load           (is_load),
        .is_store          (is_store),

        .lane_addresses    (lane_addresses),
        .store_write_data  (store_write_data),

        .load_enable_mask  (load_enable_mask),
        .store_enable_mask (store_enable_mask)

    );


    initial begin

        $dumpfile("tb_load_store_unit.vcd");
        $dumpvars(0, tb_load_store_unit);

    end


    initial begin

        base_addr  = 32'd0;
        offset     = 32'd0;

        store_data = 128'd0;

        active_mask = 4'b0000;

        is_load  = 1'b0;
        is_store = 1'b0;


        $display("");
        $display("=== Snackwell Load/Store Unit Test ===");
        $display("");


        // =================================================
        // TEST 1
        // ADDRESS GENERATION
        // =================================================

        base_addr = 32'h00001000;
        offset    = 32'h00000020;

        #1;


        if (lane_addresses[31:0] !== 32'h00001020) begin

            $display(
                "FAIL: lane0 address=%h expected=00001020",
                lane_addresses[31:0]
            );

            $fatal;

        end


        if (lane_addresses[63:32] !== 32'h00001020) begin

            $display(
                "FAIL: lane1 address=%h expected=00001020",
                lane_addresses[63:32]
            );

            $fatal;

        end


        if (lane_addresses[95:64] !== 32'h00001020) begin

            $display(
                "FAIL: lane2 address=%h expected=00001020",
                lane_addresses[95:64]
            );

            $fatal;

        end


        if (lane_addresses[127:96] !== 32'h00001020) begin

            $display(
                "FAIL: lane3 address=%h expected=00001020",
                lane_addresses[127:96]
            );

            $fatal;

        end


        $display(
            "PASS: effective address generation"
        );


        // =================================================
        // TEST 2
        // LOAD ALL ACTIVE LANES
        // =================================================

        active_mask = 4'b1111;

        is_load  = 1'b1;
        is_store = 1'b0;

        #1;


        if (load_enable_mask !== 4'b1111) begin

            $display(
                "FAIL: load mask=%b expected=1111",
                load_enable_mask
            );

            $fatal;

        end


        if (store_enable_mask !== 4'b0000) begin

            $display(
                "FAIL: store mask active during load"
            );

            $fatal;

        end


        $display(
            "PASS: full active-mask load"
        );


        // =================================================
        // TEST 3
        // MASKED LOAD
        // =================================================

        active_mask = 4'b0101;

        #1;


        if (load_enable_mask !== 4'b0101) begin

            $display(
                "FAIL: masked load=%b expected=0101",
                load_enable_mask
            );

            $fatal;

        end


        $display(
            "PASS: masked load respects active lanes"
        );


        // =================================================
        // TEST 4
        // STORE
        // =================================================

        is_load  = 1'b0;
        is_store = 1'b1;

        active_mask = 4'b1011;

        store_data = {
            32'd40,
            32'd30,
            32'd20,
            32'd10
        };

        #1;


        if (store_enable_mask !== 4'b1011) begin

            $display(
                "FAIL: store mask=%b expected=1011",
                store_enable_mask
            );

            $fatal;

        end


        if (
            store_write_data !== {
                32'd40,
                32'd30,
                32'd20,
                32'd10
            }
        ) begin

            $display(
                "FAIL: store data corrupted"
            );

            $fatal;

        end


        $display(
            "PASS: store data + active mask"
        );


        // =================================================
        // TEST 5
        // NO MEMORY OP
        // =================================================

        is_load  = 1'b0;
        is_store = 1'b0;

        active_mask = 4'b1111;

        #1;


        if (load_enable_mask !== 4'b0000) begin

            $display(
                "FAIL: load mask active during idle"
            );

            $fatal;

        end


        if (store_enable_mask !== 4'b0000) begin

            $display(
                "FAIL: store mask active during idle"
            );

            $fatal;

        end


        $display(
            "PASS: idle LSU produces no requests"
        );


        // =================================================

        $display("");
        $display(
            "ALL LOAD/STORE UNIT TESTS PASSED"
        );
        $display("");


        $finish;

    end

endmodule