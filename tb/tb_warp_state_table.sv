`timescale 1ns/1ps

module tb_warp_state_table;

    logic clk;
    logic reset;
    // READ

    logic [1:0]  read_warp_id;

    logic [31:0] read_pc;
    logic [3:0]  read_active_mask;

    logic read_valid;
    logic read_ready;
    logic read_halted;
    // UPDATE

    logic        update_enable;
    logic [1:0]  update_warp_id;

    logic        update_pc_enable;
    logic [31:0] update_pc;

    logic        update_mask_enable;
    logic [3:0]  update_active_mask;

    logic        update_valid_enable;
    logic        update_valid;

    logic        update_ready_enable;
    logic        update_ready;

    logic        update_halted_enable;
    logic        update_halted;
    // STATUS MASKS

    logic [3:0] valid_mask;
    logic [3:0] ready_mask;
    logic [3:0] halted_mask;
    // DUT

    warp_state_table dut (

        .clk                   (clk),
        .reset                 (reset),

        .read_warp_id          (read_warp_id),

        .read_pc               (read_pc),
        .read_active_mask      (read_active_mask),
        .read_valid            (read_valid),
        .read_ready            (read_ready),
        .read_halted           (read_halted),

        .update_enable         (update_enable),
        .update_warp_id        (update_warp_id),

        .update_pc_enable      (update_pc_enable),
        .update_pc             (update_pc),

        .update_mask_enable    (update_mask_enable),
        .update_active_mask    (update_active_mask),

        .update_valid_enable   (update_valid_enable),
        .update_valid          (update_valid),

        .update_ready_enable   (update_ready_enable),
        .update_ready          (update_ready),

        .update_halted_enable  (update_halted_enable),
        .update_halted         (update_halted),

        .valid_mask            (valid_mask),
        .ready_mask            (ready_mask),
        .halted_mask           (halted_mask)

    );
    // CLOCK

    always #5 clk = ~clk;
    // WAVEFORM

    initial begin

        $dumpfile("tb_warp_state_table.vcd");
        $dumpvars(0, tb_warp_state_table);

    end
    // CLEAR UPDATE CONTROLS

    task clear_updates;

        begin

            update_enable = 0;

            update_pc_enable = 0;
            update_mask_enable = 0;
            update_valid_enable = 0;
            update_ready_enable = 0;
            update_halted_enable = 0;

        end

    endtask
    // INITIALIZE A WARP

    task initialize_warp (
        input logic [1:0]  warp_id,
        input logic [31:0] pc_value,
        input logic [3:0]  active_mask_value
    );

        begin

            update_enable  = 1;
            update_warp_id = warp_id;

            update_pc_enable = 1;
            update_pc        = pc_value;

            update_mask_enable = 1;
            update_active_mask = active_mask_value;

            update_valid_enable = 1;
            update_valid        = 1;

            update_ready_enable = 1;
            update_ready        = 1;

            update_halted_enable = 1;
            update_halted        = 0;

            @(posedge clk);
            #1;

            clear_updates();

        end

    endtask
    // CHECK WARP STATE

    task check_warp (
        input logic [1:0]  warp_id,
        input logic [31:0] expected_pc,
        input logic [3:0]  expected_mask,
        input logic        expected_valid,
        input logic        expected_ready,
        input logic        expected_halted
    );

        begin

            read_warp_id = warp_id;

            #1;

            if (read_pc !== expected_pc) begin

                $display(
                    "FAIL Warp %0d PC=%h expected=%h",
                    warp_id,
                    read_pc,
                    expected_pc
                );

                $fatal;

            end

            if (read_active_mask !== expected_mask) begin

                $display(
                    "FAIL Warp %0d mask=%b expected=%b",
                    warp_id,
                    read_active_mask,
                    expected_mask
                );

                $fatal;

            end

            if (read_valid !== expected_valid) begin

                $display(
                    "FAIL Warp %0d valid",
                    warp_id
                );

                $fatal;

            end

            if (read_ready !== expected_ready) begin

                $display(
                    "FAIL Warp %0d ready",
                    warp_id
                );

                $fatal;

            end

            if (read_halted !== expected_halted) begin

                $display(
                    "FAIL Warp %0d halted",
                    warp_id
                );

                $fatal;

            end

            $display(
                "PASS Warp %0d: PC=%h mask=%b valid=%b ready=%b halted=%b",
                warp_id,
                read_pc,
                read_active_mask,
                read_valid,
                read_ready,
                read_halted
            );

        end

    endtask
    // MAIN TEST

    initial begin

        clk = 0;
        reset = 1;

        read_warp_id = 0;

        update_warp_id = 0;
        update_pc = 0;
        update_active_mask = 0;

        update_valid = 0;
        update_ready = 0;
        update_halted = 0;

        clear_updates();

        $display("");
        $display("=== Snackwell Warp State Table Test ===");
        $display("");
        // RESET

        @(posedge clk);
        #1;

        reset = 0;

        // Every warp should be empty

        check_warp(
            2'd0,
            32'd0,
            4'b0000,
            1'b0,
            1'b0,
            1'b0
        );

        check_warp(
            2'd1,
            32'd0,
            4'b0000,
            1'b0,
            1'b0,
            1'b0
        );

        $display("PASS: reset cleared warp state");
        $display("");
        // INITIALIZE FOUR WARPS

        initialize_warp(
            2'd0,
            32'h00000000,
            4'b1111
        );

        initialize_warp(
            2'd1,
            32'h00000100,
            4'b1111
        );

        initialize_warp(
            2'd2,
            32'h00000200,
            4'b1011
        );

        initialize_warp(
            2'd3,
            32'h00000300,
            4'b0111
        );
        // CHECK ALL FOUR

        check_warp(
            2'd0,
            32'h00000000,
            4'b1111,
            1,
            1,
            0
        );

        check_warp(
            2'd1,
            32'h00000100,
            4'b1111,
            1,
            1,
            0
        );

        check_warp(
            2'd2,
            32'h00000200,
            4'b1011,
            1,
            1,
            0
        );

        check_warp(
            2'd3,
            32'h00000300,
            4'b0111,
            1,
            1,
            0
        );
        // VERIFY STATUS MASKS

        #1;

        if (valid_mask !== 4'b1111) begin
            $display("FAIL valid_mask=%b", valid_mask);
            $fatal;
        end

        if (ready_mask !== 4'b1111) begin
            $display("FAIL ready_mask=%b", ready_mask);
            $fatal;
        end

        if (halted_mask !== 4'b0000) begin
            $display("FAIL halted_mask=%b", halted_mask);
            $fatal;
        end

        $display(
            "PASS: scheduler masks valid=%b ready=%b halted=%b",
            valid_mask,
            ready_mask,
            halted_mask
        );
        // UPDATE ONLY WARP 1 PC

        update_enable    = 1;
        update_warp_id   = 2'd1;

        update_pc_enable = 1;
        update_pc        = 32'h00000104;

        @(posedge clk);
        #1;

        clear_updates();

        check_warp(
            2'd1,
            32'h00000104,
            4'b1111,
            1,
            1,
            0
        );

        $display("PASS: independent PC update");
        // STALL WARP 2

        update_enable      = 1;
        update_warp_id     = 2'd2;

        update_ready_enable = 1;
        update_ready        = 0;

        @(posedge clk);
        #1;

        clear_updates();

        check_warp(
            2'd2,
            32'h00000200,
            4'b1011,
            1,
            0,
            0
        );

        if (ready_mask !== 4'b1011) begin

            $display(
                "FAIL: ready mask after stall=%b expected=1011",
                ready_mask
            );

            $fatal;

        end

        $display(
            "PASS: Warp 2 stalled, ready_mask=%b",
            ready_mask
        );
        // HALT WARP 3

        update_enable        = 1;
        update_warp_id       = 2'd3;

        update_ready_enable  = 1;
        update_ready         = 0;

        update_halted_enable = 1;
        update_halted        = 1;

        @(posedge clk);
        #1;

        clear_updates();

        check_warp(
            2'd3,
            32'h00000300,
            4'b0111,
            1,
            0,
            1
        );

        if (halted_mask !== 4'b1000) begin

            $display(
                "FAIL halted_mask=%b expected=1000",
                halted_mask
            );

            $fatal;

        end

        $display(
            "PASS: Warp 3 halted"
        );
        // CHANGE ACTIVE MASK FOR WARP 0

        update_enable       = 1;
        update_warp_id      = 2'd0;

        update_mask_enable  = 1;
        update_active_mask  = 4'b0101;

        @(posedge clk);
        #1;

        clear_updates();

        check_warp(
            2'd0,
            32'h00000000,
            4'b0101,
            1,
            1,
            0
        );

        $display(
            "PASS: active mask updated independently"
        );

        $display("");
        $display(
            "ALL WARP STATE TABLE TESTS PASSED"
        );
        $display("");

        $finish;

    end

endmodule
