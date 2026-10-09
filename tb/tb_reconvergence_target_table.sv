`timescale 1ns/1ps

module tb_reconvergence_target_table;

    logic clk;
    logic reset;

    logic write_enable;
    logic [1:0] write_warp_id;
    logic [31:0] write_pc;
    logic clear_enable = 0;
    logic [1:0] clear_warp_id = 0;

    logic [1:0] read_warp_id;
    logic [31:0] read_pc;
    logic read_valid;


    reconvergence_target_table dut (

        .clk           (clk),
        .reset         (reset),

        .write_enable  (write_enable),
        .write_warp_id (write_warp_id),
        .write_pc      (write_pc),
        .clear_enable  (clear_enable),
        .clear_warp_id (clear_warp_id),

        .read_warp_id  (read_warp_id),
        .read_pc       (read_pc),
        .read_valid    (read_valid)

    );


    always #5 clk = ~clk;


    initial begin

        $dumpfile(
            "tb_reconvergence_target_table.vcd"
        );

        $dumpvars(
            0,
            tb_reconvergence_target_table
        );

    end


    task write_target (
        input logic [1:0] warp,
        input logic [31:0] pc
    );

        begin

            write_enable  = 1'b1;
            write_warp_id = warp;
            write_pc      = pc;

            @(posedge clk);
            #1;

            write_enable = 1'b0;

        end

    endtask


    initial begin

        clk = 0;
        reset = 1;

        write_enable = 0;
        write_warp_id = 0;
        write_pc = 0;

        read_warp_id = 0;


        $display("");
        $display(
            "=== Snackwell Reconvergence Target Table Test ==="
        );
        $display("");


        // =============================================
        // RESET
        // =============================================

        @(posedge clk);
        #1;

        reset = 0;

        read_warp_id = 2'd0;

        #1;


        if (read_valid !== 1'b0) begin

            $display(
                "FAIL: W0 target valid after reset"
            );

            $fatal;

        end


        $display(
            "PASS: reset clears reconvergence targets"
        );


        // =============================================
        // WRITE W0
        // =============================================

        write_target(
            2'd0,
            32'h00000140
        );


        read_warp_id = 2'd0;
        #1;


        if (
            read_pc !== 32'h00000140 ||
            !read_valid
        ) begin

            $display(
                "FAIL: W0 reconvergence target incorrect"
            );

            $fatal;

        end


        $display(
            "PASS: W0 reconvergence target stored"
        );


        // =============================================
        // WRITE W1
        // =============================================

        write_target(
            2'd1,
            32'h00000280
        );


        read_warp_id = 2'd1;
        #1;


        if (
            read_pc !== 32'h00000280 ||
            !read_valid
        ) begin

            $display(
                "FAIL: W1 target incorrect"
            );

            $fatal;

        end


        // Ensure W0 stayed untouched

        read_warp_id = 2'd0;
        #1;


        if (
            read_pc !== 32'h00000140 ||
            !read_valid
        ) begin

            $display(
                "FAIL: W1 write corrupted W0"
            );

            $fatal;

        end


        $display(
            "PASS: targets independent per warp"
        );


        // =============================================
        // OVERWRITE W0
        // =============================================

        write_target(
            2'd0,
            32'h00000300
        );


        read_warp_id = 2'd0;
        #1;


        if (read_pc !== 32'h00000300) begin

            $display(
                "FAIL: W0 target overwrite failed"
            );

            $fatal;

        end


        $display(
            "PASS: reconvergence target updates correctly"
        );


        $display("");
        $display(
            "ALL RECONVERGENCE TARGET TABLE TESTS PASSED"
        );
        $display("");


        @(negedge clk);
        clear_enable = 1;
        clear_warp_id = 0;
        @(negedge clk);
        clear_enable = 0;
        if (read_valid !== 1'b0) $fatal(1, "Consumed target stayed valid");
        read_warp_id = 1;
        #1;
        if (read_valid !== 1'b1) $fatal(1, "Clear affected another warp");
        $display("PASS: target consumption is isolated per warp");
        $finish;

    end

endmodule
