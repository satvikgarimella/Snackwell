`timescale 1ns/1ps

module tb_program_counter;

    logic clk;
    logic reset;
    logic enable;

    logic [31:0] pc;

    program_counter dut (
        .clk    (clk),
        .reset  (reset),
        .enable (enable),
        .pc     (pc)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_program_counter.vcd");
        $dumpvars(0, tb_program_counter);
    end

    task check_pc(
        input logic [31:0] expected
    );
        begin
            #1;

            if (pc !== expected) begin
                $display(
                    "FAIL: PC=%h expected=%h",
                    pc,
                    expected
                );
                $fatal;
            end
            else begin
                $display(
                    "PASS: PC=%h",
                    pc
                );
            end
        end
    endtask

    initial begin

        clk    = 0;
        reset  = 1;
        enable = 0;

        $display("");
        $display("=== Snackwell Program Counter Test ===");
        $display("");

        // Reset
        @(posedge clk);
        check_pc(32'h00000000);

        reset = 0;

        // Increment
        enable = 1;

        @(posedge clk);
        check_pc(32'h00000004);

        @(posedge clk);
        check_pc(32'h00000008);

        @(posedge clk);
        check_pc(32'h0000000C);

        @(posedge clk);
        check_pc(32'h00000010);

        // Stall
        enable = 0;

        @(posedge clk);
        check_pc(32'h00000010);

        @(posedge clk);
        check_pc(32'h00000010);

        $display("PASS: PC held during stall");

        // Resume
        enable = 1;

        @(posedge clk);
        check_pc(32'h00000014);

        @(posedge clk);
        check_pc(32'h00000018);

        // Reset again
        reset = 1;

        @(posedge clk);
        check_pc(32'h00000000);

        $display("");
        $display("ALL PROGRAM COUNTER TESTS PASSED");
        $display("");

        $finish;

    end

endmodule