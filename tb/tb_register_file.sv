`timescale 1ns/1ps

module tb_register_file;

    logic clk;
    logic reset;

    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;

    logic [31:0] rs1_data;
    logic [31:0] rs2_data;

    logic write_enable;
    logic [3:0] rd_addr;
    logic [31:0] write_data;

    register_file dut (
        .clk(clk),
        .reset(reset),
        .rs1_addr(rs1_addr),
        .rs2_addr(rs2_addr),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data),
        .write_enable(write_enable),
        .rd_addr(rd_addr),
        .write_data(write_data)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_register_file.vcd");
        $dumpvars(0, tb_register_file);
    end

    task write_register(
        input logic [3:0] addr,
        input logic [31:0] data
    );
        begin
            rd_addr      = addr;
            write_data   = data;
            write_enable = 1'b1;

            @(posedge clk);
            #1;

            write_enable = 1'b0;
        end
    endtask

    task check_register(
        input logic [3:0] addr,
        input logic [31:0] expected
    );
        begin
            rs1_addr = addr;
            #1;

            if (rs1_data !== expected) begin
                $display(
                    "FAIL: R%0d = %h expected %h",
                    addr,
                    rs1_data,
                    expected
                );
                $fatal;
            end
            else begin
                $display(
                    "PASS: R%0d = %h",
                    addr,
                    rs1_data
                );
            end
        end
    endtask

    initial begin

        clk = 0;
        reset = 1;

        rs1_addr = 0;
        rs2_addr = 0;

        write_enable = 0;
        rd_addr = 0;
        write_data = 0;

        $display("");
        $display("=== Snackwell Register File Test ===");
        $display("");

        // Reset
        @(posedge clk);
        #1;

        reset = 0;

        check_register(4'd0, 32'd0);
        check_register(4'd3, 32'd0);
        check_register(4'd15, 32'd0);

        // Write R3
        write_register(4'd3, 32'd42);

        check_register(4'd3, 32'd42);

        // Write R7
        write_register(4'd7, 32'd100);

        check_register(4'd7, 32'd100);

        // Verify R3 was not changed
        check_register(4'd3, 32'd42);

        // Test simultaneous reads
        rs1_addr = 4'd3;
        rs2_addr = 4'd7;

        #1;

        if (
            rs1_data !== 32'd42 ||
            rs2_data !== 32'd100
        ) begin

            $display(
                "FAIL simultaneous read: rs1=%0d rs2=%0d",
                rs1_data,
                rs2_data
            );

            $fatal;
        end

        $display(
            "PASS simultaneous read: R3=%0d R7=%0d",
            rs1_data,
            rs2_data
        );

        $display("");
        $display("ALL REGISTER FILE TESTS PASSED");
        $display("");

        $finish;
    end

endmodule