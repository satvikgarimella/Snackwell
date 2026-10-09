`timescale 1ns/1ps

module tb_integer_alu;

    import snackwell_pkg::*;

    logic [5:0]  opcode;
    logic [31:0] operand_a;
    logic [31:0] operand_b;
    logic [31:0] result;

    integer_alu dut (
        .opcode    (opcode),
        .operand_a (operand_a),
        .operand_b (operand_b),
        .result    (result)
    );

    initial begin
        $dumpfile("tb_integer_alu.vcd");
        $dumpvars(0, tb_integer_alu);
    end

    task check_result(
        input logic [5:0]  test_opcode,
        input logic [31:0] test_a,
        input logic [31:0] test_b,
        input logic [31:0] expected
    );

        begin
            opcode    = test_opcode;
            operand_a = test_a;
            operand_b = test_b;

            #10;

            if (result !== expected) begin
                $display(
                    "FAIL: opcode=%h A=%h B=%h result=%h expected=%h",
                    opcode,
                    operand_a,
                    operand_b,
                    result,
                    expected
                );
                $fatal;
            end
            else begin
                $display(
                    "PASS: opcode=%h A=%h B=%h result=%h",
                    opcode,
                    operand_a,
                    operand_b,
                    result
                );
            end
        end

    endtask

    initial begin

        $display("");
        $display("=== Snackwell Integer ALU Test ===");
        $display("");

        check_result(OP_ADD, 32'd7,  32'd8, 32'd15);
        check_result(OP_SUB, 32'd20, 32'd5, 32'd15);
        check_result(OP_ADDI, 32'd10, 32'hffffffff, 32'd9);
        check_result(OP_ADDI, 32'hffffffff, 32'd1, 32'd0);
        check_result(OP_SHR, 32'h80000000, 32'd31, 32'd1);
        check_result(OP_SHR, 32'h80000000, 32'd32, 32'h80000000);
        check_result(OP_SLTU, 32'hffffffff, 32'd0, 32'd0);
        check_result(OP_SLTU, 32'd0, 32'hffffffff, 32'd1);
        check_result(OP_FLT, 32'hbf800000, 32'h3f800000, 32'd1);
        check_result(OP_FLT, 32'hc0000000, 32'hbf800000, 32'd1);
        check_result(OP_FLT, 32'h80000000, 32'd0, 32'd0);
        check_result(OP_FLT, 32'h7f800001, 32'h7f800000, 32'd0);
        check_result(OP_FLT, 32'hff800000, 32'h7f800000, 32'd1);

        check_result(
            OP_AND,
            32'h0000F0F0,
            32'h00000FF0,
            32'h000000F0
        );

        check_result(
            OP_OR,
            32'h0000F000,
            32'h00000F00,
            32'h0000FF00
        );

        check_result(
            OP_XOR,
            32'h0000FFFF,
            32'h00000F0F,
            32'h0000F0F0
        );

        check_result(
            OP_SUB,
            32'd7,
            32'd8,
            32'hFFFFFFFF
        );

        $display("");
        $display("ALL INTEGER ALU TESTS PASSED");
        $display("");

        $finish;
    end

endmodule
