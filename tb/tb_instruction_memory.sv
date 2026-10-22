`timescale 1ns/1ps

module tb_instruction_memory;

    import snackwell_pkg::*;

    logic [31:0] pc;
    logic [31:0] instruction;

    instruction_memory dut (
        .clk(1'b0), .host_write_enable(1'b0), .host_word_address(10'd0),
        .host_write_data(32'd0), .host_read_data(),
        .pc          (pc),
        .instruction (instruction)
    );

    initial begin
        $dumpfile("tb_instruction_memory.vcd");
        $dumpvars(0, tb_instruction_memory);
    end

    task check_instruction(
        input logic [31:0] test_pc,
        input logic [31:0] expected
    );

        begin

            pc = test_pc;

            #10;

            if (instruction !== expected) begin

                $display(
                    "FAIL: PC=%h instruction=%h expected=%h",
                    pc,
                    instruction,
                    expected
                );

                $fatal;

            end

            else begin

                $display(
                    "PASS: PC=%h instruction=%h",
                    pc,
                    instruction
                );

            end

        end

    endtask

    initial begin

        $display("");
        $display("=== Snackwell Instruction Memory Test ===");
        $display("");

        // ADD R3, R1, R2
        check_instruction(
            32'h00000000,
            {
                OP_ADD,
                4'd3,
                4'd1,
                4'd2,
                14'd0
            }
        );

        // SUB R4, R1, R2
        check_instruction(
            32'h00000004,
            {
                OP_SUB,
                4'd4,
                4'd1,
                4'd2,
                14'd0
            }
        );

        // XOR R5, R1, R2
        check_instruction(
            32'h00000008,
            {
                OP_XOR,
                4'd5,
                4'd1,
                4'd2,
                14'd0
            }
        );

        // EXIT
        check_instruction(
            32'h0000000C,
            {
                OP_EXIT,
                4'd0,
                4'd0,
                4'd0,
                14'd0
            }
        );

        // NOP region
        check_instruction(
            32'h00000010,
            32'd0
        );

        $display("");
        // The full PC must address the expanded memory without wrapping.
        dut.memory[16] = {OP_TID, 4'd7, 22'd0};
        check_instruction(32'd64, {OP_TID, 4'd7, 22'd0});
        dut.memory[1023] = {OP_ADDI, 4'd2, 4'd1, 4'd0, 14'd7};
        check_instruction(32'd4092, {OP_ADDI, 4'd2, 4'd1, 4'd0, 14'd7});
        check_instruction(32'd4096, {OP_EXIT, 26'd0});
        check_instruction(32'hfffffffc, {OP_EXIT, 26'd0});
        check_instruction(32'd2, {OP_EXIT, 26'd0});
        $display("ALL INSTRUCTION MEMORY TESTS PASSED");
        $display("");

        $finish;

    end

endmodule
