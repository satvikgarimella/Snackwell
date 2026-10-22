`timescale 1ns/1ps

module tb_instruction_frontend;

    import snackwell_pkg::*;

    logic clk;
    logic reset;
    logic enable;

    logic [31:0] pc;
    logic [31:0] instruction;

    logic [5:0] opcode;

    logic [3:0] rd_addr;
    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;

    logic [13:0] immediate;
    logic [31:0] immediate_extended;

    instruction_frontend dut (
        .clk                (clk),
        .reset              (reset),
        .enable             (enable),

        .pc                 (pc),
        .instruction        (instruction),

        .opcode             (opcode),

        .rd_addr            (rd_addr),
        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),

        .immediate          (immediate),
        .immediate_extended (immediate_extended)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_instruction_frontend.vcd");
        $dumpvars(0, tb_instruction_frontend);
    end

    task check_decode(
        input logic [31:0] expected_pc,
        input logic [5:0]  expected_opcode,
        input logic [3:0]  expected_rd,
        input logic [3:0]  expected_rs1,
        input logic [3:0]  expected_rs2
    );

        begin

            #1;

            if (
                pc       !== expected_pc     ||
                opcode   !== expected_opcode ||
                rd_addr  !== expected_rd     ||
                rs1_addr !== expected_rs1    ||
                rs2_addr !== expected_rs2
            ) begin

                $display("");
                $display("FAIL");

                $display(
                    "PC=%h opcode=%h rd=%0d rs1=%0d rs2=%0d",
                    pc,
                    opcode,
                    rd_addr,
                    rs1_addr,
                    rs2_addr
                );

                $fatal;

            end

            else begin

                $display(
                    "PASS: PC=%h opcode=%h rd=R%0d rs1=R%0d rs2=R%0d",
                    pc,
                    opcode,
                    rd_addr,
                    rs1_addr,
                    rs2_addr
                );

            end

        end

    endtask

    initial begin

        clk    = 0;
        reset  = 1;
        enable = 0;

        $display("");
        $display("=== Snackwell Instruction Frontend Test ===");
        $display("");

        // RESET
        //
        // PC = 0
        // ADD R3,R1,R2

        @(posedge clk);
        #1;

        reset = 0;

        check_decode(
            32'h00000000,
            OP_ADD,
            4'd3,
            4'd1,
            4'd2
        );

        // NEXT INSTRUCTION
        //
        // PC = 4
        // SUB R4,R1,R2

        enable = 1;

        @(posedge clk);

        check_decode(
            32'h00000004,
            OP_SUB,
            4'd4,
            4'd1,
            4'd2
        );

        // NEXT
        //
        // PC = 8
        // XOR R5,R1,R2

        @(posedge clk);

        check_decode(
            32'h00000008,
            OP_XOR,
            4'd5,
            4'd1,
            4'd2
        );

        // EXIT
        //
        // PC = C

        @(posedge clk);

        check_decode(
            32'h0000000C,
            OP_EXIT,
            4'd0,
            4'd0,
            4'd0
        );
        // TEST STALL

        enable = 0;

        @(posedge clk);

        check_decode(
            32'h0000000C,
            OP_EXIT,
            4'd0,
            4'd0,
            4'd0
        );

        $display("");
        $display("PASS: frontend held instruction during stall");

        $display("");
        $display("ALL INSTRUCTION FRONTEND TESTS PASSED");
        $display("");

        $finish;

    end

endmodule
