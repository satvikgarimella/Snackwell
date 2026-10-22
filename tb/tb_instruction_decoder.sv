`timescale 1ns/1ps

module tb_instruction_decoder;

    import snackwell_pkg::*;

    logic [31:0] instruction;

    logic [5:0]  opcode;

    logic [3:0]  rd_addr;
    logic [3:0]  rs1_addr;
    logic [3:0]  rs2_addr;

    logic [13:0] immediate;
    logic [31:0] immediate_extended;

    instruction_decoder dut (

        .instruction        (instruction),

        .opcode             (opcode),

        .rd_addr            (rd_addr),
        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),

        .immediate          (immediate),
        .immediate_extended (immediate_extended)

    );

    initial begin

        $dumpfile("tb_instruction_decoder.vcd");
        $dumpvars(0, tb_instruction_decoder);

    end

    initial begin

        $display("");
        $display("=== Snackwell Instruction Decoder Test ===");
        $display("");

        // TEST 1
        //
        // ADD R3, R1, R2

        instruction = {
            OP_ADD,     // opcode
            4'd3,       // rd
            4'd1,       // rs1
            4'd2,       // rs2
            14'd0       // unused immediate
        };

        #10;

        if (opcode !== OP_ADD) begin
            $display("FAIL: ADD opcode");
            $fatal;
        end

        if (rd_addr !== 4'd3) begin
            $display("FAIL: ADD rd");
            $fatal;
        end

        if (rs1_addr !== 4'd1) begin
            $display("FAIL: ADD rs1");
            $fatal;
        end

        if (rs2_addr !== 4'd2) begin
            $display("FAIL: ADD rs2");
            $fatal;
        end

        $display(
            "PASS: ADD R%0d, R%0d, R%0d",
            rd_addr,
            rs1_addr,
            rs2_addr
        );

        // TEST 2
        //
        // SUB R7, R5, R4

        instruction = {
            OP_SUB,
            4'd7,
            4'd5,
            4'd4,
            14'd0
        };

        #10;

        if (
            opcode   !== OP_SUB ||
            rd_addr  !== 4'd7   ||
            rs1_addr !== 4'd5   ||
            rs2_addr !== 4'd4
        ) begin

            $display("FAIL: SUB decode");
            $fatal;

        end

        $display(
            "PASS: SUB R%0d, R%0d, R%0d",
            rd_addr,
            rs1_addr,
            rs2_addr
        );

        // TEST 3
        //
        // ADDI R6, R2, 10

        instruction = {
            OP_ADDI,
            4'd6,
            4'd2,
            4'd0,       // unused rs2
            14'd10
        };

        #10;

        if (
            opcode             !== OP_ADDI ||
            rd_addr            !== 4'd6    ||
            rs1_addr           !== 4'd2    ||
            immediate_extended !== 32'd10
        ) begin

            $display("FAIL: ADDI decode");
            $fatal;

        end

        $display(
            "PASS: ADDI R%0d, R%0d, %0d",
            rd_addr,
            rs1_addr,
            immediate_extended
        );

        // TEST 4
        //
        // Negative immediate = -1

        instruction = {
            OP_ADDI,
            4'd3,
            4'd1,
            4'd0,
            14'h3FFF
        };

        #10;

        if (immediate_extended !== 32'hFFFFFFFF) begin

            $display(
                "FAIL: sign extension = %h",
                immediate_extended
            );

            $fatal;

        end

        $display(
            "PASS: negative immediate sign extension = %h",
            immediate_extended
        );

        $display("");
        $display("ALL INSTRUCTION DECODER TESTS PASSED");
        $display("");

        $finish;

    end

endmodule
