`timescale 1ns/1ps
// Scalar contracts: numerical results, bounded latency, capture and reset.
module tb_fp32_arithmetic;
    import snackwell_pkg::*;
    logic clk = 0, rst = 1, start = 0;
    logic [5:0] opcode = OP_FMUL;
    logic [31:0] operand_a = 0, operand_b = 0;
    wire mul_busy, mul_done, add_busy, add_done;
    wire [31:0] product, sum;
    wire busy = opcode == OP_FMUL ? mul_busy : add_busy;
    wire done = opcode == OP_FMUL ? mul_done : add_done;
    wire [31:0] result = opcode == OP_FMUL ? product : sum;
    fp32_mul_wrapper mul (
        .clk(clk), .rst(rst), .start(start && opcode == snackwell_pkg::OP_FMUL),
        .operand_a(operand_a), .operand_b(operand_b),
        .busy(mul_busy), .done(mul_done), .result(product)
    );
    fp32_addsub add (
        .clk(clk), .rst(rst), .start(start && opcode != snackwell_pkg::OP_FMUL),
        .subtract(opcode == snackwell_pkg::OP_FSUB), .operand_a(operand_a), .operand_b(operand_b),
        .busy(add_busy), .done(add_done), .result(sum)
    );
    always #5 clk = ~clk;

    integer fd, parsed, count = 0;
    reg [31:0] op_word, mask_word;
    reg [31:0] a[0:3], b[0:3], expected[0:3];
    string vector_path;

    task automatic check_operation(input logic [5:0] op, input logic [31:0] av, bv, wanted);
        integer elapsed;
        logic [31:0] saved;
        begin
            @(negedge clk);
            opcode = op; operand_a = av; operand_b = bv; start = 1;
            @(negedge clk);
            start = 0;
            operand_a = 32'h7fc00001; operand_b = 32'h7fc00001;
            elapsed = 0;
            while (!done && elapsed < 5) begin
                // An extra request while busy must not replace the captured operands.
                start = busy && elapsed == 1;
                @(negedge clk);
                elapsed = elapsed + 1;
            end
            start = 0;
            if (!done || busy || elapsed > 4)
                $fatal(1, "Latency exceeded four clocks: op=%h a=%h b=%h clocks=%0d",
                    op, av, bv, elapsed);
            if (wanted[30:23] == 8'hff && wanted[22:0] != 0) begin
                if (result !== (op == OP_FMUL ? 32'hffc00000 : 32'h7fc00000))
                    $fatal(1, "Wrong canonical NaN: op=%h got=%h", op, result);
            end else if (result !== wanted)
                $fatal(1, "Scalar op=%h a=%h b=%h got=%h expected=%h",
                    op, av, bv, result, wanted);
            saved = result;
            @(negedge clk);
            if (busy || done || result !== saved)
                $fatal(1, "Result must persist and done must pulse once");
            count = count + 1;
        end
    endtask

    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path)) $fatal(1, "Missing vector file");
        fd = $fopen(vector_path, "r");
        if (!fd) $fatal(1, "Cannot open vector file");
        repeat (3) @(negedge clk);
        rst = 0;
        while (!$feof(fd)) begin
            parsed = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h %h %h %h %h\n",
                op_word, mask_word, a[0], b[0], expected[0], a[1], b[1], expected[1],
                a[2], b[2], expected[2], a[3], b[3], expected[3]);
            if (parsed != 14) $fatal(1, "Malformed vector");
            for (integer lane = 0; lane < 4; lane = lane + 1)
                check_operation(op_word[5:0], a[lane], b[lane], expected[lane]);
        end
        $fclose(fd);

        // Abort each in-flight phase, including the edge that would produce done.
        for (integer op = OP_FMUL; op <= OP_FSUB; op = op + 1) begin
            for (integer delay_cycles = 0; delay_cycles < 4; delay_cycles = delay_cycles + 1) begin
                @(negedge clk);
                opcode = 6'(op); operand_a = 32'h3fc00000; operand_b = 32'h3f800000; start = 1;
                @(negedge clk);
                start = 0;
                repeat (delay_cycles) @(negedge clk);
                rst = 1;
                @(negedge clk);
                rst = 0;
                repeat (6) begin
                    @(negedge clk);
                    if (mul_busy || mul_done || add_busy || add_done || product !== 0 || sum !== 0)
                        $fatal(1, "Stale arithmetic completion after reset");
                end
                check_operation(OP_FMUL, 32'h40000000, 32'h40400000, 32'h40c00000);
                check_operation(OP_FADD, 32'h40000000, 32'h40400000, 32'h40a00000);
            end
        end
        $display("PASS: %0d scalar FP results, four-clock bound, busy rejection and reset", count);
        $finish;
    end
    initial begin
        #100000000;
        $fatal(1, "Scalar FP test timeout");
    end
endmodule
