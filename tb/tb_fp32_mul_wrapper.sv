`timescale 1ns/1ps

module tb_fp32_mul_wrapper;

    logic clk;
    logic rst;

    logic        start;
    logic [31:0] operand_a;
    logic [31:0] operand_b;

    logic        busy;
    logic        done;
    logic [31:0] result;


    fp32_mul_wrapper dut (
        .clk       (clk),
        .rst       (rst),

        .start     (start),
        .operand_a (operand_a),
        .operand_b (operand_b),

        .busy      (busy),
        .done      (done),
        .result    (result)
    );


    // 10 ns clock period
    always #5 clk = ~clk;


    initial begin

        $dumpfile("tb_fp32_mul_wrapper.vcd");
        $dumpvars(0, tb_fp32_mul_wrapper);

        clk       = 1'b0;
        rst       = 1'b1;

        start     = 1'b0;
        operand_a = 32'd0;
        operand_b = 32'd0;


        // =============================================
        // RESET
        // =============================================

        repeat (3) @(posedge clk);

        // Change reset AWAY from sampling edge
        @(negedge clk);
        rst = 1'b0;


        // Give hardware one clean clock after reset
        @(posedge clk);


        // =============================================
        // TEST:
        //
        // 2.5 × 4.0 = 10.0
        //
        // 2.5  = 0x40200000
        // 4.0  = 0x40800000
        // 10.0 = 0x41200000
        // =============================================

        @(negedge clk);

        operand_a = 32'h40200000;
        operand_b = 32'h40800000;
        start     = 1'b1;


        // Wrapper sees start HIGH on this posedge
        @(posedge clk);


        // Drop start safely after it was sampled
        @(negedge clk);
        start = 1'b0;


        // =============================================
        // WAIT FOR RESULT
        // =============================================

        fork

            begin : wait_for_result

                wait(done === 1'b1);

                $display(
                    "FP32 multiplication finished after time %0t",
                    $time
                );

                if (result !== 32'h41200000) begin

                    $display(
                        "FAIL: result=%h expected=41200000",
                        result
                    );

                    $fatal;

                end


                $display(
                    "PASS: 2.5 * 4.0 = 10.0"
                );

                $display(
                    "ALL FP32 MUL WRAPPER TESTS PASSED"
                );

                #10;
                $finish;

            end


            begin : timeout

                repeat (200) @(posedge clk);

                $display(
                    "FAIL: timeout waiting for FP32 multiplier"
                );

                $display(
                    "wrapper state = %0d",
                    dut.state
                );

                $display(
                    "busy = %b",
                    busy
                );

                $display(
                    "A: stb=%b ack=%b",
                    dut.input_a_stb,
                    dut.input_a_ack
                );

                $display(
                    "B: stb=%b ack=%b",
                    dut.input_b_stb,
                    dut.input_b_ack
                );

                $display(
                    "Z: stb=%b ack=%b",
                    dut.output_z_stb,
                    dut.output_z_ack
                );

                $fatal;

            end

        join_any

        disable fork;

    end

endmodule