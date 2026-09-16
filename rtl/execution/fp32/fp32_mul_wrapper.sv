module fp32_mul_wrapper (
    input  logic        clk,
    input  logic        rst,

    input  logic        start,
    input  logic [31:0] operand_a,
    input  logic [31:0] operand_b,

    output logic        busy,
    output logic        done,
    output logic [31:0] result
);

    logic [31:0] saved_a;
    logic [31:0] saved_b;

    wire input_a_stb;
    wire input_b_stb;

    wire input_a_ack;
    wire input_b_ack;

    wire [31:0] output_z;
    wire        output_z_stb;
    wire        output_z_ack;


    typedef enum logic [2:0] {
        IDLE,
        SEND_A,
        SEND_B,
        WAIT_RESULT,
        ACK_RESULT
    } state_t;

    state_t state;


    // =====================================================
    // HANDSHAKE SIGNALS
    //
    // The strobe stays HIGH for the entire state.
    // This avoids timing/race problems with registered stb.
    // =====================================================

    assign input_a_stb =
        (state == SEND_A);

    assign input_b_stb =
        (state == SEND_B);

    assign output_z_ack =
        (state == ACK_RESULT);

    assign busy =
        (state != IDLE);


    // =====================================================
    // EXTERNAL FP32 MULTIPLIER
    // =====================================================

    multiplier u_multiplier (
        .input_a      (saved_a),
        .input_b      (saved_b),

        .input_a_stb  (input_a_stb),
        .input_b_stb  (input_b_stb),

        .output_z_ack (output_z_ack),

        .clk          (clk),
        .rst          (rst),

        .output_z     (output_z),
        .output_z_stb (output_z_stb),

        .input_a_ack  (input_a_ack),
        .input_b_ack  (input_b_ack)
    );


    // =====================================================
    // WRAPPER CONTROL
    // =====================================================

    always_ff @(posedge clk) begin

        if (rst) begin

            state   <= IDLE;

            saved_a <= 32'd0;
            saved_b <= 32'd0;

            result  <= 32'd0;
            done    <= 1'b0;

        end

        else begin

            // done is a one-cycle pulse
            done <= 1'b0;


            case (state)

                // =========================================
                // WAIT FOR AN FMUL REQUEST
                // =========================================

                IDLE: begin

                    if (start) begin

                        saved_a <= operand_a;
                        saved_b <= operand_b;

                        state <= SEND_A;

                    end

                end


                // =========================================
                // SEND OPERAND A
                //
                // input_a_stb is automatically HIGH
                // for this entire state.
                // =========================================

                SEND_A: begin

                    if (input_a_ack) begin

                        state <= SEND_B;

                    end

                end


                // =========================================
                // SEND OPERAND B
                //
                // input_b_stb stays HIGH until ACK.
                // =========================================

                SEND_B: begin

                    if (input_b_ack) begin

                        state <= WAIT_RESULT;

                    end

                end


                // =========================================
                // WAIT FOR FP MULTIPLICATION
                // =========================================

                WAIT_RESULT: begin

                    if (output_z_stb) begin

                        result <= output_z;

                        state <= ACK_RESULT;

                    end

                end


                // =========================================
                // ACKNOWLEDGE RESULT
                //
                // output_z_ack is HIGH for this state.
                // =========================================

                ACK_RESULT: begin

                    done <= 1'b1;

                    state <= IDLE;

                end


                default: begin

                    state <= IDLE;

                end

            endcase

        end

    end

endmodule