// Output-stationary PE. Each accepted step retains its A/B tokens until advance.
// Arithmetic is round32(accumulator + round32(A * B)), not fused FMA.
// Tokens may advance once the product reaches the adder. The next multiply
// can overlap that addition; arithmetic_busy also covers the final sum update.
module fp32_mac_pe (
    input  logic        clk,
    input  logic        reset,
    input  logic        clear,
    input  logic        step_valid,
    output logic        step_ready,
    input  logic        a_valid,
    input  logic        b_valid,
    input  logic [31:0] a_data,
    input  logic [31:0] b_data,
    input  logic        advance,
    output logic        step_done,
    output logic        a_out_valid,
    output logic        b_out_valid,
    output logic [31:0] a_out,
    output logic [31:0] b_out,
    output logic [31:0] accumulator,
    output logic        arithmetic_busy
);
    typedef enum logic [1:0] {IDLE, MULTIPLY, WAIT_ADD, COMPLETE} state_t;
    state_t state;
    logic [31:0] saved_a, saved_b;
    logic saved_a_valid, saved_b_valid;
    wire accept = step_valid && step_ready;
    wire mul_done, add_done, add_busy;
    wire [31:0] product, sum;
    wire add_start = ((state == MULTIPLY && mul_done) || state == WAIT_ADD) &&
        !add_busy && !reset && !clear;

    assign step_ready = state == IDLE && !reset && !clear;
    assign step_done = state == COMPLETE && !reset && !clear;
    assign arithmetic_busy = state == MULTIPLY || state == WAIT_ADD || add_busy || add_done;

    fp32_mul_wrapper multiplier (
        .clk(clk), .rst(reset || clear),
        .start(accept && a_valid && b_valid),
        .operand_a(a_data), .operand_b(b_data),
        .busy(), .done(mul_done), .result(product)
    );
    fp32_addsub adder (
        .clk(clk), .rst(reset || clear),
        .start(add_start), .subtract(1'b0),
        // Forward the preceding sum when its completion and the next add
        // coincide. This preserves k order even with short special-case paths.
        .operand_a(add_done ? sum : accumulator), .operand_b(product),
        .busy(add_busy), .done(add_done), .result(sum)
    );

    always_ff @(posedge clk) begin
        if (reset || clear) begin
            state <= IDLE;
            saved_a <= 32'd0;
            saved_b <= 32'd0;
            saved_a_valid <= 1'b0;
            saved_b_valid <= 1'b0;
            a_out <= 32'd0;
            b_out <= 32'd0;
            a_out_valid <= 1'b0;
            b_out_valid <= 1'b0;
            accumulator <= 32'd0;
        end else begin
            if (add_done) accumulator <= sum;
            case (state)
                IDLE: if (accept) begin
                    saved_a <= a_data;
                    saved_b <= b_data;
                    saved_a_valid <= a_valid;
                    saved_b_valid <= b_valid;
                    // Forward each stream independently even when this PE
                    // has no matching pair. Zero is valid numerical data.
                    if (a_valid && b_valid) state <= MULTIPLY;
                    else state <= COMPLETE;
                end
                MULTIPLY: if (mul_done) begin
                    if (add_start) state <= COMPLETE;
                    else state <= WAIT_ADD;
                end
                WAIT_ADD: if (add_start) state <= COMPLETE;
                COMPLETE: if (advance) begin
                    a_out <= saved_a;
                    b_out <= saved_b;
                    a_out_valid <= saved_a_valid;
                    b_out_valid <= saved_b_valid;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
