// Output-stationary PE. Each accepted step retains its A/B tokens until advance.
// FP32 arithmetic is round32(accumulator + round32(A * B)), not fused FMA.
// NVFP4 tokens carry an E2M1 nibble and UE4M3 scale; their product is exact.
// Tokens may advance once the product reaches the adder. The next multiply
// can overlap that addition; arithmetic_busy also covers the final sum update.
module fp32_mac_pe #(
    parameter bit INITIAL_ACCUMULATOR = 0
) (
    input  logic        clk,
    input  logic        reset,
    input  logic        clear,
    input  logic        nvfp4_mode,
    input  logic [31:0] initial_accumulator,
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
    logic saved_a_valid, saved_b_valid, saved_nvfp4;
    logic [31:0] saved_narrow_product;
    wire accept = step_valid && step_ready;
    wire mul_done, add_done, add_busy;
    wire [31:0] product, wide_product, narrow_product, sum;
    wire add_start = ((state == MULTIPLY && mul_done) || state == WAIT_ADD) &&
        !add_busy && !reset && !clear;

    assign step_ready = state == IDLE && !reset && !clear;
    assign step_done = state == COMPLETE && !reset && !clear;
    assign arithmetic_busy = state == MULTIPLY || state == WAIT_ADD || add_busy || add_done;
    assign product = saved_nvfp4 ? saved_narrow_product : wide_product;

    nvfp4_product narrow_multiplier (
        .a(nvfp4_mode ? a_data[3:0] : 4'd0), .b(nvfp4_mode ? b_data[3:0] : 4'd0),
        .scale_a(nvfp4_mode ? a_data[11:4] : 8'd0),
        .scale_b(nvfp4_mode ? b_data[11:4] : 8'd0), .product(narrow_product)
    );

    fp32_mul_wrapper multiplier (
        .clk(clk), .rst(reset || clear),
        .start(accept && a_valid && b_valid && !nvfp4_mode),
        .operand_a(a_data), .operand_b(b_data),
        .busy(), .done(mul_done), .result(wide_product)
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
            saved_nvfp4 <= 1'b0;
            saved_narrow_product <= 32'd0;
            a_out <= 32'd0;
            b_out <= 32'd0;
            a_out_valid <= 1'b0;
            b_out_valid <= 1'b0;
            accumulator <= (!reset && INITIAL_ACCUMULATOR) ? initial_accumulator : 32'd0;
        end else begin
            if (add_done) accumulator <= sum;
            case (state)
                IDLE: if (accept) begin
                    saved_a <= a_data;
                    saved_b <= b_data;
                    saved_a_valid <= a_valid;
                    saved_b_valid <= b_valid;
                    saved_nvfp4 <= nvfp4_mode;
                    saved_narrow_product <= narrow_product;
                    // Forward each stream independently even when this PE
                    // has no matching pair. Zero is valid numerical data.
                    if (a_valid && b_valid) begin
                        if (nvfp4_mode) state <= WAIT_ADD;
                        else state <= MULTIPLY;
                    end
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
