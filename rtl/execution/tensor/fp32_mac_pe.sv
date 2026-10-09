// Output-stationary PE. Each accepted step retains its A/B tokens until advance.
// Arithmetic is round32(accumulator + round32(A * B)), not fused FMA.
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
    output logic [31:0] accumulator
);
    typedef enum logic [2:0] {IDLE, MULTIPLY, ADD_START, ADD_WAIT, COMPLETE} state_t;
    state_t state;
    logic [31:0] saved_a, saved_b;
    logic saved_a_valid, saved_b_valid;
    wire accept = step_valid && step_ready;
    wire mul_done, add_done;
    wire [31:0] product, sum;

    assign step_ready = state == IDLE && !reset && !clear;
    assign step_done = state == COMPLETE && !reset && !clear;

    fp32_mul_wrapper multiplier (
        .clk(clk), .rst(reset || clear),
        .start(accept && a_valid && b_valid),
        .operand_a(a_data), .operand_b(b_data),
        .busy(), .done(mul_done), .result(product)
    );
    fp32_addsub adder (
        .clk(clk), .rst(reset || clear),
        .start(state == ADD_START && !reset && !clear), .subtract(1'b0),
        .operand_a(accumulator), .operand_b(product),
        .busy(), .done(add_done), .result(sum)
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
                MULTIPLY: if (mul_done) state <= ADD_START;
                ADD_START: state <= ADD_WAIT;
                ADD_WAIT: if (add_done) begin
                    accumulator <= sum;
                    state <= COMPLETE;
                end
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
