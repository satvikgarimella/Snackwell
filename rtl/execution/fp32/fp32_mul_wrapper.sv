// Binary32 multiply with round-to-nearest, ties-to-even and gradual underflow.
// The original start/busy/done interface is retained. Finite nonzero operands
// finish four clocks after acceptance; special cases finish on acceptance.
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
    typedef enum logic [2:0] {IDLE, MULTIPLY, NORMALIZE, DENORMALIZE, ROUND_PACK} state_t;
    state_t state;
    logic sign;
    logic [23:0] significand_a, significand_b;
    logic [47:0] product;
    logic signed [9:0] exponent;
    wire [5:0] normalize_shift = leading_zeros(product);
    wire [24:0] rounded = {1'b0, product[47:24]} +
        {24'd0, (product[23] && ((|product[22:0]) || product[24]))};
    wire signed [9:0] packed_exponent = exponent + 10'sd127;

    assign busy = state != IDLE;

    function automatic signed [9:0] unbiased_exponent(input logic [7:0] field);
        unbiased_exponent = field == 0 ? -10'sd126 : $signed({2'b00, field}) - 10'sd127;
    endfunction

    function automatic [5:0] leading_zeros(input logic [47:0] value);
        logic [63:0] shifted;
        logic [5:0] count;
        begin
            shifted = {value, 16'd0};
            count = 0;
            if (shifted[63:32] == 0) begin count = count + 32; shifted = shifted << 32; end
            if (shifted[63:48] == 0) begin count = count + 16; shifted = shifted << 16; end
            if (shifted[63:56] == 0) begin count = count + 8; shifted = shifted << 8; end
            if (shifted[63:60] == 0) begin count = count + 4; shifted = shifted << 4; end
            if (shifted[63:62] == 0) begin count = count + 2; shifted = shifted << 2; end
            if (!shifted[63]) count = count + 1;
            leading_zeros = value == 0 ? 6'd48 : count;
        end
    endfunction

    function automatic [47:0] shift_right_jam(input logic [47:0] value,
                                              input logic [9:0] amount);
        logic [47:0] shifted;
        begin
            if (amount == 0) shift_right_jam = value;
            else if (amount >= 48) shift_right_jam = {47'd0, |value};
            else begin
                shifted = value >> amount;
                shifted[0] = shifted[0] | (|(value << (10'd48 - amount)));
                shift_right_jam = shifted;
            end
        end
    endfunction

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            result <= 32'd0;
            sign <= 1'b0;
            significand_a <= 24'd0;
            significand_b <= 24'd0;
            product <= 48'd0;
            exponent <= 10'sd0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    if ((operand_a[30:23] == 8'hff && operand_a[22:0] != 0) ||
                        (operand_b[30:23] == 8'hff && operand_b[22:0] != 0) ||
                        (operand_a[30:0] == 31'h7f800000 && operand_b[30:0] == 0) ||
                        (operand_b[30:0] == 31'h7f800000 && operand_a[30:0] == 0)) begin
                        // Preserve the old multiplier's canonical NaN encoding.
                        result <= 32'hffc00000;
                        done <= 1'b1;
                    end else if (operand_a[30:23] == 8'hff || operand_b[30:23] == 8'hff) begin
                        result <= {operand_a[31] ^ operand_b[31], 8'hff, 23'd0};
                        done <= 1'b1;
                    end else if (operand_a[30:0] == 0 || operand_b[30:0] == 0) begin
                        result <= {operand_a[31] ^ operand_b[31], 31'd0};
                        done <= 1'b1;
                    end else begin
                        sign <= operand_a[31] ^ operand_b[31];
                        significand_a <= {operand_a[30:23] != 0, operand_a[22:0]};
                        significand_b <= {operand_b[30:23] != 0, operand_b[22:0]};
                        exponent <= unbiased_exponent(operand_a[30:23]) +
                                    unbiased_exponent(operand_b[30:23]) + 10'sd1;
                        state <= MULTIPLY;
                    end
                end
                MULTIPLY: begin
                    product <= significand_a * significand_b;
                    state <= NORMALIZE;
                end
                NORMALIZE: begin
                    product <= product << normalize_shift;
                    exponent <= exponent - $signed({4'd0, normalize_shift});
                    state <= DENORMALIZE;
                end
                DENORMALIZE: begin
                    if (exponent < -10'sd126) begin
                        product <= shift_right_jam(product, $unsigned(-10'sd126 - exponent));
                        exponent <= -10'sd126;
                    end
                    state <= ROUND_PACK;
                end
                ROUND_PACK: begin
                    if (exponent > 10'sd127 || (rounded[24] && exponent == 10'sd127))
                        result <= {sign, 8'hff, 23'd0};
                    else if (rounded[24])
                        result <= {sign, (packed_exponent[7:0] + 8'd1), rounded[23:1]};
                    else if (!rounded[23])
                        result <= {sign, 8'd0, rounded[22:0]};
                    else
                        result <= {sign, packed_exponent[7:0], rounded[22:0]};
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
