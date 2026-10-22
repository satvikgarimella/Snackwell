// Binary32 add/subtract, round-to-nearest, ties-to-even.
// Alignment and normalization each take one cycle, independent of exponent gaps.
// Subnormals are preserved. NaNs are canonicalized; exception flags and
// selectable rounding modes are not implemented. Requests while busy are ignored.
module fp32_addsub (
    input  logic        clk,
    input  logic        rst,
    input  logic        start,
    input  logic        subtract,
    input  logic [31:0] operand_a,
    input  logic [31:0] operand_b,
    output logic        busy,
    output logic        done,
    output logic [31:0] result
);
    typedef enum logic [2:0] {IDLE, ALIGN, COMBINE, NORMALIZE, ROUND_PACK} state_t;
    state_t state;
    logic sign_large, sign_small;
    logic [8:0] exponent_large, exponent_small;
    // 24 significand bits followed by guard, round, and sticky bits.
    logic [26:0] mantissa_large, mantissa_small, mantissa_result;
    logic [27:0] sum;
    logic [24:0] rounded;
    logic [31:0] effective_b;
    logic [8:0] exponent_gap;
    logic [4:0] normalize_shift;

    function automatic [26:0] shift_right_jam(input logic [26:0] value,
                                              input logic [8:0] amount);
        logic [26:0] shifted;
        begin
            if (amount == 0) shift_right_jam = value;
            else if (amount >= 27) shift_right_jam = {26'd0, |value};
            else begin
                shifted = value >> amount;
                // Keep every discarded bit in the sticky bit for rounding.
                shifted[0] = shifted[0] | (|(value << (9'd27 - amount)));
                shift_right_jam = shifted;
            end
        end
    endfunction

    function automatic [4:0] leading_zeros(input logic [26:0] value);
        logic [31:0] shifted;
        logic [4:0] count;
        begin
            shifted = {value, 5'd0};
            count = 0;
            if (shifted[31:16] == 0) begin count = count + 16; shifted = shifted << 16; end
            if (shifted[31:24] == 0) begin count = count + 8; shifted = shifted << 8; end
            if (shifted[31:28] == 0) begin count = count + 4; shifted = shifted << 4; end
            if (shifted[31:30] == 0) begin count = count + 2; shifted = shifted << 2; end
            if (!shifted[31]) count = count + 1;
            leading_zeros = value == 0 ? 5'd27 : count;
        end
    endfunction

    assign busy = state != IDLE;
    assign effective_b = {operand_b[31] ^ subtract, operand_b[30:0]};
    assign exponent_gap = exponent_large - exponent_small;
    // Stop at exponent 1: values below the normal range retain subnormal bits.
    assign normalize_shift = {4'd0, leading_zeros(mantissa_result)} < exponent_large
        ? leading_zeros(mantissa_result) : 5'(exponent_large - 9'd1);
    assign sum = {1'b0, mantissa_large} + {1'b0, mantissa_small};
    assign rounded = {1'b0, mantissa_result[26:3]} +
        {24'd0, (mantissa_result[2] &&
                 (mantissa_result[1] || mantissa_result[0] || mantissa_result[3]))};

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            result <= 32'd0;
            sign_large <= 1'b0;
            sign_small <= 1'b0;
            exponent_large <= 9'd0;
            exponent_small <= 9'd0;
            mantissa_large <= 27'd0;
            mantissa_small <= 27'd0;
            mantissa_result <= 27'd0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    if ((operand_a[30:23] == 8'hff && operand_a[22:0] != 0) ||
                        (effective_b[30:23] == 8'hff && effective_b[22:0] != 0)) begin
                        result <= 32'h7fc00000;
                        done <= 1'b1;
                    end else if (operand_a[30:23] == 8'hff &&
                                 effective_b[30:23] == 8'hff &&
                                 operand_a[31] != effective_b[31]) begin
                        result <= 32'h7fc00000;
                        done <= 1'b1;
                    end else if (operand_a[30:23] == 8'hff) begin
                        result <= operand_a;
                        done <= 1'b1;
                    end else if (effective_b[30:23] == 8'hff) begin
                        result <= effective_b;
                        done <= 1'b1;
                    end else if (operand_a[30:0] == 0 && effective_b[30:0] == 0) begin
                        result <= {operand_a[31] & effective_b[31], 31'd0};
                        done <= 1'b1;
                    end else if (operand_a[30:0] == 0) begin
                        result <= effective_b;
                        done <= 1'b1;
                    end else if (effective_b[30:0] == 0) begin
                        result <= operand_a;
                        done <= 1'b1;
                    end else begin
                        // Order by magnitude so subtraction is nonnegative.
                        if (operand_a[30:0] >= effective_b[30:0]) begin
                            sign_large <= operand_a[31];
                            sign_small <= effective_b[31];
                            exponent_large <= operand_a[30:23] == 0 ? 9'd1 : {1'b0, operand_a[30:23]};
                            exponent_small <= effective_b[30:23] == 0 ? 9'd1 : {1'b0, effective_b[30:23]};
                            mantissa_large <= {operand_a[30:23] != 0, operand_a[22:0], 3'b0};
                            mantissa_small <= {effective_b[30:23] != 0, effective_b[22:0], 3'b0};
                        end else begin
                            sign_large <= effective_b[31];
                            sign_small <= operand_a[31];
                            exponent_large <= effective_b[30:23] == 0 ? 9'd1 : {1'b0, effective_b[30:23]};
                            exponent_small <= operand_a[30:23] == 0 ? 9'd1 : {1'b0, operand_a[30:23]};
                            mantissa_large <= {effective_b[30:23] != 0, effective_b[22:0], 3'b0};
                            mantissa_small <= {operand_a[30:23] != 0, operand_a[22:0], 3'b0};
                        end
                        state <= ALIGN;
                    end
                end
                ALIGN: begin
                    mantissa_small <= shift_right_jam(mantissa_small, exponent_gap);
                    state <= COMBINE;
                end
                COMBINE: begin
                    if (sign_large == sign_small) begin
                        if (sum[27]) begin
                            mantissa_result <= {sum[27:2], |sum[1:0]};
                            exponent_large <= exponent_large + 9'd1;
                        end else mantissa_result <= sum[26:0];
                    end else mantissa_result <= mantissa_large - mantissa_small;
                    state <= NORMALIZE;
                end
                NORMALIZE: begin
                    if (mantissa_result == 0) begin
                        result <= 32'd0; // exact cancellation in nearest-even mode
                        done <= 1'b1;
                        state <= IDLE;
                    end else begin
                        mantissa_result <= mantissa_result << normalize_shift;
                        exponent_large <= exponent_large - {4'd0, normalize_shift};
                        state <= ROUND_PACK;
                    end
                end
                ROUND_PACK: begin
                    if (exponent_large >= 9'd255 ||
                        (rounded[24] && exponent_large == 9'd254))
                        result <= {sign_large, 8'hff, 23'd0};
                    else if (rounded[24])
                        result <= {sign_large, (exponent_large[7:0] + 8'd1), rounded[23:1]};
                    else if (exponent_large == 9'd1 && !rounded[23])
                        result <= {sign_large, 8'd0, rounded[22:0]};
                    else
                        result <= {sign_large, exponent_large[7:0], rounded[22:0]};
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
