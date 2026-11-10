// Exact product of two E2M1 values with unsigned E4M3 block scales.
// Each scaled significand fits in eight bits. All nonzero finite products
// are normal binary32 numbers and need no rounding before FP32 accumulation.
module nvfp4_product (
    input  logic [3:0] a, b,
    input  logic [7:0] scale_a, scale_b,
    output logic [31:0] product
);
    logic [3:0] magnitude_a, magnitude_b, significand_a, significand_b;
    logic [7:0] scaled_a, scaled_b;
    logic [15:0] coefficient;
    logic [23:0] normalized;
    integer exponent_a, exponent_b, leading, exponent;

    // Twice the E2M1 magnitude, including its subnormal 0.5.
    function automatic logic [3:0] twice_magnitude(input logic [2:0] code);
        case (code)
            0: twice_magnitude = 0;
            1: twice_magnitude = 1;
            2: twice_magnitude = 2;
            3: twice_magnitude = 3;
            4: twice_magnitude = 4;
            5: twice_magnitude = 6;
            6: twice_magnitude = 8;
            7: twice_magnitude = 12;
        endcase
    endfunction

    always @(*) begin
        magnitude_a = twice_magnitude(a[2:0]);
        magnitude_b = twice_magnitude(b[2:0]);
        significand_a = {scale_a[6:3] != 0, scale_a[2:0]};
        significand_b = {scale_b[6:3] != 0, scale_b[2:0]};
        exponent_a = scale_a[6:3] == 0 ? -9 : int'({1'b0, scale_a[6:3]}) - 10;
        exponent_b = scale_b[6:3] == 0 ? -9 : int'({1'b0, scale_b[6:3]}) - 10;
        scaled_a = magnitude_a * significand_a;
        scaled_b = magnitude_b * significand_b;
        coefficient = scaled_a * scaled_b;
        leading = 0;
        for (integer bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
            if (coefficient[bit_index]) leading = bit_index;
        normalized = {8'd0, coefficient} << (23 - leading);
        exponent = 127 + leading + exponent_a + exponent_b - 2;
        product = coefficient == 0 ? {a[3] ^ b[3], 31'd0} :
            {a[3] ^ b[3], exponent[7:0], normalized[22:0]};
        // The command interface rejects these scales. Keep direct PE use
        // deterministic too: invalid padded bits or NaN produce canonical NaN.
        if (scale_a[7] || scale_b[7] || scale_a == 8'h7f || scale_b == 8'h7f)
            product = 32'h7fc00000;
    end
endmodule
