// Row-major tiles: element [r][c] occupies bits 32*(4*r+c) +: 32.
// At logical step t, row r receives A[r][t-r], column c receives B[t-c][c].
// Tokens outside k=0..3 are invalid bubbles, including the drain steps.
module systolic_feeder_4x4 (
    input  logic [511:0] a_tile,
    input  logic [511:0] b_tile,
    input  logic [3:0]   step_index,
    output logic [127:0] a_edges,
    output logic [127:0] b_edges,
    output logic [3:0]   a_valid,
    output logic [3:0]   b_valid
);
    integer k;
    always @(*) begin
        a_edges = 128'd0;
        b_edges = 128'd0;
        a_valid = 4'd0;
        b_valid = 4'd0;
        k = 0;
        for (integer i = 0; i < 4; i = i + 1) begin
            k = int'(step_index) - i;
            if (k >= 0 && k < 4) begin
                a_valid[i] = 1'b1;
                b_valid[i] = 1'b1;
                a_edges[i*32 +: 32] = a_tile[(i*4+k)*32 +: 32];
                b_edges[i*32 +: 32] = b_tile[(k*4+i)*32 +: 32];
            end
        end
    end
endmodule
