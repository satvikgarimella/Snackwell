// One C=A*B tile at a time, with 16 physical FP32 MAC PEs.
// Operand transport advances in lockstep after every PE has completed a step.
// enable pauses step launch/transport, never the clocks of in-flight arithmetic.
module systolic_array_4x4 (
    input  logic         clk,
    input  logic         reset,
    input  logic         enable,
    input  logic         request_valid,
    output logic         request_ready,
    input  logic [511:0] a_tile,
    input  logic [511:0] b_tile,
    output logic         busy,
    output logic         result_valid,
    input  logic         result_ready,
    output logic [511:0] result_tile,
    output logic [3:0]   step_index,
    output logic         step_advance,
    output wire [15:0]   pe_done_mask
);
    typedef enum logic [2:0] {IDLE, CLEAR, START_STEP, WAIT_STEP, RESULT} state_t;
    state_t state;
    logic [511:0] saved_a, saved_b;
    wire [127:0] a_edges, b_edges;
    wire [3:0] a_edge_valid, b_edge_valid;
    wire [15:0] pe_ready_mask;
    wire [511:0] accumulators;
    wire pe_clear = state == CLEAR;
    wire pe_start = state == START_STEP && enable && (&pe_ready_mask) && !reset;
    wire [31:0] a_links [0:3][0:4];
    wire [31:0] b_links [0:4][0:3];
    wire a_valid_links [0:3][0:4];
    wire b_valid_links [0:4][0:3];

    assign request_ready = state == IDLE && !reset;
    assign busy = state != IDLE;
    assign result_valid = state == RESULT && !reset;
    assign step_advance = state == WAIT_STEP && (&pe_done_mask) && enable && !reset;

    systolic_feeder_4x4 feeder (
        .a_tile(saved_a), .b_tile(saved_b), .step_index(step_index),
        .a_edges(a_edges), .b_edges(b_edges),
        .a_valid(a_edge_valid), .b_valid(b_edge_valid)
    );

    for (genvar r = 0; r < 4; r = r + 1) begin : rows
        assign a_links[r][0] = a_edges[r*32 +: 32];
        assign a_valid_links[r][0] = a_edge_valid[r];
        for (genvar c = 0; c < 4; c = c + 1) begin : columns
            if (r == 0) begin : top_edge
                assign b_links[0][c] = b_edges[c*32 +: 32];
                assign b_valid_links[0][c] = b_edge_valid[c];
            end
            fp32_mac_pe pe (
                .clk(clk), .reset(reset), .clear(pe_clear),
                .step_valid(pe_start), .step_ready(pe_ready_mask[r*4+c]),
                .a_valid(a_valid_links[r][c]), .b_valid(b_valid_links[r][c]),
                .a_data(a_links[r][c]), .b_data(b_links[r][c]),
                .advance(step_advance), .step_done(pe_done_mask[r*4+c]),
                .a_out_valid(a_valid_links[r][c+1]), .b_out_valid(b_valid_links[r+1][c]),
                .a_out(a_links[r][c+1]), .b_out(b_links[r+1][c]),
                .accumulator(accumulators[(r*4+c)*32 +: 32])
            );
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            saved_a <= 512'd0;
            saved_b <= 512'd0;
            result_tile <= 512'd0;
            step_index <= 4'd0;
        end else begin
            case (state)
                IDLE: if (request_valid && request_ready) begin
                    saved_a <= a_tile;
                    saved_b <= b_tile;
                    step_index <= 4'd0;
                    state <= CLEAR;
                end
                CLEAR: state <= START_STEP;
                START_STEP: if (pe_start) state <= WAIT_STEP;
                WAIT_STEP: if (step_advance) begin
                    if (step_index == 4'd9) begin
                        result_tile <= accumulators;
                        state <= RESULT;
                    end else begin
                        step_index <= step_index + 4'd1;
                        state <= START_STEP;
                    end
                end
                RESULT: if (result_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
