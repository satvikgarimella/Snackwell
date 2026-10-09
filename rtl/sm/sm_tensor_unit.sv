// Register-file adapter for TMUL: four vector registers per row-major tile.
// Owns both read ports for four gather cycles, then releases them while the
// word-interface tensor core runs. The destination reservation lasts until
// the last row actually commits through the SM's shared writeback port.
module sm_tensor_unit (
    input  logic         clk,
    input  logic         reset,
    input  logic         request_valid,
    output logic         request_ready,
    input  logic [1:0]   request_warp_id,
    input  logic [3:0]   request_a_base,
    input  logic [3:0]   request_b_base,
    input  logic [3:0]   request_d_base,
    input  logic [3:0]   request_write_mask,

    output logic         gather_active,
    output logic [1:0]   warp_id,
    output logic [3:0]   read_a_addr,
    output logic [3:0]   read_b_addr,
    input  logic [127:0] read_a_data,
    input  logic [127:0] read_b_data,

    output logic         busy,
    output logic [15:0]  destination_mask,
    output logic         row_valid,
    input  logic         row_ready,
    output logic [3:0]   row_rd_addr,
    output logic [3:0]   row_write_mask,
    output logic [127:0] row_data,
    output logic         row_last,
    input  logic         last_row_committed
);
    typedef enum logic [2:0] {IDLE, GATHER, LOAD, START, RESULT, COMMIT} state_t;
    state_t state;
    logic [3:0] a_base, b_base, d_base;
    logic [1:0] gather_row;
    logic [4:0] load_word;
    logic [511:0] a_tile, b_tile;
    logic [95:0] row_buffer;
    wire load_ready, start_ready, result_valid, result_ready, result_last;
    wire [3:0] result_index;
    wire [31:0] result_data;

    assign request_ready = state == IDLE && !reset;
    assign busy = state != IDLE;
    assign gather_active = state == GATHER;
    assign read_a_addr = a_base + {2'd0, gather_row};
    assign read_b_addr = b_base + {2'd0, gather_row};
    assign destination_mask = busy ? (16'h000f << d_base) : 16'd0;
    assign row_valid = state == RESULT && result_valid && result_index[1:0] == 2'd3;
    assign row_data = {result_data, row_buffer};
    assign row_rd_addr = d_base + {2'd0, result_index[3:2]};
    assign row_last = result_last;
    assign result_ready = state == RESULT && (result_index[1:0] != 2'd3 || row_ready);

    tensor_core tensor (
        .clk(clk), .reset(reset),
        // As with SIMD FP, pausing SM issue does not strand an accepted job.
        .enable(1'b1),
        .load_valid(state == LOAD), .load_ready(load_ready),
        .load_matrix_b(load_word[4]), .load_index(load_word[3:0]),
        .load_data(load_word[4] ? b_tile[load_word[3:0]*32 +: 32]
                               : a_tile[load_word[3:0]*32 +: 32]),
        .start_valid(state == START), .start_ready(start_ready), .busy(),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_index(result_index), .result_data(result_data), .result_last(result_last)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            warp_id <= 2'd0;
            a_base <= 4'd0;
            b_base <= 4'd0;
            d_base <= 4'd0;
            row_write_mask <= 4'd0;
            gather_row <= 2'd0;
            load_word <= 5'd0;
            a_tile <= 512'd0;
            b_tile <= 512'd0;
            row_buffer <= 96'd0;
        end else begin
            case (state)
                IDLE: if (request_valid && request_ready) begin
                    warp_id <= request_warp_id;
                    a_base <= request_a_base;
                    b_base <= request_b_base;
                    d_base <= request_d_base;
                    row_write_mask <= request_write_mask;
                    gather_row <= 2'd0;
                    state <= GATHER;
                end
                GATHER: begin
                    a_tile[gather_row*128 +: 128] <= read_a_data;
                    b_tile[gather_row*128 +: 128] <= read_b_data;
                    if (gather_row == 2'd3) begin
                        load_word <= 5'd0;
                        state <= LOAD;
                    end else gather_row <= gather_row + 2'd1;
                end
                LOAD: if (load_ready) begin
                    if (load_word == 5'd31) state <= START;
                    else load_word <= load_word + 5'd1;
                end
                START: if (start_ready) state <= RESULT;
                RESULT: if (result_valid && result_ready) begin
                    if (result_index[1:0] != 2'd3)
                        row_buffer[result_index[1:0]*32 +: 32] <= result_data;
                    if (result_last) state <= COMMIT;
                end
                COMMIT: if (last_row_committed) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
