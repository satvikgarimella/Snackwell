// Register-file adapter for TMUL: four vector registers per row-major tile.
// Owns both read ports for four gather cycles, then releases them while the
// systolic array runs. The destination reservation lasts until
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
    // Keep trace state IDs stable; state 2 was the former serial operand load.
    typedef enum logic [2:0] {IDLE=0, GATHER=1, START=3, RESULT=4, COMMIT=5} state_t;
    state_t state;
    logic [3:0] a_base, b_base, d_base;
    logic [1:0] gather_row;
    logic [1:0] result_row;
    logic [511:0] a_tile, b_tile;
    wire start_ready, result_valid;
    wire [511:0] result_tile;

    assign request_ready = state == IDLE && !reset;
    assign busy = state != IDLE;
    assign gather_active = state == GATHER;
    assign read_a_addr = a_base + {2'd0, gather_row};
    assign read_b_addr = b_base + {2'd0, gather_row};
    assign destination_mask = busy ? (16'h000f << d_base) : 16'd0;
    assign row_valid = state == RESULT && result_valid;
    assign row_data = result_tile[result_row*128 +: 128];
    assign row_rd_addr = d_base + {2'd0, result_row};
    assign row_last = row_valid && result_row == 2'd3;

    systolic_array_4x4 tensor (
        .clk(clk), .reset(reset),
        // As with SIMD FP, pausing SM issue does not strand an accepted job.
        .enable(1'b1),
        .request_valid(state == START), .request_ready(start_ready), .busy(),
        .a_tile(a_tile), .b_tile(b_tile),
        .result_valid(result_valid), .result_ready(row_last && row_ready),
        .result_tile(result_tile), .step_index(), .step_advance(), .pe_done_mask()
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
            result_row <= 2'd0;
            a_tile <= 512'd0;
            b_tile <= 512'd0;
        end else begin
            case (state)
                IDLE: if (request_valid && request_ready) begin
                    warp_id <= request_warp_id;
                    a_base <= request_a_base;
                    b_base <= request_b_base;
                    d_base <= request_d_base;
                    row_write_mask <= request_write_mask;
                    gather_row <= 2'd0;
                    result_row <= 2'd0;
                    state <= GATHER;
                end
                GATHER: begin
                    a_tile[gather_row*128 +: 128] <= read_a_data;
                    b_tile[gather_row*128 +: 128] <= read_b_data;
                    if (gather_row == 2'd3) begin
                        state <= START;
                    end else gather_row <= gather_row + 2'd1;
                end
                START: if (start_ready) state <= RESULT;
                RESULT: if (row_valid && row_ready) begin
                    if (row_last) state <= COMMIT;
                    else result_row <= result_row + 2'd1;
                end
                COMMIT: if (last_row_committed) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
