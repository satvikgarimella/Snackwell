// One vector instruction at a time, with one external word request outstanding.
// The external endpoint must share reset and discard pending responses on reset.
// Loads target ordinary RAM: equal addresses within an instruction share a read.
module global_memory_unit #(
    // Keep the serial implementation available for controlled comparisons.
    parameter bit BROADCAST_LOADS =
`ifdef SNACKWELL_SERIAL_GLOBAL_LOADS
        1'b0
`else
        1'b1
`endif
) (
    input  logic clk, reset,
    input  logic command_valid,
    output logic command_ready,
    input  logic command_store, command_status,
    input  logic [1:0] command_warp,
    input  logic [3:0] command_rd, command_mask,
    input  logic [127:0] command_bases, command_data,
    input  logic [31:0] command_offset,
    output logic busy,
    output logic [1:0] warp_id,
    output logic [11:0] status_bits,

    output logic request_valid,
    input  logic request_ready,
    output logic request_write,
    output logic [31:0] request_address, request_data,
    input  logic response_valid,
    output logic response_ready,
    input  logic response_error,
    input  logic [31:0] response_data,

    output logic completion_valid,
    input  logic completion_ready,
    output logic completion_writes_rd,
    output logic [3:0] completion_rd, completion_mask,
    output logic [127:0] completion_data
);
    typedef enum logic [2:0] {IDLE, SELECT_LANE, REQUEST, RESPONSE, COMPLETE} state_t;
    state_t state;
    logic [1:0] lane;
    logic [127:0] bases, stores;
    logic [31:0] offset;
    logic [3:0] remaining_mask, response_mask, selected_mask;
    logic [2:0] status [0:3];
    logic signed [32:0] address_sum;

    assign command_ready = !reset && state == IDLE;
    assign busy = state != IDLE;
    assign address_sum = $signed({1'b0, bases[lane*32 +: 32]}) +
                         $signed({offset[31], offset});
    assign request_valid = !reset && state == REQUEST;
    assign response_ready = !reset && state == RESPONSE;
    assign completion_valid = !reset && state == COMPLETE;
    assign completion_writes_rd = !request_write;
    always_comb begin
        selected_mask = 0;
        for (integer i = 0; i < 4; i = i + 1) begin
            // The offset is common to all lanes. Equal bases therefore have
            // equal effective addresses, including alignment/overflow checks.
            selected_mask[i] = remaining_mask[i] &&
                (i == int'(lane) || (BROADCAST_LOADS && !request_write &&
                 bases[i*32 +: 32] == bases[lane*32 +: 32]));
        end
    end
    for (genvar w = 0; w < 4; w = w + 1) begin : status_outputs
        assign status_bits[w*3 +: 3] = status[w];
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            lane <= 0;
            warp_id <= 0;
            bases <= 0;
            stores <= 0;
            offset <= 0;
            remaining_mask <= 0;
            response_mask <= 0;
            request_write <= 0;
            request_address <= 0;
            request_data <= 0;
            completion_rd <= 0;
            completion_mask <= 0;
            completion_data <= 0;
            for (integer w = 0; w < 4; w = w + 1) status[w] <= 0;
        end else begin
            case (state)
                IDLE: if (command_valid) begin
                    warp_id <= command_warp;
                    completion_rd <= command_rd;
                    completion_mask <= command_mask;
                    remaining_mask <= command_mask;
                    completion_data <= command_status
                        ? {4{29'd0, status[command_warp]}} : 128'd0;
                    bases <= command_bases;
                    stores <= command_data;
                    offset <= command_offset;
                    request_write <= command_store && !command_status;
                    lane <= 0;
                    if (command_status) state <= COMPLETE;
                    else state <= SELECT_LANE;
                end
                SELECT_LANE: begin
                    if (remaining_mask[lane] && !address_sum[32] && address_sum[1:0] == 0) begin
                        request_address <= address_sum[31:0];
                        request_data <= stores[lane*32 +: 32];
                        response_mask <= selected_mask;
                        state <= REQUEST;
                    end else begin
                        if (remaining_mask[lane]) begin
                            // Overflow includes a negative effective address.
                            status[warp_id] <= status[warp_id] |
                                {1'b0, address_sum[32], (address_sum[1:0] != 0)};
                            remaining_mask[lane] <= 0;
                        end
                        if (lane == 3) state <= COMPLETE;
                        else lane <= lane + 1'b1;
                    end
                end
                REQUEST: if (request_ready) state <= RESPONSE;
                RESPONSE: if (response_valid) begin
                    if (response_error) status[warp_id] <= status[warp_id] | 3'b100;
                    remaining_mask <= remaining_mask & ~response_mask;
                    if (!request_write) begin
                        for (integer i = 0; i < 4; i = i + 1)
                            if (response_mask[i])
                                completion_data[i*32 +: 32] <= response_error ? 32'd0 : response_data;
                    end
                    if (lane == 3) state <= COMPLETE;
                    else begin
                        lane <= lane + 1'b1;
                        state <= SELECT_LANE;
                    end
                end
                COMPLETE: if (completion_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
