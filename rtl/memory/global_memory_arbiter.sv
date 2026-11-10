// Two clients, one external transaction at a time. A stalled request keeps its
// grant until its response is consumed. Round-robin preference prevents starvation.
module global_memory_arbiter (
    input logic clk, reset,
    input logic lsu_valid, lsu_write,
    output logic lsu_ready,
    input logic [31:0] lsu_address, lsu_data,
    output logic lsu_response_valid,
    input logic lsu_response_ready,
    output logic lsu_error,
    output logic [31:0] lsu_result,
    input logic copy_valid,
    output logic copy_ready,
    input logic [31:0] copy_address,
    output logic copy_response_valid,
    input logic copy_response_ready,
    output logic copy_error,
    output logic [31:0] copy_result,
    output logic request_valid, request_write,
    input logic request_ready,
    output logic [31:0] request_address, request_data,
    input logic response_valid, response_error,
    output logic response_ready,
    input logic [31:0] response_data,
    output logic busy
);
    typedef enum logic [1:0] {IDLE, REQUEST, RESPONSE} state_t;
    state_t state;
    logic owner_copy, prefer_copy, select_copy;
    assign select_copy = state == IDLE ? copy_valid && (!lsu_valid || prefer_copy) : owner_copy;
    assign request_valid = !reset && state != RESPONSE && (select_copy ? copy_valid : lsu_valid);
    assign request_write = !select_copy && lsu_write;
    assign request_address = select_copy ? copy_address : lsu_address;
    assign request_data = select_copy ? 32'd0 : lsu_data;
    assign lsu_ready = !reset && state != RESPONSE && !select_copy && request_ready;
    assign copy_ready = !reset && state != RESPONSE && select_copy && request_ready;
    assign lsu_response_valid = !reset && state == RESPONSE && !owner_copy && response_valid;
    assign copy_response_valid = !reset && state == RESPONSE && owner_copy && response_valid;
    assign response_ready = !reset && state == RESPONSE &&
                            (owner_copy ? copy_response_ready : lsu_response_ready);
    assign lsu_error = response_error;
    assign copy_error = response_error;
    assign lsu_result = response_data;
    assign copy_result = response_data;
    assign busy = state != IDLE || request_valid;
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            owner_copy <= 0;
            prefer_copy <= 0;
        end else case (state)
            IDLE: if (request_valid) begin
                owner_copy <= select_copy;
                if (request_ready) state <= RESPONSE;
                else state <= REQUEST;
            end
            REQUEST: if (request_valid && request_ready) state <= RESPONSE;
            RESPONSE: if (response_valid && response_ready) begin
                prefer_copy <= !owner_copy;
                state <= IDLE;
            end
            default: state <= IDLE;
        endcase
    end
endmodule
