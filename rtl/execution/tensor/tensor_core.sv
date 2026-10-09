// Word interface around the array. Load all 16 words of A and B in any order,
// then launch. Results stream in row-major order and remain stable when stalled.
module tensor_core (
    input  logic        clk,
    input  logic        reset,
    input  logic        enable,
    input  logic        load_valid,
    output logic        load_ready,
    input  logic        load_matrix_b,
    input  logic [3:0]  load_index,
    input  logic [31:0] load_data,
    input  logic        start_valid,
    output logic        start_ready,
    output logic        busy,
    output logic        result_valid,
    input  logic        result_ready,
    output logic [3:0]  result_index,
    output logic [31:0] result_data,
    output logic        result_last
);
    logic [511:0] a_buffer, b_buffer;
    logic [15:0] a_loaded, b_loaded;
    wire array_ready, array_result_valid;
    wire [511:0] array_result;

    assign load_ready = !busy && !reset;
    // A simultaneous load has priority, so the command sees the updated tile
    // on a later handshake. Duplicate writes do not count as new elements.
    assign start_ready = (&a_loaded) && (&b_loaded) && array_ready && !load_valid && !reset;
    assign result_valid = array_result_valid;
    assign result_data = array_result[result_index*32 +: 32];
    assign result_last = result_valid && result_index == 4'd15;

    systolic_array_4x4 array (
        .clk(clk), .reset(reset), .enable(enable),
        .request_valid(start_valid && start_ready), .request_ready(array_ready),
        .a_tile(a_buffer), .b_tile(b_buffer), .busy(busy),
        .result_valid(array_result_valid), .result_ready(result_ready && result_last),
        .result_tile(array_result), .step_index(), .step_advance(), .pe_done_mask()
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            a_buffer <= 512'd0;
            b_buffer <= 512'd0;
            a_loaded <= 16'd0;
            b_loaded <= 16'd0;
            result_index <= 4'd0;
        end else begin
            if (load_valid && load_ready) begin
                if (load_matrix_b) begin
                    b_buffer[load_index*32 +: 32] <= load_data;
                    b_loaded[load_index] <= 1'b1;
                end else begin
                    a_buffer[load_index*32 +: 32] <= load_data;
                    a_loaded[load_index] <= 1'b1;
                end
            end
            if (start_valid && start_ready) begin
                a_loaded <= 16'd0;
                b_loaded <= 16'd0;
            end
            if (result_valid && result_ready)
                result_index <= result_last ? 4'd0 : result_index + 4'd1;
        end
    end
endmodule
