// Simulation-only external RAM. It returns stored words, never kernel answers.
// The DUT and this endpoint share reset; memory contents survive link reset.
module global_memory_model #(
    parameter integer WORDS = 16384
) (
    input logic clk, reset,
    input logic request_valid, request_write,
    output logic request_ready,
    input logic [31:0] request_address, request_data,
    output logic response_valid, response_error,
    input logic response_ready,
    output logic [31:0] response_data,
    input logic hold_requests, hold_responses,
    input integer latency,
    output integer reads, writes, responses, request_stalls
);
    logic [31:0] memory [0:WORDS-1];
    logic pending, saved_write, stalled;
    logic [31:0] saved_address, saved_data;
    logic [64:0] stalled_request;
    integer cycle, remaining;
    assign request_ready = !reset && !hold_requests && !pending && !response_valid && cycle % 7 >= 3;

    always @(posedge clk) begin
        if (reset) begin
            pending <= 0;
            response_valid <= 0;
            response_error <= 0;
            response_data <= 0;
            saved_write <= 0;
            saved_address <= 0;
            saved_data <= 0;
            stalled <= 0;
            stalled_request <= 0;
            cycle <= 0;
            remaining <= 0;
            reads <= 0; writes <= 0; responses <= 0; request_stalls <= 0;
        end else begin
            cycle <= cycle + 1;
            if (stalled && (!request_valid ||
                {request_write, request_address, request_data} !== stalled_request))
                $fatal(1, "Global request changed under backpressure");
            stalled <= request_valid && !request_ready;
            stalled_request <= {request_write, request_address, request_data};
            if (request_valid && !request_ready) request_stalls <= request_stalls + 1;
            if (response_valid && response_ready) begin
                response_valid <= 0;
                responses <= responses + 1;
            end
            if (request_valid && request_ready) begin
                if (pending || response_valid) $fatal(1, "Multiple external requests outstanding");
                pending <= 1;
                saved_write <= request_write;
                saved_address <= request_address;
                saved_data <= request_data;
                remaining <= latency + int'(request_address[5:2]) % 5;
                if (request_write) writes <= writes + 1;
                else reads <= reads + 1;
            end
            if (pending && !hold_responses) begin
                if (remaining != 0) remaining <= remaining - 1;
                else begin
                    pending <= 0;
                    response_valid <= 1;
                    response_error <= saved_address[1:0] != 0 || saved_address >= WORDS*4;
                    response_data <= 0;
                    if (saved_address[1:0] == 0 && saved_address < WORDS*4) begin
                        if (saved_write) memory[saved_address/4] <= saved_data;
                        else response_data <= memory[saved_address/4];
                    end
                end
            end
        end
    end
endmodule
