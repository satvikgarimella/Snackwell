module shared_memory (
    input  logic         clk,
    input  logic         reset,

    // One vector request at a time. All operands/masks are captured on
    // request_valid && request_ready, including requests with no active lanes.
    input  logic         request_valid,
    output logic         request_ready,
    input  logic [127:0] lane_addresses,
    input  logic [127:0] store_data,
    input  logic [3:0]   load_enable_mask,
    input  logic [3:0]   store_enable_mask,

    // Both loads and stores produce a completion. Load data (inactive lanes
    // zero) remains stable until response_valid && response_ready.
    output logic         response_valid,
    input  logic         response_ready,
    output logic [127:0] load_data,

    // Conflicts among the incoming request, or the lanes still being served.
    output logic [3:0]   conflict_mask,
    output logic         bank_conflict
);
    // 4 KB, four banks of 256 32-bit words. Byte address [3:2] selects
    // the bank, [11:4] the row. Alignment/address aliasing is unchanged.
    logic [31:0] bank0 [0:255];
    logic [31:0] bank1 [0:255];
    logic [31:0] bank2 [0:255];
    logic [31:0] bank3 [0:255];

    logic servicing;
    logic [127:0] saved_addresses, saved_store_data;
    logic [3:0] saved_load_mask, saved_store_mask, pending_mask;
    logic [3:0] service_mask, banks_used;
    logic [127:0] conflict_addresses;
    logic [3:0] conflict_requests;

    assign request_ready = !reset && !servicing && !response_valid;
    assign bank_conflict = |conflict_mask;

    // Each bank services its lowest-numbered pending lane independently.
    // A four-way collision takes four service cycles; different banks work
    // in parallel. Same-address stores consequently commit in lane order.
    always @(*) begin
        service_mask = 4'b0;
        banks_used = 4'b0;
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
            if (pending_mask[lane] && !banks_used[saved_addresses[lane*32+2 +: 2]]) begin
                service_mask[lane] = 1'b1;
                banks_used[saved_addresses[lane*32+2 +: 2]] = 1'b1;
            end
        end
    end

    always @(*) begin
        conflict_addresses = servicing ? saved_addresses : lane_addresses;
        conflict_requests = servicing ? pending_mask :
            ((request_valid && request_ready) ? load_enable_mask | store_enable_mask : 4'b0);
        conflict_mask = 4'b0;
        for (integer a = 0; a < 4; a = a + 1)
            for (integer b = 0; b < 4; b = b + 1)
                if (a != b && conflict_requests[a] && conflict_requests[b] &&
                    conflict_addresses[a*32+2 +: 2] == conflict_addresses[b*32+2 +: 2])
                    conflict_mask[a] = 1'b1;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            servicing <= 1'b0;
            response_valid <= 1'b0;
            saved_addresses <= 128'b0;
            saved_store_data <= 128'b0;
            saved_load_mask <= 4'b0;
            saved_store_mask <= 4'b0;
            pending_mask <= 4'b0;
            load_data <= 128'b0;
            for (integer row = 0; row < 256; row = row + 1) begin
                bank0[row] <= 32'b0;
                bank1[row] <= 32'b0;
                bank2[row] <= 32'b0;
                bank3[row] <= 32'b0;
            end
        end else begin
            if (response_valid && response_ready)
                response_valid <= 1'b0;

            if (request_valid && request_ready) begin
                saved_addresses <= lane_addresses;
                saved_store_data <= store_data;
                saved_load_mask <= load_enable_mask;
                saved_store_mask <= store_enable_mask;
                pending_mask <= load_enable_mask | store_enable_mask;
                load_data <= 128'b0;
                servicing <= 1'b1;
            end

            if (servicing) begin
                for (integer lane = 0; lane < 4; lane = lane + 1) begin
                    if (service_mask[lane]) begin
                        // A lane requesting both reads and writes observes
                        // the old word, then stores its new value at this edge.
                        if (saved_load_mask[lane]) begin
                            case (saved_addresses[lane*32+2 +: 2])
                                0: load_data[lane*32 +: 32] <= bank0[saved_addresses[lane*32+4 +: 8]];
                                1: load_data[lane*32 +: 32] <= bank1[saved_addresses[lane*32+4 +: 8]];
                                2: load_data[lane*32 +: 32] <= bank2[saved_addresses[lane*32+4 +: 8]];
                                3: load_data[lane*32 +: 32] <= bank3[saved_addresses[lane*32+4 +: 8]];
                            endcase
                        end
                        if (saved_store_mask[lane]) begin
                            case (saved_addresses[lane*32+2 +: 2])
                                0: bank0[saved_addresses[lane*32+4 +: 8]] <= saved_store_data[lane*32 +: 32];
                                1: bank1[saved_addresses[lane*32+4 +: 8]] <= saved_store_data[lane*32 +: 32];
                                2: bank2[saved_addresses[lane*32+4 +: 8]] <= saved_store_data[lane*32 +: 32];
                                3: bank3[saved_addresses[lane*32+4 +: 8]] <= saved_store_data[lane*32 +: 32];
                            endcase
                        end
                    end
                end
                pending_mask <= pending_mask & ~service_mask;
                if ((pending_mask & ~service_mask) == 4'b0) begin
                    servicing <= 1'b0;
                    response_valid <= 1'b1;
                end
            end
        end
    end
endmodule
