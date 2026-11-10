// Rectangular global-to-shared copy with two protected destination regions.
// Descriptor, ticket, failure, and ownership rules: docs/ASYNC_COPY.md.
module async_copy_unit (
    input logic clk, reset,
    input logic command_valid,
    output logic command_ready,
    input logic [5:0] opcode,
    input logic [1:0] warp_id,
    input logic [3:0] active_mask, rd_addr, rs1_addr, rs2_addr,
    input logic [13:0] immediate,
    input logic [127:0] source,
    input logic memory_idle,
    output logic busy,
    output logic [3:0] warp_pending,
    output logic [31:0] status_bits,

    // Guard ordinary shared operations and a candidate tensor operand gather.
    input logic [127:0] probe_addresses,
    input logic probe_store,
    input logic [31:0] matrix_a, matrix_b,
    input logic matrix_valid, matrix_nvfp4,
    output logic shared_access_ready, matrix_access_ready,

    output logic global_valid,
    input logic global_ready,
    output logic [31:0] global_address,
    input logic global_response_valid, global_error,
    output logic global_response_ready,
    input logic [31:0] global_data,
    output logic shared_valid,
    input logic shared_ready,
    output logic [31:0] shared_address, shared_data,
    input logic shared_response_valid,
    output logic shared_response_ready,

    output logic row_valid,
    input logic row_ready,
    output logic [1:0] row_warp,
    output logic [3:0] row_rd, row_mask,
    output logic [127:0] row_data
);
    import snackwell_pkg::*;
    typedef enum logic [2:0] {IDLE, GLOBAL_REQUEST, GLOBAL_RESPONSE, SHARED_REQUEST, SHARED_RESPONSE} state_t;
    state_t state;
    logic [1:0] live, completed, failed;
    logic [3:0] consumers [0:1], waited [0:1];
    logic [31:0] generation [0:1];
    logic [11:0] region_first [0:1], region_last [0:1];
    logic [7:0] errors [0:3];
    logic job_slot;
    logic [1:0] job_warp;
    logic [31:0] source_base, source_stride;
    logic [11:0] destination_base, destination_stride;
    logic [4:0] rows, columns, row_index, column_index;
    logic encoding_valid, descriptor_valid, returns_register;
    logic [7:0] command_error;
    logic [127:0] command_result;
    logic [63:0] source_last, destination_last;
    logic [31:0] matrix_last_a, matrix_last_b;
    logic [11:0] probe_word;
    wire slot = immediate[0];
    wire [3:0] participants = immediate[4:1];
    wire [7:0] desc_rows = source[71:64];
    wire [7:0] desc_columns = source[79:72];
    wire [31:0] desc_destination = source[127:96];

    assign busy = state != IDLE;
    assign global_valid = !reset && state == GLOBAL_REQUEST;
    assign global_response_ready = !reset && state == GLOBAL_RESPONSE;
    assign shared_valid = !reset && state == SHARED_REQUEST;
    assign shared_response_ready = !reset && state == SHARED_RESPONSE;
    assign global_address = source_base + {27'd0, row_index} * source_stride +
                            {25'd0, column_index, 2'b0};
    assign shared_address = {20'd0, destination_base} +
                            {27'd0, row_index} * {20'd0, destination_stride} +
                            {25'd0, column_index, 2'b0};
    for (genvar w = 0; w < 4; w = w + 1) begin : statuses
        assign status_bits[w*8 +: 8] = errors[w];
    end

    always @(*) begin
        warp_pending = consumers[0] | consumers[1];
        if (busy) warp_pending[job_warp] = 1'b1;
        if (row_valid) warp_pending[row_warp] = 1'b1;
        shared_access_ready = 1'b1;
        matrix_access_ready = 1'b1;
        probe_word = 0;
        matrix_last_a = tensor_descriptor_last(matrix_a, matrix_nvfp4);
        matrix_last_b = tensor_descriptor_last(matrix_b, matrix_nvfp4);
        for (integer s = 0; s < 2; s = s + 1) begin
            if (live[s]) begin
                for (integer lane = 0; lane < 4; lane = lane + 1) begin
                    // Preserve legacy shared-memory aliasing, including word alignment.
                    probe_word = {probe_addresses[lane*32+2 +: 10], 2'b0};
                    if (active_mask[lane] && probe_word >= region_first[s] && probe_word <= region_last[s] &&
                        (probe_store || !completed[s] || failed[s] || !waited[s][warp_id]))
                        shared_access_ready = 1'b0;
                end
                if (matrix_valid && (!completed[s] || failed[s] || !waited[s][warp_id]) &&
                    ((tensor_descriptor_valid(matrix_a, matrix_nvfp4) && matrix_a[11:0] <= region_last[s] &&
                      matrix_last_a >= {20'd0, region_first[s]}) ||
                     (tensor_descriptor_valid(matrix_b, matrix_nvfp4) && matrix_b[11:0] <= region_last[s] &&
                      matrix_last_b >= {20'd0, region_first[s]}))) matrix_access_ready = 1'b0;
            end
        end
    end

    always @(*) begin
        source_last = {32'd0, source[31:0]} +
            ({56'd0, desc_rows} - 64'd1) * {32'd0, source[63:32]} +
            {54'd0, desc_columns, 2'b0} - 64'd1;
        destination_last = {52'd0, desc_destination[11:0]} +
            ({56'd0, desc_rows} - 64'd1) * {52'd0, desc_destination[23:12]} +
            {54'd0, desc_columns, 2'b0} - 64'd1;
        descriptor_valid = desc_rows >= 1 && desc_rows <= 16 &&
            desc_columns >= 1 && desc_columns <= 16 && source[95:80] == 0 &&
            source[1:0] == 0 && source[33:32] == 0 &&
            source[63:32] >= {22'd0, desc_columns, 2'b0} && source_last <= 64'hffffffff &&
            desc_destination[31:24] == 0 && desc_destination[1:0] == 0 &&
            desc_destination[13:12] == 0 &&
            {20'd0, desc_destination[23:12]} >= {22'd0, desc_columns, 2'b0} && destination_last < 4096;
        command_ready = !reset && (!row_valid || row_ready);
        command_error = 0;
        command_result = 0;
        returns_register = opcode == OP_CPASYNC || opcode == OP_CPSTATUS;
        encoding_valid = 1'b0;
        case (opcode)
            OP_CPASYNC: encoding_valid = rs2_addr == 0 && immediate[13:5] == 0 && participants != 0;
            OP_CPWAIT: encoding_valid = rd_addr == 0 && rs2_addr == 0 && immediate[13:1] == 0;
            OP_CPRELEASE: encoding_valid = rd_addr == 0 && rs1_addr == 0 && rs2_addr == 0 && immediate[13:1] == 0;
            OP_CPSTATUS: encoding_valid = rs1_addr == 0 && rs2_addr == 0 && immediate == 0;
            default: encoding_valid = 1'b0;
        endcase
        if (!encoding_valid || active_mask != 15 ||
            (opcode == OP_CPWAIT && source != {4{source[31:0]}})) command_error = 8'h01;
        else case (opcode)
            OP_CPASYNC: begin
                if (!descriptor_valid) command_error = 8'h04;
                else if (generation[slot] == 32'hffffffff) command_error = 8'h20;
                else if (busy || live[slot] || !memory_idle ||
                    (live[!slot] && desc_destination[11:0] <= region_last[!slot] &&
                     destination_last >= {52'd0, region_first[!slot]})) command_ready = 1'b0;
                else command_result = {4{generation[slot] + 32'd1}};
            end
            OP_CPWAIT: begin
                if (source[31:0] == 0 || source[31:0] < generation[slot]) command_error = 8'h08;
                else if (source[31:0] > generation[slot]) command_ready = 1'b0;
                else if (!live[slot]) command_error = 8'h08;
                else if (!consumers[slot][warp_id]) command_error = 8'h02;
                else if (!completed[slot]) command_ready = 1'b0;
                // A failed copy can be waited and released, but never read.
            end
            OP_CPRELEASE: begin
                if (!consumers[slot][warp_id]) command_error = 8'h02;
                else if (!waited[slot][warp_id]) command_error = 8'h40;
                else if (!memory_idle) command_ready = 1'b0;
            end
            OP_CPSTATUS: command_result = {4{24'd0, errors[warp_id]}};
            default: command_error = 8'h01;
        endcase
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            live <= 0; completed <= 0; failed <= 0;
            job_slot <= 0; job_warp <= 0;
            source_base <= 0; source_stride <= 0;
            destination_base <= 0; destination_stride <= 0;
            rows <= 0; columns <= 0; row_index <= 0; column_index <= 0;
            shared_data <= 0;
            row_valid <= 0; row_warp <= 0; row_rd <= 0; row_mask <= 0; row_data <= 0;
            for (integer s = 0; s < 2; s = s + 1) begin
                consumers[s] <= 0; waited[s] <= 0; generation[s] <= 0;
                region_first[s] <= 0; region_last[s] <= 0;
            end
            for (integer w = 0; w < 4; w = w + 1) errors[w] <= 0;
        end else begin
            if (row_valid && row_ready) row_valid <= 0;
            case (state)
                GLOBAL_REQUEST: if (global_ready) state <= GLOBAL_RESPONSE;
                GLOBAL_RESPONSE: if (global_response_valid) begin
                    if (global_error) begin
                        failed[job_slot] <= 1;
                        completed[job_slot] <= 1;
                        errors[job_warp] <= errors[job_warp] | 8'h10;
                        state <= IDLE;
                    end else begin
                        shared_data <= global_data;
                        state <= SHARED_REQUEST;
                    end
                end
                SHARED_REQUEST: if (shared_ready) state <= SHARED_RESPONSE;
                SHARED_RESPONSE: if (shared_response_valid) begin
                    if (column_index + 1'b1 == columns && row_index + 1'b1 == rows) begin
                        completed[job_slot] <= 1;
                        state <= IDLE;
                    end else begin
                        if (column_index + 1'b1 == columns) begin
                            column_index <= 0;
                            row_index <= row_index + 1'b1;
                        end else column_index <= column_index + 1'b1;
                        state <= GLOBAL_REQUEST;
                    end
                end
                default: begin end
            endcase
            if (command_valid && command_ready) begin
                if (returns_register) begin
                    row_valid <= 1;
                    row_warp <= warp_id; row_rd <= rd_addr; row_mask <= active_mask;
                    row_data <= command_error == 0 ? command_result : 128'd0;
                end
                // Preserve simultaneous bus and command errors for the same warp.
                if (command_error != 0) errors[warp_id] <= errors[warp_id] | command_error |
                    ((state == GLOBAL_RESPONSE && global_response_valid && global_error && job_warp == warp_id)
                        ? 8'h10 : 8'h00);
                else case (opcode)
                    OP_CPASYNC: begin
                        live[slot] <= 1; completed[slot] <= 0; failed[slot] <= 0;
                        consumers[slot] <= participants; waited[slot] <= 0;
                        generation[slot] <= generation[slot] + 1'b1;
                        region_first[slot] <= desc_destination[11:0]; region_last[slot] <= destination_last[11:0];
                        job_slot <= slot; job_warp <= warp_id;
                        source_base <= source[31:0]; source_stride <= source[63:32];
                        destination_base <= desc_destination[11:0]; destination_stride <= desc_destination[23:12];
                        rows <= desc_rows[4:0]; columns <= desc_columns[4:0];
                        row_index <= 0; column_index <= 0;
                        state <= GLOBAL_REQUEST;
                    end
                    OP_CPWAIT: begin
                        waited[slot][warp_id] <= 1;
                        if (failed[slot]) errors[warp_id] <= errors[warp_id] | 8'h10;
                    end
                    OP_CPRELEASE: begin
                        consumers[slot][warp_id] <= 0;
                        waited[slot][warp_id] <= 0;
                        if (consumers[slot] == (4'b0001 << warp_id)) live[slot] <= 0;
                    end
                    default: begin end
                endcase
            end
        end
    end
endmodule
