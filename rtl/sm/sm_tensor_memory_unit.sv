// Two architectural 4x4 FP32 accumulator tiles, backed by the shared array.
// See docs/TENSOR_MEMORY.md for instruction, ownership and numerical contracts.
module sm_tensor_memory_unit (
    input  logic clk, reset,
    input  logic command_valid,
    output logic command_ready,
    input  logic [5:0] opcode,
    input  logic [1:0] warp_id,
    input  logic [3:0] active_mask, rd_addr, rs1_addr, rs2_addr,
    input  logic [13:0] immediate,
    input  logic [127:0] source_a, source_b,
    input  logic array_available, memory_idle,
    output logic busy, memory_lock,
    output logic matrix_guard_valid,
    output logic [3:0] warp_pending,

    output logic memory_request_valid,
    input  logic memory_request_ready,
    output logic [127:0] memory_addresses,
    input  logic memory_response_valid,
    output logic memory_response_ready,
    input  logic [127:0] memory_data,

    output logic array_request_valid,
    output logic array_nvfp4_mode,
    input  logic array_request_ready,
    output logic [511:0] a_tile, b_tile, c_tile,
    input  logic array_result_valid,
    output logic array_result_ready,
    input  logic [511:0] array_result,

    output logic row_valid,
    input  logic row_ready,
    output logic [1:0] row_warp_id,
    output logic [3:0] row_rd_addr, row_write_mask,
    output logic [127:0] row_data
);
    import snackwell_pkg::*;
    typedef enum logic [2:0] {IDLE, FETCH_REQUEST, FETCH_RESPONSE, START, COMPUTE, PREPARE} state_t;
    state_t state;
    logic [511:0] tiles [0:1];
    logic [1:0] owner [0:1];
    logic [1:0] allocated, job_live, committed, completed, failed;
    logic [3:0] initialized [0:1], readers [0:1], visible [0:1];
    logic [31:0] generation [0:1];
    logic [31:0] errors [0:3];
    logic job_slot, job_accumulate, job_nvfp4;
    logic [1:0] chunk;
    logic [255:0] packed_a, packed_b;
    logic [31:0] scales_a, scales_b;
    logic [31:0] descriptor_a, descriptor_b;
    logic [2:0] fetch_row;
    logic [31:0] fetch_descriptor;
    logic fetch_b, bad_scale;
    logic [2:0] fetch_word;
    logic [31:0] command_error;
    logic [127:0] command_result;
    logic encoding_valid, returns_register;
    wire slot = immediate[0];
    wire [1:0] tile_row = immediate[2:1];
    wire [3:0] participants = immediate[4:1];
    wire own_slot = allocated[slot] && owner[slot] == warp_id;
    wire slot_busy = busy && job_slot == slot;
    wire uniform_a = source_a == {4{source_a[31:0]}};
    wire uniform_b = source_b == {4{source_b[31:0]}};
    wire writes_matrix = tensor_matrix_opcode(opcode);
    wire command_nvfp4 = tensor_nvfp4_opcode(opcode);
    wire command_accumulate = opcode == OP_TMMA_ACC || opcode == OP_TMMA_NVFP4_ACC;
    wire unpublished = job_live[slot] && !committed[slot];
    wire scale_failure = state == FETCH_RESPONSE && memory_response_valid &&
        job_nvfp4 && fetch_word == 2 && bad_scale;

    assign busy = state != IDLE;
    assign memory_lock = state == FETCH_REQUEST || state == FETCH_RESPONSE;
    assign memory_request_valid = state == FETCH_REQUEST && !reset;
    assign memory_response_ready = state == FETCH_RESPONSE && !reset;
    assign array_request_valid = state == START && !reset;
    assign array_nvfp4_mode = job_nvfp4;
    assign array_result_ready = state == COMPUTE && !reset;
    assign c_tile = (job_accumulate || (job_nvfp4 && chunk != 0)) ? tiles[job_slot] : 512'd0;
    assign matrix_guard_valid = writes_matrix && command_error == 0;
    always @(*) begin
        fetch_b = job_nvfp4 ? fetch_row >= 3 : fetch_row[2];
        fetch_word = fetch_b ? fetch_row - 3 : fetch_row;
        fetch_descriptor = fetch_b ? descriptor_b : descriptor_a;
        for (integer lane = 0; lane < 4; lane = lane + 1)
            memory_addresses[lane*32 +: 32] = job_nvfp4 ?
                {20'd0, fetch_descriptor[11:0]} + lane * {20'd0, fetch_descriptor[23:12]} +
                {27'd0, fetch_word, 2'b0} :
                {20'd0, fetch_descriptor[11:0]} +
                {30'd0, fetch_row[1:0]} * {20'd0, fetch_descriptor[23:12]} + 4 * lane;
        bad_scale = 1'b0;
        for (integer lane = 0; lane < 4; lane = lane + 1)
            if (memory_data[lane*32+7 +: 25] != 0 || memory_data[lane*32 +: 7] == 7'h7f)
                bad_scale = 1'b1;
        warp_pending = 4'd0;
        for (integer tile = 0; tile < 2; tile = tile + 1) begin
            if (allocated[tile]) warp_pending[owner[tile]] = 1'b1;
            warp_pending = warp_pending | readers[tile];
        end
        if (row_valid) warp_pending[row_warp_id] = 1'b1;
    end

    // Invalid commands retire with a sticky per-warp status and no tile/memory
    // side effects. Register-returning failures still retire their scoreboard.
    // Resource waits use command_ready, never a register dependency stall.
    always @(*) begin
        command_ready = !reset && (!row_valid || row_ready);
        command_error = 32'd0;
        command_result = 128'd0;
        returns_register = opcode == OP_TLD || opcode == OP_TCOMMIT || opcode == OP_TSTATUS;
        encoding_valid = 1'b0;
        case (opcode)
            OP_TALLOC, OP_TFREE, OP_TRELEASE:
                encoding_valid = rd_addr == 0 && rs1_addr == 0 && rs2_addr == 0 && immediate[13:1] == 0;
            OP_TST:
                encoding_valid = rd_addr == 0 && rs2_addr == 0 && immediate[13:3] == 0;
            OP_TLD:
                encoding_valid = rs1_addr == 0 && rs2_addr == 0 && immediate[13:3] == 0;
            OP_TMMA, OP_TMMA_ACC, OP_TMMA_NVFP4, OP_TMMA_NVFP4_ACC:
                encoding_valid = rd_addr == 0 && immediate[13:1] == 0;
            OP_TCOMMIT:
                encoding_valid = rs1_addr == 0 && rs2_addr == 0 && immediate[13:5] == 0 && participants != 0;
            OP_TWAIT:
                encoding_valid = rd_addr == 0 && rs2_addr == 0 && immediate[13:1] == 0;
            OP_TSTATUS:
                encoding_valid = rs1_addr == 0 && rs2_addr == 0 && immediate == 0;
            default: encoding_valid = 1'b0;
        endcase
        if (!encoding_valid || active_mask != 4'hf ||
            ((writes_matrix || opcode == OP_TWAIT) && !uniform_a) ||
            (writes_matrix && !uniform_b)) command_error = 32'h01;
        else case (opcode)
            OP_TALLOC: begin
                if (allocated[slot] && owner[slot] == warp_id) command_error = 32'h02;
                else if (allocated[slot]) command_ready = 1'b0;
            end
            OP_TFREE, OP_TST, OP_TMMA, OP_TMMA_ACC, OP_TMMA_NVFP4, OP_TMMA_NVFP4_ACC: begin
                if (!own_slot) command_error = 32'h02;
                else if (writes_matrix &&
                         (!tensor_descriptor_valid(source_a[31:0], command_nvfp4) ||
                          !tensor_descriptor_valid(source_b[31:0], command_nvfp4)))
                    command_error = 32'h08;
                else if (writes_matrix && generation[slot] == 32'hffffffff) command_error = 32'h20;
                else if (command_accumulate && (initialized[slot] != 4'hf || failed[slot])) command_error = 32'h04;
                else if (slot_busy || readers[slot] != 0 || unpublished ||
                         (writes_matrix && (busy || !array_available || !memory_idle)))
                    command_ready = 1'b0;
            end
            OP_TCOMMIT: begin
                if (!own_slot) command_error = 32'h02;
                else if (!job_live[slot] || committed[slot]) command_error = 32'h04;
                else command_result = {4{generation[slot]}};
            end
            OP_TWAIT: begin
                if (source_a[31:0] == 0 || source_a[31:0] < generation[slot]) command_error = 32'h10;
                else if (source_a[31:0] > generation[slot]) command_ready = 1'b0;
                else if (!allocated[slot] || !job_live[slot]) command_error = 32'h10;
                else if (!committed[slot]) command_ready = 1'b0;
                else if (!readers[slot][warp_id]) command_error = 32'h02;
                else if (!completed[slot]) command_ready = 1'b0;
            end
            OP_TLD: begin
                if (!allocated[slot]) command_error = 32'h02;
                else if (slot_busy || failed[slot] || !initialized[slot][tile_row] ||
                         (job_live[slot] ? !visible[slot][warp_id] : !own_slot)) command_error = 32'h04;
                else command_result = tiles[slot][tile_row*128 +: 128];
            end
            OP_TRELEASE: begin
                if (!visible[slot][warp_id]) command_error = 32'h02;
            end
            OP_TSTATUS: command_result = {4{errors[warp_id]}};
            default: command_error = 32'h01;
        endcase
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            allocated <= 0; job_live <= 0; committed <= 0; completed <= 0; failed <= 0;
            job_slot <= 0; job_accumulate <= 0; job_nvfp4 <= 0; chunk <= 0;
            packed_a <= 0; packed_b <= 0; scales_a <= 0; scales_b <= 0;
            descriptor_a <= 0; descriptor_b <= 0; fetch_row <= 0;
            a_tile <= 0; b_tile <= 0;
            row_valid <= 0; row_warp_id <= 0; row_rd_addr <= 0;
            row_write_mask <= 0; row_data <= 0;
            for (integer tile = 0; tile < 2; tile = tile + 1) begin
                tiles[tile] <= 0; owner[tile] <= 0; generation[tile] <= 0;
                initialized[tile] <= 0; readers[tile] <= 0; visible[tile] <= 0;
            end
            for (integer warp = 0; warp < 4; warp = warp + 1) errors[warp] <= 0;
        end else begin
            if (row_valid && row_ready) row_valid <= 1'b0;
            case (state)
                FETCH_REQUEST: if (memory_request_ready) state <= FETCH_RESPONSE;
                FETCH_RESPONSE: if (memory_response_valid) begin
                    if (job_nvfp4) begin
                        for (integer lane = 0; lane < 4; lane = lane + 1) begin
                            if (fetch_word == 2) begin
                                if (fetch_b) scales_b[lane*8 +: 8] <= memory_data[lane*32 +: 8];
                                else scales_a[lane*8 +: 8] <= memory_data[lane*32 +: 8];
                            end else if (fetch_b)
                                packed_b[lane*64+int'(fetch_word)*32 +: 32] <= memory_data[lane*32 +: 32];
                            else packed_a[lane*64+int'(fetch_word)*32 +: 32] <= memory_data[lane*32 +: 32];
                        end
                    end else if (fetch_row[2]) b_tile[fetch_row[1:0]*128 +: 128] <= memory_data;
                    else a_tile[fetch_row[1:0]*128 +: 128] <= memory_data;
                    if (scale_failure) begin
                        failed[job_slot] <= 1'b1;
                        completed[job_slot] <= 1'b1;
                        errors[owner[job_slot]] <= errors[owner[job_slot]] | 32'h40;
                        state <= IDLE;
                    end else if (job_nvfp4 && fetch_row == 5) state <= PREPARE;
                    else if (!job_nvfp4 && fetch_row == 7) state <= START;
                    else begin
                        fetch_row <= fetch_row + 1'b1;
                        state <= FETCH_REQUEST;
                    end
                end
                PREPARE: begin
                    // B is stored as four K-vectors (columns); present each
                    // four-wide step to the array in its usual row-major order.
                    for (integer vector_index = 0; vector_index < 4; vector_index = vector_index + 1)
                        for (integer k = 0; k < 4; k = k + 1) begin
                            a_tile[(vector_index*4+k)*32 +: 32] <= {20'd0, scales_a[vector_index*8 +: 8],
                                packed_a[vector_index*64+(int'(chunk)*4+k)*4 +: 4]};
                            b_tile[(k*4+vector_index)*32 +: 32] <= {20'd0, scales_b[vector_index*8 +: 8],
                                packed_b[vector_index*64+(int'(chunk)*4+k)*4 +: 4]};
                        end
                    state <= START;
                end
                START: if (array_request_ready) state <= COMPUTE;
                COMPUTE: if (array_result_valid) begin
                    tiles[job_slot] <= array_result;
                    if (job_nvfp4 && chunk != 3) begin
                        chunk <= chunk + 1'b1;
                        state <= PREPARE;
                    end else begin
                        initialized[job_slot] <= 4'hf;
                        completed[job_slot] <= 1'b1;
                        state <= IDLE;
                    end
                end
                default: begin end
            endcase
            if (command_valid && command_ready) begin
                if (returns_register) begin
                    row_valid <= 1'b1;
                    row_warp_id <= warp_id; row_rd_addr <= rd_addr;
                    row_write_mask <= active_mask;
                    row_data <= command_error == 0 ? command_result : 128'd0;
                end
                if (command_error != 0) errors[warp_id] <= errors[warp_id] | command_error |
                    ((scale_failure && owner[job_slot] == warp_id) ? 32'h40 : 32'd0);
                else case (opcode)
                    OP_TALLOC: begin
                        allocated[slot] <= 1'b1; owner[slot] <= warp_id;
                        initialized[slot] <= 0; job_live[slot] <= 0;
                        committed[slot] <= 0; completed[slot] <= 0;
                        failed[slot] <= 0;
                        readers[slot] <= 0; visible[slot] <= 0;
                    end
                    OP_TFREE: begin
                        allocated[slot] <= 0; initialized[slot] <= 0;
                        job_live[slot] <= 0; committed[slot] <= 0; completed[slot] <= 0;
                        visible[slot] <= 0;
                        failed[slot] <= 0;
                    end
                    OP_TST: begin
                        tiles[slot][tile_row*128 +: 128] <= source_a;
                        initialized[slot][tile_row] <= 1'b1;
                        job_live[slot] <= 0; committed[slot] <= 0; completed[slot] <= 0;
                        visible[slot] <= 0;
                        failed[slot] <= 0;
                    end
                    OP_TMMA, OP_TMMA_ACC, OP_TMMA_NVFP4, OP_TMMA_NVFP4_ACC: begin
                        job_slot <= slot; job_accumulate <= command_accumulate;
                        job_nvfp4 <= command_nvfp4; chunk <= 0; failed[slot] <= 0;
                        descriptor_a <= source_a[31:0]; descriptor_b <= source_b[31:0];
                        fetch_row <= 0; state <= FETCH_REQUEST;
                        generation[slot] <= generation[slot] + 1'b1;
                        job_live[slot] <= 1'b1; committed[slot] <= 0; completed[slot] <= 0;
                        visible[slot] <= 0;
                    end
                    OP_TCOMMIT: begin
                        committed[slot] <= 1'b1;
                        readers[slot] <= participants;
                    end
                    OP_TWAIT: begin
                        visible[slot][warp_id] <= 1'b1;
                        if (failed[slot]) errors[warp_id] <= errors[warp_id] | 32'h40;
                    end
                    OP_TRELEASE: begin
                        visible[slot][warp_id] <= 0;
                        readers[slot][warp_id] <= 0;
                    end
                    default: begin end
                endcase
            end
        end
    end
endmodule
