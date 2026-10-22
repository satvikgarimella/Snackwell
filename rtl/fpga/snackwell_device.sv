// Board-independent host endpoint. All signals use clk; a physical transport
// must cross into this clock domain before driving the request interface.
module snackwell_device (
    input  logic        clk,
    input  logic        reset,
    input  logic        request_valid,
    output logic        request_ready,
    input  logic        request_write,
    input  logic [31:0] request_address,
    input  logic [31:0] request_data,
    output logic        response_valid,
    input  logic        response_ready,
    output logic        response_error,
    output logic [31:0] response_data,
    output logic        busy,
    output logic        done,
    output logic        fault
);
    typedef enum logic [2:0] {IDLE, RESET_START, LAUNCH, RUN, RESET_ABORT} engine_state_t;
    engine_state_t state;
    logic [1:0] launch_warp;
    logic [31:0] entry_pc, cycle_limit, cycles;
    logic [3:0] warp_mask, halted_mask;
    logic [15:0] active_masks;
    logic core_idle, core_reset;
    logic memory_wait, memory_write;
    logic host_memory_ready, host_memory_response_valid;
    logic [31:0] host_memory_read_data, host_program_read_data;
    logic accept, program_access, memory_access, aligned;
    logic host_memory_valid, host_program_write;

    assign busy = state != IDLE;
    assign core_reset = reset || state == RESET_START || state == RESET_ABORT;
    assign aligned = request_address[1:0] == 0;
    assign program_access = request_address[31:12] == 20'h00001;
    assign memory_access = request_address[31:12] == 20'h00002;
    // One outstanding command, including its response. Memory accesses use the
    // same banked memory controller as SM instructions, without a second port.
    assign request_ready = !reset && !response_valid && !memory_wait &&
        (!(memory_access && aligned && !busy) || host_memory_ready);
    assign accept = request_valid && request_ready;
    assign host_memory_valid = accept && aligned && memory_access && !busy;
    assign host_program_write = accept && aligned && program_access && !busy && request_write;

    multiwarp_execution_core #(
        .HOST_PORTS(1), .SIMULATION_INIT(0), .CLEAR_SHARED_ON_RESET(0)
    ) core (
        .clk(clk), .reset(core_reset), .run_enable(state == RUN && cycles < cycle_limit),
        .launch_enable(state == LAUNCH && warp_mask[launch_warp]),
        .launch_warp_id(launch_warp), .launch_pc(entry_pc),
        .launch_active_mask(active_masks[launch_warp*4 +: 4]),
        .debug_write_enable(1'b0), .debug_warp_id(2'd0), .debug_lane_id(2'd0),
        .debug_reg_addr(4'd0), .debug_write_data(32'd0),
        .host_program_write(host_program_write),
        .host_program_address(request_address[11:2]), .host_program_data(request_data),
        .host_program_read_data(host_program_read_data),
        .host_memory_valid(host_memory_valid), .host_memory_ready(host_memory_ready),
        .host_memory_write(request_write), .host_memory_address(request_address[11:2]),
        .host_memory_data(request_data), .host_memory_read_data(host_memory_read_data),
        .host_memory_response_valid(host_memory_response_valid),
        .host_memory_response_ready(memory_wait && !response_valid),
        .execution_idle(core_idle), .halted_mask(halted_mask),
        .issue_valid(), .issue_accept(), .selected_warp_id(), .selected_pc(),
        .instruction(), .opcode(), .selected_active_mask(), .valid_mask(),
        .ready_mask(), .blocked_mask(), .dependency_stall(), .resource_stall(),
        .rs1_data(), .rs2_data(), .alu_result(), .shared_bank_conflict(),
        .shared_conflict_mask(), .branch_divergent(), .branch_taken_mask(),
        .branch_not_taken_mask(), .reconvergence_event()
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            launch_warp <= 0;
            entry_pc <= 0;
            warp_mask <= 4'hf;
            active_masks <= 16'hffff;
            cycle_limit <= 32'd1000000;
            cycles <= 0;
            done <= 0;
            fault <= 0;
            memory_wait <= 0;
            memory_write <= 0;
            response_valid <= 0;
            response_error <= 0;
            response_data <= 0;
        end else begin
            if (response_valid && response_ready) response_valid <= 0;
            if (host_memory_response_valid && memory_wait && !response_valid) begin
                memory_wait <= 0;
                response_valid <= 1;
                response_error <= 0;
                response_data <= memory_write ? 32'd0 : host_memory_read_data;
            end

            case (state)
                RESET_START: begin
                    launch_warp <= 0;
                    state <= LAUNCH;
                end
                LAUNCH: begin
                    if (launch_warp == 3) state <= RUN;
                    else launch_warp <= launch_warp + 1'b1;
                end
                RUN: begin
                    if ((halted_mask & warp_mask) == warp_mask && core_idle) begin
                        done <= 1;
                        state <= IDLE;
                    end else if (cycles == cycle_limit) begin
                        fault <= 1;
                        state <= RESET_ABORT;
                    end else cycles <= cycles + 1'b1;
                end
                RESET_ABORT: state <= IDLE;
                default: begin end
            endcase

            if (accept) begin
                response_valid <= 1;
                response_error <= 0;
                response_data <= 0;
                if (!aligned) response_error <= 1;
                else if (program_access || memory_access) begin
                    if (busy) response_error <= 1;
                    else if (program_access) begin
                        if (!request_write) response_data <= host_program_read_data;
                    end else begin
                        response_valid <= 0;
                        memory_wait <= 1;
                        memory_write <= request_write;
                    end
                end else if (request_write) begin
                    // Abort remains available during execution. Configuration
                    // cannot change beneath a running kernel.
                    if (request_address == 32'h08 && request_data == 2) begin
                        state <= RESET_ABORT;
                        done <= 0;
                        fault <= 1;
                    end else if (busy) response_error <= 1;
                    else case (request_address)
                        32'h08: begin
                            if (request_data != 1 || warp_mask == 0)
                                response_error <= 1;
                            else begin
                                state <= RESET_START;
                                cycles <= 0;
                                done <= 0;
                                fault <= 0;
                            end
                        end
                        32'h0c: begin
                            if (request_data[1:0] != 0 || request_data >= 4096)
                                response_error <= 1;
                            else entry_pc <= request_data;
                        end
                        32'h10: begin
                            if (request_data == 0 || request_data > 15) response_error <= 1;
                            else warp_mask <= request_data[3:0];
                        end
                        32'h14: begin
                            if (request_data[31:16] != 0) response_error <= 1;
                            else active_masks <= request_data[15:0];
                        end
                        32'h18: begin
                            if (request_data == 0) response_error <= 1;
                            else cycle_limit <= request_data;
                        end
                        default: response_error <= 1;
                    endcase
                end else case (request_address)
                    32'h00: response_data <= 32'h534e4b31; // SNK1: interface version 1
                    32'h04: response_data <= {20'd0, halted_mask, 5'd0, fault, done, busy};
                    32'h0c: response_data <= entry_pc;
                    32'h10: response_data <= {28'd0, warp_mask};
                    32'h14: response_data <= {16'd0, active_masks};
                    32'h18: response_data <= cycle_limit;
                    32'h1c: response_data <= cycles;
                    default: response_error <= 1;
                endcase
            end
        end
    end
endmodule
