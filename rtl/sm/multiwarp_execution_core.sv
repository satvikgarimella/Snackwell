module multiwarp_execution_core #(
    parameter bit HOST_PORTS = 0,
    parameter bit SIMULATION_INIT = 1,
    parameter bit CLEAR_SHARED_ON_RESET = 1
) (
    input logic clk,
    input logic reset,
    input logic run_enable,
    // Host access is accepted only while execution is stopped and drained.
    input  logic        host_program_write,
    input  logic [9:0]  host_program_address,
    input  logic [31:0] host_program_data,
    output logic [31:0] host_program_read_data,
    input  logic        host_memory_valid,
    output logic        host_memory_ready,
    input  logic        host_memory_write,
    input  logic [9:0]  host_memory_address,
    input  logic [31:0] host_memory_data,
    output logic        host_memory_response_valid,
    input  logic        host_memory_response_ready,
    output logic [31:0] host_memory_read_data,
    output logic        execution_idle,
    // External word-memory master, independent of shared memory and host MMIO.
    output logic        global_request_valid,
    input  logic        global_request_ready,
    output logic        global_request_write,
    output logic [31:0] global_request_address, global_request_data,
    input  logic        global_response_valid,
    output logic        global_response_ready,
    input  logic        global_response_error,
    input  logic [31:0] global_response_data,
    output logic global_idle,
    output logic [11:0] global_status,
    output logic [31:0] copy_status,
    // WARP LAUNCH

    input logic        launch_enable,
    input logic [1:0]  launch_warp_id,
    input logic [31:0] launch_pc,
    input logic [3:0]  launch_active_mask,
    // DEBUG REGISTER PRELOAD

    input logic        debug_write_enable,
    input logic [1:0]  debug_warp_id,
    input logic [1:0]  debug_lane_id,
    input logic [3:0]  debug_reg_addr,
    input logic [31:0] debug_write_data,
    // STATUS

    output logic       issue_valid,
    output logic       issue_accept,
    output logic [1:0] selected_warp_id,

    output logic [31:0] selected_pc,
    output logic [31:0] instruction,
    output logic [5:0]  opcode,

    output logic [3:0] selected_active_mask,

    output logic [3:0] valid_mask,
    output logic [3:0] ready_mask,
    output logic [3:0] halted_mask,
    output logic [3:0] blocked_mask,

    output logic dependency_stall,
    output logic resource_stall,

    output logic [127:0] rs1_data,
    output logic [127:0] rs2_data,
    output logic [127:0] alu_result,

    // Shared memory
    output logic       shared_bank_conflict,
    output logic [3:0] shared_conflict_mask,

    // Control-flow debug
    output logic       branch_divergent,
    output logic [3:0] branch_taken_mask,
    output logic [3:0] branch_not_taken_mask,

    output logic       reconvergence_event
);

    import snackwell_pkg::*;
    // DECODE

    logic [3:0] rd_addr;
    logic [3:0] rs1_addr;
    logic [3:0] rs2_addr;

    logic [13:0] immediate;
    logic [31:0] immediate_extended;
    // OPCODE CLASSIFICATION

    logic is_integer_op;
    logic is_fp_op;
    logic is_tensor_op;
    logic is_tmem_op;
    logic is_global_op;
    logic is_copy_op;
    logic is_shared_load;
    logic is_shared_store;
    logic is_branch;
    logic is_ssy;

    logic uses_rs1;
    logic uses_rs2;
    logic writes_rd;

    logic issue_uses_rs1;
    logic issue_uses_rs2;
    logic issue_writes_rd;
    // STATE UPDATE

    logic        state_update_enable;
    logic [1:0]  state_update_warp_id;

    logic        update_pc_enable;
    logic [31:0] update_pc;

    logic        update_mask_enable;
    logic [3:0]  update_active_mask;

    logic update_valid_enable;
    logic update_valid;

    logic update_ready_enable;
    logic update_ready;

    logic update_halted_enable;
    logic update_halted;
    // SELECTED WARP

    logic selected_valid;
    logic selected_ready;
    logic selected_halted;

    logic [3:0] eligible_mask;
    logic [15:0] selected_warp_busy_mask;
    // EXECUTION

    logic execution_unit_ready;
    logic at_reconvergence_point;

    logic [127:0] current_alu_result;
    logic [127:0] integer_alu_result, integer_operand_b;
    // LSU

    logic [127:0] lsu_addresses;
    logic [127:0] lsu_store_data;

    logic [3:0] lsu_load_mask;
    logic [3:0] lsu_store_mask;

    logic [127:0] shared_load_data;

    // One shared-memory instruction remains in flight until all lanes finish
    // and its completion is accepted. Operand snapshots live in shared_mem.
    logic shared_start, shared_request_ready, shared_response_valid, shared_response_ready;
    logic shared_inflight, shared_is_load, shared_completion_valid, shared_take;
    logic [1:0] shared_warp_id;
    logic [3:0] shared_rd_addr, shared_write_mask;
    logic memory_response_valid, memory_response_ready;
    logic host_memory_pending, host_memory_accept;
    logic global_start, global_ready, global_busy, global_completion_valid;
    logic global_completion_ready, global_writes_rd, global_take;
    logic [1:0] global_warp;
    logic [3:0] global_rd, global_mask;
    logic [127:0] global_data;
    // WRITEBACK

    logic         wb_valid;
    logic [1:0]   wb_warp_id;
    logic [3:0]   wb_rd_addr;
    logic [3:0]   wb_write_mask;
    logic [127:0] wb_result;

    // A single shared FP unit retains its operands internally. Its completion
    // tag is independent of whichever warp the scheduler subsequently selects.
    logic fp_start, fp_busy, fp_done, fp_inflight, fp_result_pending;
    logic [127:0] fp_result;
    logic [1:0] fp_warp_id;
    logic [3:0] fp_rd_addr, fp_write_mask;
    logic fp_completion_valid, fp_take;
    logic wb_fire, wb_slot_available;

    logic tensor_start, tensor_ready, tensor_busy, tensor_gather;
    logic [1:0] tensor_warp_id;
    logic [3:0] tensor_a_addr, tensor_b_addr;
    logic [15:0] tensor_destination_mask, selected_tensor_busy_mask;
    logic [15:0] tensor_operand_mask;
    logic tensor_dependency;
    logic tensor_row_valid, tensor_take, tensor_row_last;
    logic [3:0] tensor_rd_addr, tensor_write_mask;
    logic [127:0] tensor_result;
    logic wb_is_tensor, wb_tensor_last;
    logic [1:0] rf_read_warp_id;
    logic [3:0] rf_rs1_addr, rf_rs2_addr;

    logic tmem_start, tmem_ready, tmem_busy, tmem_memory_lock;
    logic tmem_nvfp4_mode, tmem_matrix_guard_valid;
    logic [3:0] tmem_warp_pending;
    logic tmem_memory_valid, tmem_memory_response_ready, tmem_memory_pending;
    logic [127:0] tmem_memory_addresses;
    logic tmem_array_valid, tmem_array_ready, tmem_array_result_valid, tmem_array_result_ready;
    logic [511:0] tmem_a, tmem_b, tmem_c, tmem_array_result;
    logic tmem_row_valid, tmem_take;
    logic [1:0] tmem_row_warp;
    logic [3:0] tmem_row_rd, tmem_row_mask;
    logic [127:0] tmem_row_data;

    logic copy_start, copy_ready, copy_busy, copy_shared_access, copy_matrix_access;
    logic [3:0] copy_warp_pending;
    logic copy_row_valid, copy_take;
    logic [1:0] copy_row_warp;
    logic [3:0] copy_row_rd, copy_row_mask;
    logic [127:0] copy_row_data;
    logic copy_shared_valid, copy_shared_ready, copy_shared_pending, copy_shared_response_ready;
    logic [31:0] copy_shared_address, copy_shared_data;
    logic copy_global_valid, copy_global_ready, copy_global_response_valid;
    logic copy_global_response_ready, copy_global_error;
    logic [31:0] copy_global_address, copy_global_data;
    logic lsu_global_valid, lsu_global_ready, lsu_global_write, lsu_global_response_valid;
    logic lsu_global_response_ready, lsu_global_error, global_bus_busy;
    logic [31:0] lsu_global_address, lsu_global_data, lsu_global_result;

    assign wb_fire = wb_valid && !debug_write_enable;
    assign wb_slot_available = !wb_valid || wb_fire;
    assign fp_completion_valid = fp_inflight && (fp_done || fp_result_pending);
    assign fp_take = fp_completion_valid && wb_slot_available;
    assign tensor_take = tensor_row_valid && wb_slot_available && !fp_completion_valid;
    assign tmem_take = tmem_row_valid && wb_slot_available &&
                       !fp_completion_valid && !tensor_row_valid;
    assign copy_take = copy_row_valid && wb_slot_available &&
                       !fp_completion_valid && !tensor_row_valid && !tmem_row_valid;
    assign shared_completion_valid = shared_inflight && shared_is_load && shared_response_valid;
    assign shared_response_ready = !shared_is_load ||
        (wb_slot_available && !fp_completion_valid && !tensor_row_valid && !tmem_row_valid && !copy_row_valid);
    assign shared_take = shared_completion_valid && shared_response_ready;

    assign global_completion_ready = !global_writes_rd ||
        (wb_slot_available && !fp_completion_valid && !tensor_row_valid &&
         !tmem_row_valid && !copy_row_valid && !shared_completion_valid);
    assign global_take = global_completion_valid && global_writes_rd && global_completion_ready;
    assign global_idle = !global_busy && !copy_busy && !global_bus_busy;

    assign selected_tensor_busy_mask = (selected_warp_id == tensor_warp_id)
                                      ? tensor_destination_mask : 16'd0;
    assign tensor_operand_mask = (16'h000f << rs1_addr) |
                                 (16'h000f << rs2_addr) | (16'h000f << rd_addr);
    // The scalar scoreboard retains ordinary destinations; this reservation
    // covers all four tensor rows until the final architectural write commits.
    assign tensor_dependency = !at_reconvergence_point &&
        ((is_tensor_op && |(tensor_operand_mask &
                          (selected_warp_busy_mask | selected_tensor_busy_mask))) ||
         (uses_rs1 && selected_tensor_busy_mask[rs1_addr]) ||
         (uses_rs2 && selected_tensor_busy_mask[rs2_addr]) ||
         (writes_rd && selected_tensor_busy_mask[rd_addr]));

    assign rf_read_warp_id = tensor_gather ? tensor_warp_id : selected_warp_id;
    assign rf_rs1_addr = tensor_gather ? tensor_a_addr : rs1_addr;
    assign rf_rs2_addr = tensor_gather ? tensor_b_addr : rs2_addr;
    // BRANCH UNIT

    logic [3:0] branch_taken;
    logic [3:0] branch_not_taken;

    logic branch_uniform_taken;
    logic branch_uniform_not_taken;

    logic [31:0] branch_target;
    logic [31:0] branch_fallthrough;

    assign branch_taken_mask =
        branch_taken;

    assign branch_not_taken_mask =
        branch_not_taken;
    // RECONVERGENCE TARGET TABLE

    logic [31:0] reconv_target_pc;
    logic        reconv_target_valid;

    logic [31:0] ssy_target_pc;

    logic target_write_enable;
    logic target_clear_enable;

    assign ssy_target_pc =
        selected_pc +
        32'd4 +
        (immediate_extended << 2);
    // RECONVERGENCE STACK

    logic [31:0] stack_deferred_pc;
    logic [3:0]  stack_deferred_mask;

    logic [31:0] stack_reconverge_pc;
    logic [3:0]  stack_reconverge_mask;

    logic stack_deferred_pending;

    logic stack_empty;
    logic stack_full;
    logic [2:0] stack_depth;

    logic stack_overflow_attempt;
    logic stack_underflow_attempt;
    // DIVERGENCE CONTROLLER COMMANDS

    logic        divergence_pc_update_enable;
    logic [31:0] divergence_next_pc;

    logic        divergence_mask_update_enable;
    logic [3:0]  divergence_next_mask;

    logic stack_push_enable;

    logic [31:0] stack_push_deferred_pc;
    logic [3:0]  stack_push_deferred_mask;

    logic [31:0] stack_push_reconverge_pc;
    logic [3:0]  stack_push_reconverge_mask;

    logic stack_pop_enable;

    logic stack_update_pending_enable;
    logic stack_update_deferred_pending;

    logic switch_to_deferred_path;
    logic restore_reconverged_mask;

    logic missing_reconvergence_target;

    // Current PC is a reconvergence point.
    assign at_reconvergence_point =
        !stack_empty &&
        (selected_pc == stack_reconverge_pc);

    // This event acts like an internal control-flow micro-op.
    // We do NOT execute the instruction currently sitting at
    // the JOIN address until reconvergence handling finishes.

    logic instruction_execute_accept;

    assign fp_start = instruction_execute_accept && is_fp_op;
    assign tensor_start = instruction_execute_accept && is_tensor_op;
    assign tmem_start = instruction_execute_accept && is_tmem_op;
    assign copy_start = instruction_execute_accept && is_copy_op;
    assign global_start = instruction_execute_accept && is_global_op;
    assign shared_start = instruction_execute_accept && (is_shared_load || is_shared_store);

    assign instruction_execute_accept =
        issue_accept &&
        !at_reconvergence_point;
    // INSTRUCTION MEMORY

    assign execution_idle = !fp_inflight && !tensor_busy && !shared_inflight &&
        !wb_valid && !host_memory_pending && !global_busy && !tmem_busy && !tmem_row_valid &&
        tmem_warp_pending == 0 && !copy_busy && !copy_row_valid && copy_warp_pending == 0;
    assign host_memory_ready = HOST_PORTS && !reset && !run_enable &&
        !launch_enable && execution_idle && shared_request_ready;
    assign host_memory_accept = host_memory_valid && host_memory_ready;
    assign host_memory_response_valid = host_memory_pending && memory_response_valid;
    assign host_memory_read_data = shared_load_data[31:0];
    assign shared_response_valid = memory_response_valid && !host_memory_pending && !tmem_memory_pending && !copy_shared_pending;
    assign memory_response_ready = host_memory_pending
        ? host_memory_response_ready : tmem_memory_pending
        ? tmem_memory_response_ready : copy_shared_pending
        ? copy_shared_response_ready : shared_response_ready;

    always_ff @(posedge clk) begin
        if (reset)
            host_memory_pending <= 1'b0;
        else begin
            if (host_memory_accept) host_memory_pending <= 1'b1;
            if (host_memory_response_valid && host_memory_response_ready)
                host_memory_pending <= 1'b0;
        end
    end

    instruction_memory #(
        .HOST_WRITABLE(HOST_PORTS), .SIMULATION_INIT(SIMULATION_INIT)
    ) imem (

        .pc          (selected_pc),
        .instruction (instruction),
        .clk(clk),
        .host_write_enable(HOST_PORTS && !reset && !run_enable && !launch_enable &&
                           execution_idle && host_program_write),
        .host_word_address(host_program_address),
        .host_write_data(host_program_data),
        .host_read_data(host_program_read_data)

    );
    // DECODER

    instruction_decoder decoder (

        .instruction        (instruction),

        .opcode             (opcode),

        .rd_addr            (rd_addr),
        .rs1_addr           (rs1_addr),
        .rs2_addr           (rs2_addr),

        .immediate          (immediate),
        .immediate_extended (immediate_extended)

    );
    // OPCODE CONTROL

    always @(*) begin

        is_integer_op   = 1'b0;
        is_fp_op        = 1'b0;
        is_tensor_op    = 1'b0;
        is_tmem_op      = 1'b0;
        is_global_op    = 1'b0;
        is_copy_op      = 1'b0;
        is_shared_load  = 1'b0;
        is_shared_store = 1'b0;
        is_branch       = 1'b0;
        is_ssy          = 1'b0;

        uses_rs1  = 1'b0;
        uses_rs2  = 1'b0;
        writes_rd = 1'b0;

        case (opcode)

            OP_ADD,
            OP_SUB,
            OP_AND,
            OP_OR,
            OP_XOR,
            OP_SHR,
            OP_SLTU,
            OP_FLT: begin

                is_integer_op = 1'b1;

                uses_rs1  = 1'b1;
                uses_rs2  = 1'b1;
                writes_rd = 1'b1;

            end

            OP_ADDI: begin
                is_integer_op = 1'b1;
                uses_rs1 = 1'b1;
                writes_rd = 1'b1;
            end

            OP_TID: begin
                is_integer_op = 1'b1;
                writes_rd = 1'b1;
            end

            OP_FMUL,
            OP_FADD,
            OP_FSUB: begin
                is_fp_op = 1'b1;
                uses_rs1 = 1'b1;
                uses_rs2 = 1'b1;
                writes_rd = 1'b1;
            end

            OP_TMUL: begin
                // Invalid tile bases/reserved bits act as NOP, matching the
                // SM's current unsupported-encoding behavior. Never wrap rows.
                is_tensor_op = rs1_addr <= 4'd12 && rs2_addr <= 4'd12 &&
                               rd_addr <= 4'd12 && immediate == 14'd0;
                // All source and destination rows are checked together by
                // tensor_dependency, and allocated by sm_tensor_unit.
            end

            OP_TALLOC, OP_TFREE, OP_TST, OP_TLD, OP_TMMA, OP_TMMA_ACC, OP_TMMA_NVFP4, OP_TMMA_NVFP4_ACC,
            OP_TCOMMIT, OP_TWAIT, OP_TRELEASE, OP_TSTATUS: begin
                is_tmem_op = 1'b1;
                uses_rs1 = opcode == OP_TST || tensor_matrix_opcode(opcode) || opcode == OP_TWAIT;
                uses_rs2 = tensor_matrix_opcode(opcode);
                writes_rd = opcode == OP_TLD || opcode == OP_TCOMMIT || opcode == OP_TSTATUS;
            end

            OP_CPASYNC, OP_CPWAIT, OP_CPRELEASE, OP_CPSTATUS: begin
                is_copy_op = 1'b1;
                uses_rs1 = opcode == OP_CPASYNC || opcode == OP_CPWAIT;
                writes_rd = opcode == OP_CPASYNC || opcode == OP_CPSTATUS;
            end

            OP_LD_GLOBAL, OP_ST_GLOBAL, OP_GSTATUS: begin
                is_global_op = 1'b1;
                uses_rs1 = opcode != OP_GSTATUS;
                uses_rs2 = opcode == OP_ST_GLOBAL;
                writes_rd = opcode != OP_ST_GLOBAL;
            end

            OP_LD_SHARED: begin

                is_shared_load = 1'b1;

                uses_rs1  = 1'b1;
                writes_rd = 1'b1;

            end

            OP_ST_SHARED: begin

                is_shared_store = 1'b1;

                uses_rs1 = 1'b1;
                uses_rs2 = 1'b1;

            end

            OP_BEQ: begin

                is_branch = 1'b1;

                uses_rs1 = 1'b1;
                uses_rs2 = 1'b1;

            end

            OP_SSY: begin

                is_ssy = 1'b1;

            end

            OP_EXIT: begin
            end

            default: begin
            end

        endcase

    end

    // At a reconvergence PC, suppress architectural
    // operands/destination for the fetched instruction.
    //
    // This prevents the scoreboard from treating the JOIN
    // instruction as having executed while the controller
    // is switching masks/paths.

    assign issue_uses_rs1 =
        at_reconvergence_point
            ? 1'b0
            : uses_rs1;

    assign issue_uses_rs2 =
        at_reconvergence_point
            ? 1'b0
            : uses_rs2;

    assign issue_writes_rd =
        at_reconvergence_point
            ? 1'b0
            : writes_rd;

    assign execution_unit_ready =
        run_enable && !launch_enable && !debug_write_enable && !tensor_gather &&
        (at_reconvergence_point ||
         ((!is_fp_op || !fp_inflight) &&
          (!is_tensor_op || tensor_ready) &&
          (!is_tmem_op || (tmem_ready &&
            (!tensor_matrix_opcode(opcode) || copy_matrix_access))) &&
          (!is_copy_op || copy_ready) &&
          (!is_global_op || global_ready) &&
          (!(is_shared_load || is_shared_store) || (shared_request_ready && !tmem_memory_lock &&
            !copy_shared_valid && copy_shared_access)) &&
          (!is_integer_op ||
           (wb_slot_available && !fp_completion_valid && !tensor_row_valid &&
            !tmem_row_valid && !copy_row_valid && !shared_completion_valid &&
            !(global_completion_valid && global_writes_rd))) &&
          (opcode != OP_EXIT ||
           (selected_warp_busy_mask == 16'd0 && selected_tensor_busy_mask == 16'd0 &&
            !(shared_inflight && shared_warp_id == selected_warp_id) &&
            !(global_busy && global_warp == selected_warp_id) &&
            !tmem_warp_pending[selected_warp_id] && !copy_warp_pending[selected_warp_id]))));
    // WARP CONTROL

    warp_control_unit warp_control (

        .clk                  (clk),
        .reset                (reset),

        .issue_enable         (run_enable && !launch_enable && !debug_write_enable),

        .update_enable        (state_update_enable),
        .update_warp_id       (state_update_warp_id),

        .update_pc_enable     (update_pc_enable),
        .update_pc            (update_pc),

        .update_mask_enable   (update_mask_enable),
        .update_active_mask   (update_active_mask),

        .update_valid_enable  (update_valid_enable),
        .update_valid         (update_valid),

        .update_ready_enable  (update_ready_enable),
        .update_ready         (update_ready),

        .update_halted_enable (update_halted_enable),
        .update_halted        (update_halted),

        .rs1_addr             (rs1_addr),
        .rs2_addr             (rs2_addr),
        .rd_addr              (rd_addr),

        .uses_rs1             (issue_uses_rs1),
        .uses_rs2             (issue_uses_rs2),
        .writes_rd            (issue_writes_rd),
        .additional_dependency (tensor_dependency),

        .execution_unit_ready (execution_unit_ready),

        .clear_busy_enable    (wb_fire),
        .clear_busy_warp_id   (wb_warp_id),
        .clear_busy_reg       (wb_rd_addr),

        .issue_valid          (issue_valid),
        .issue_accept         (issue_accept),
        .selected_warp_id     (selected_warp_id),

        .dependency_stall     (dependency_stall),
        .resource_stall       (resource_stall),

        .selected_pc          (selected_pc),
        .selected_active_mask (selected_active_mask),

        .selected_valid       (selected_valid),
        .selected_ready       (selected_ready),
        .selected_halted      (selected_halted),

        .valid_mask           (valid_mask),
        .ready_mask           (ready_mask),
        .halted_mask          (halted_mask),

        .blocked_mask         (blocked_mask),
        .eligible_mask        (eligible_mask),
        .selected_warp_busy_mask (selected_warp_busy_mask)

    );
    // REGISTER FILE

    multiwarp_register_file regfile (

        .clk                (clk),
        .reset              (reset),

        .read_warp_id       (rf_read_warp_id),

        .rs1_addr           (rf_rs1_addr),
        .rs2_addr           (rf_rs2_addr),

        .rs1_data           (rs1_data),
        .rs2_data           (rs2_data),

        .write_enable       (wb_fire),
        .write_warp_id      (wb_warp_id),

        .rd_addr            (wb_rd_addr),
        .write_data         (wb_result),
        .write_mask         (wb_write_mask),

        .debug_write_enable (debug_write_enable),
        .debug_warp_id      (debug_warp_id),
        .debug_lane_id      (debug_lane_id),
        .debug_reg_addr     (debug_reg_addr),
        .debug_write_data   (debug_write_data)

    );
    // INTEGER SIMD

    simd_integer_execute execute (

        .opcode    (opcode),

        .operand_a (rs1_data),
        .operand_b (integer_operand_b),

        .result    (integer_alu_result)

    );

    assign integer_operand_b = opcode == OP_ADDI ? {4{immediate_extended}} : rs2_data;
    // The SM currently has four resident warps of four lanes. TID supplies
    // a local thread index, independent of register contents or active mask.
    assign current_alu_result = opcode == OP_TID ?
        {28'd0, selected_warp_id, 2'd3, 28'd0, selected_warp_id, 2'd2,
         28'd0, selected_warp_id, 2'd1, 28'd0, selected_warp_id, 2'd0} : integer_alu_result;

    assign alu_result =
        current_alu_result;

    simd_fp32_execute fp_execute (
        .clk(clk), .rst(reset), .start(fp_start), .opcode(opcode),
        .operand_a(rs1_data), .operand_b(rs2_data),
        .active_mask(selected_active_mask),
        .busy(fp_busy), .done(fp_done), .result(fp_result)
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            fp_inflight <= 1'b0;
            fp_result_pending <= 1'b0;
            fp_warp_id <= 2'd0;
            fp_rd_addr <= 4'd0;
            fp_write_mask <= 4'd0;
        end else begin
            if (fp_start) begin
                fp_inflight <= 1'b1;
                fp_warp_id <= selected_warp_id;
                fp_rd_addr <= rd_addr;
                fp_write_mask <= selected_active_mask;
            end
            if (fp_done) fp_result_pending <= 1'b1;
            if (fp_take) begin
                fp_inflight <= 1'b0;
                fp_result_pending <= 1'b0;
            end
        end
    end
    // TENSOR EXECUTION AND REGISTER TRANSFERS

    sm_tensor_unit #(.AUX_PORT(1)) tensor_execute (
        .clk(clk), .reset(reset),
        .request_valid(tensor_start), .request_ready(tensor_ready),
        .request_warp_id(selected_warp_id),
        .request_a_base(rs1_addr), .request_b_base(rs2_addr),
        .request_d_base(rd_addr), .request_write_mask(selected_active_mask),
        .gather_active(tensor_gather), .warp_id(tensor_warp_id),
        .read_a_addr(tensor_a_addr), .read_b_addr(tensor_b_addr),
        .read_a_data(rs1_data), .read_b_data(rs2_data),
        .busy(tensor_busy), .destination_mask(tensor_destination_mask),
        .row_valid(tensor_row_valid), .row_ready(tensor_take),
        .row_rd_addr(tensor_rd_addr), .row_write_mask(tensor_write_mask),
        .row_data(tensor_result), .row_last(tensor_row_last),
        .last_row_committed(wb_fire && wb_is_tensor && wb_tensor_last),
        .aux_busy(tmem_busy), .aux_request_valid(tmem_array_valid),
        .aux_nvfp4_mode(tmem_nvfp4_mode),
        .aux_request_ready(tmem_array_ready),
        .aux_a_tile(tmem_a), .aux_b_tile(tmem_b), .aux_c_tile(tmem_c),
        .aux_result_valid(tmem_array_result_valid), .aux_result_ready(tmem_array_result_ready),
        .aux_result_tile(tmem_array_result)
    );

    sm_tensor_memory_unit tensor_memory (
        .clk(clk), .reset(reset), .command_valid(tmem_start), .command_ready(tmem_ready),
        .opcode(opcode), .warp_id(selected_warp_id), .active_mask(selected_active_mask),
        .rd_addr(rd_addr), .rs1_addr(rs1_addr), .rs2_addr(rs2_addr), .immediate(immediate),
        .source_a(rs1_data), .source_b(rs2_data), .array_available(!tensor_busy),
        .memory_idle(shared_request_ready && !shared_inflight && !host_memory_pending &&
                     !copy_shared_valid && !copy_shared_pending),
        .busy(tmem_busy), .memory_lock(tmem_memory_lock), .warp_pending(tmem_warp_pending),
        .matrix_guard_valid(tmem_matrix_guard_valid),
        .memory_request_valid(tmem_memory_valid), .memory_request_ready(shared_request_ready),
        .memory_addresses(tmem_memory_addresses),
        .memory_response_valid(memory_response_valid && tmem_memory_pending),
        .memory_response_ready(tmem_memory_response_ready), .memory_data(shared_load_data),
        .array_request_valid(tmem_array_valid), .array_request_ready(tmem_array_ready),
        .array_nvfp4_mode(tmem_nvfp4_mode),
        .a_tile(tmem_a), .b_tile(tmem_b), .c_tile(tmem_c),
        .array_result_valid(tmem_array_result_valid), .array_result_ready(tmem_array_result_ready),
        .array_result(tmem_array_result),
        .row_valid(tmem_row_valid), .row_ready(tmem_take), .row_warp_id(tmem_row_warp),
        .row_rd_addr(tmem_row_rd), .row_write_mask(tmem_row_mask), .row_data(tmem_row_data)
    );

    always_ff @(posedge clk) begin
        if (reset) tmem_memory_pending <= 1'b0;
        else begin
            if (tmem_memory_valid && shared_request_ready) tmem_memory_pending <= 1'b1;
            if (memory_response_valid && tmem_memory_pending && tmem_memory_response_ready)
                tmem_memory_pending <= 1'b0;
        end
    end
    // BRANCH UNIT

    branch_unit branch (

        .current_pc        (selected_pc),

        .operand_a         (rs1_data),
        .operand_b         (rs2_data),

        .active_mask       (selected_active_mask),

        .immediate         (immediate_extended),

        .taken_mask        (branch_taken),
        .not_taken_mask    (branch_not_taken),

        .uniform_taken     (branch_uniform_taken),
        .uniform_not_taken (branch_uniform_not_taken),
        .divergent         (branch_divergent),

        .branch_target     (branch_target),
        .fallthrough_pc    (branch_fallthrough)

    );
    // RECONVERGENCE TARGET TABLE

    assign target_write_enable =
        run_enable &&
        instruction_execute_accept &&
        is_ssy;

    assign target_clear_enable =
        run_enable &&
        instruction_execute_accept &&
        is_branch;

    reconvergence_target_table reconv_targets (

        .clk           (clk),
        .reset         (reset),

        .write_enable  (target_write_enable),
        .write_warp_id (selected_warp_id),
        .write_pc      (ssy_target_pc),

        .clear_enable  (target_clear_enable),
        .clear_warp_id (selected_warp_id),

        .read_warp_id  (selected_warp_id),

        .read_pc       (reconv_target_pc),
        .read_valid    (reconv_target_valid)

    );
    // RECONVERGENCE STACK

    reconvergence_stack reconv_stack (

        .clk                     (clk),
        .reset                   (reset),

        .warp_id                 (selected_warp_id),

        .push_enable             (stack_push_enable && issue_accept),

        .push_deferred_pc        (stack_push_deferred_pc),
        .push_deferred_mask      (stack_push_deferred_mask),

        .push_reconverge_pc      (stack_push_reconverge_pc),
        .push_reconverge_mask    (stack_push_reconverge_mask),

        .pop_enable              (stack_pop_enable && issue_accept),

        .update_pending_enable   (stack_update_pending_enable && issue_accept),
        .update_deferred_pending (stack_update_deferred_pending),

        .top_deferred_pc         (stack_deferred_pc),
        .top_deferred_mask       (stack_deferred_mask),

        .top_reconverge_pc       (stack_reconverge_pc),
        .top_reconverge_mask     (stack_reconverge_mask),

        .top_deferred_pending    (stack_deferred_pending),

        .empty                   (stack_empty),
        .full                    (stack_full),
        .depth                   (stack_depth),

        .overflow_attempt        (stack_overflow_attempt),
        .underflow_attempt       (stack_underflow_attempt)

    );
    // DIVERGENCE CONTROLLER

    divergence_controller divergence_control (

        .current_pc                    (selected_pc),
        .current_active_mask           (selected_active_mask),

        .branch_valid                  (
            run_enable &&
            instruction_execute_accept &&
            is_branch
        ),

        .taken_mask                    (branch_taken),
        .not_taken_mask                (branch_not_taken),

        .uniform_taken                 (branch_uniform_taken),
        .uniform_not_taken             (branch_uniform_not_taken),
        .divergent                     (branch_divergent),

        .branch_target                 (branch_target),
        .fallthrough_pc                (branch_fallthrough),

        .reconvergence_target_valid    (reconv_target_valid),
        .reconvergence_target_pc       (reconv_target_pc),

        .stack_empty                   (stack_empty),

        .stack_deferred_pc             (stack_deferred_pc),
        .stack_deferred_mask           (stack_deferred_mask),

        .stack_reconverge_pc           (stack_reconverge_pc),
        .stack_reconverge_mask         (stack_reconverge_mask),

        .stack_deferred_pending        (stack_deferred_pending),

        .pc_update_enable              (divergence_pc_update_enable),
        .next_pc                       (divergence_next_pc),

        .mask_update_enable            (divergence_mask_update_enable),
        .next_active_mask              (divergence_next_mask),

        .stack_push_enable             (stack_push_enable),

        .stack_push_deferred_pc        (stack_push_deferred_pc),
        .stack_push_deferred_mask      (stack_push_deferred_mask),

        .stack_push_reconverge_pc      (stack_push_reconverge_pc),
        .stack_push_reconverge_mask    (stack_push_reconverge_mask),

        .stack_pop_enable              (stack_pop_enable),

        .stack_update_pending_enable   (stack_update_pending_enable),
        .stack_update_deferred_pending (stack_update_deferred_pending),

        .reconvergence_event           (reconvergence_event),
        .switch_to_deferred_path       (switch_to_deferred_path),
        .restore_reconverged_mask      (restore_reconverged_mask),

        .missing_reconvergence_target  (missing_reconvergence_target)

    );
    // LSU

    load_store_unit lsu (

        .base_addresses    (rs1_data),
        .offset            (immediate_extended),

        .store_data        (rs2_data),

        .active_mask       (selected_active_mask),

        .is_load           (
            run_enable &&
            instruction_execute_accept &&
            is_shared_load
        ),

        .is_store          (
            run_enable &&
            instruction_execute_accept &&
            is_shared_store
        ),

        .lane_addresses    (lsu_addresses),
        .store_write_data  (lsu_store_data),

        .load_enable_mask  (lsu_load_mask),
        .store_enable_mask (lsu_store_mask)

    );
    global_memory_unit global_memory (
        .clk(clk), .reset(reset), .command_valid(global_start), .command_ready(global_ready),
        .command_store(opcode == snackwell_pkg::OP_ST_GLOBAL),
        .command_status(opcode == snackwell_pkg::OP_GSTATUS),
        .command_warp(selected_warp_id), .command_rd(rd_addr), .command_mask(selected_active_mask),
        .command_bases(rs1_data), .command_data(rs2_data), .command_offset(immediate_extended),
        .busy(global_busy), .warp_id(global_warp), .status_bits(global_status),
        .request_valid(lsu_global_valid), .request_ready(lsu_global_ready),
        .request_write(lsu_global_write), .request_address(lsu_global_address),
        .request_data(lsu_global_data), .response_valid(lsu_global_response_valid),
        .response_ready(lsu_global_response_ready), .response_error(lsu_global_error),
        .response_data(lsu_global_result), .completion_valid(global_completion_valid),
        .completion_ready(global_completion_ready), .completion_writes_rd(global_writes_rd),
        .completion_rd(global_rd), .completion_mask(global_mask), .completion_data(global_data)
    );
    global_memory_arbiter global_bus (
        .clk(clk), .reset(reset),
        .lsu_valid(lsu_global_valid), .lsu_ready(lsu_global_ready), .lsu_write(lsu_global_write),
        .lsu_address(lsu_global_address), .lsu_data(lsu_global_data),
        .lsu_response_valid(lsu_global_response_valid), .lsu_response_ready(lsu_global_response_ready),
        .lsu_error(lsu_global_error), .lsu_result(lsu_global_result),
        .copy_valid(copy_global_valid), .copy_ready(copy_global_ready), .copy_address(copy_global_address),
        .copy_response_valid(copy_global_response_valid), .copy_response_ready(copy_global_response_ready),
        .copy_error(copy_global_error), .copy_result(copy_global_data),
        .request_valid(global_request_valid), .request_ready(global_request_ready),
        .request_write(global_request_write), .request_address(global_request_address),
        .request_data(global_request_data), .response_valid(global_response_valid),
        .response_ready(global_response_ready), .response_error(global_response_error),
        .response_data(global_response_data), .busy(global_bus_busy)
    );

    async_copy_unit async_copy (
        .clk(clk), .reset(reset), .command_valid(copy_start), .command_ready(copy_ready),
        .opcode(opcode), .warp_id(selected_warp_id), .active_mask(selected_active_mask),
        .rd_addr(rd_addr), .rs1_addr(rs1_addr), .rs2_addr(rs2_addr), .immediate(immediate), .source(rs1_data),
        .memory_idle(shared_request_ready && !shared_inflight && !host_memory_pending &&
                     !tmem_memory_lock && !copy_shared_pending),
        .busy(copy_busy), .warp_pending(copy_warp_pending), .status_bits(copy_status),
        .probe_addresses(lsu_addresses), .probe_store(is_shared_store),
        .matrix_a(rs1_data[31:0]), .matrix_b(rs2_data[31:0]),
        .matrix_valid(tmem_matrix_guard_valid), .matrix_nvfp4(tensor_nvfp4_opcode(opcode)),
        .shared_access_ready(copy_shared_access), .matrix_access_ready(copy_matrix_access),
        .global_valid(copy_global_valid), .global_ready(copy_global_ready), .global_address(copy_global_address),
        .global_response_valid(copy_global_response_valid), .global_response_ready(copy_global_response_ready),
        .global_error(copy_global_error), .global_data(copy_global_data),
        .shared_valid(copy_shared_valid), .shared_ready(copy_shared_ready),
        .shared_address(copy_shared_address), .shared_data(copy_shared_data),
        .shared_response_valid(copy_shared_pending && memory_response_valid),
        .shared_response_ready(copy_shared_response_ready),
        .row_valid(copy_row_valid), .row_ready(copy_take), .row_warp(copy_row_warp),
        .row_rd(copy_row_rd), .row_mask(copy_row_mask), .row_data(copy_row_data)
    );
    assign copy_shared_ready = shared_request_ready && !tmem_memory_lock && !host_memory_accept;
    always_ff @(posedge clk) begin
        if (reset) copy_shared_pending <= 0;
        else begin
            if (copy_shared_valid && copy_shared_ready) copy_shared_pending <= 1;
            if (copy_shared_pending && memory_response_valid && copy_shared_response_ready)
                copy_shared_pending <= 0;
        end
    end
    // SHARED MEMORY

    shared_memory #(.CLEAR_ON_RESET(CLEAR_SHARED_ON_RESET)) shared_mem (

        .clk               (clk),
        .reset             (reset),

        .request_valid     (shared_start || host_memory_accept || tmem_memory_valid ||
                            (copy_shared_valid && !tmem_memory_lock)),
        .request_ready     (shared_request_ready),
        .response_valid    (memory_response_valid),
        .response_ready    (memory_response_ready),

        .lane_addresses    (host_memory_accept
                            ? {96'd0, 20'd0, host_memory_address, 2'b0} :
                            tmem_memory_lock ? tmem_memory_addresses :
                            copy_shared_valid ? {96'd0, copy_shared_address} : lsu_addresses),
        .store_data        (host_memory_accept ? {96'd0, host_memory_data} :
                            copy_shared_valid && !tmem_memory_lock ? {96'd0, copy_shared_data} : lsu_store_data),

        .load_enable_mask  (
            host_memory_accept ? {3'b0, !host_memory_write} : tmem_memory_lock
                ? 4'hf : copy_shared_valid ? 4'd0 : instruction_execute_accept
                ? lsu_load_mask
                : 4'b0000
        ),

        .store_enable_mask (
            host_memory_accept ? {3'b0, host_memory_write} : tmem_memory_lock ? 4'd0 :
            copy_shared_valid ? 4'b0001 : instruction_execute_accept
                ? lsu_store_mask
                : 4'b0000
        ),

        .load_data         (shared_load_data),

        .conflict_mask     (shared_conflict_mask),
        .bank_conflict     (shared_bank_conflict)

    );

    always_ff @(posedge clk) begin
        if (reset) begin
            shared_inflight <= 1'b0;
            shared_is_load <= 1'b0;
            shared_warp_id <= 2'd0;
            shared_rd_addr <= 4'd0;
            shared_write_mask <= 4'd0;
        end else begin
            if (shared_start) begin
                shared_inflight <= 1'b1;
                shared_is_load <= is_shared_load;
                shared_warp_id <= selected_warp_id;
                shared_rd_addr <= rd_addr;
                shared_write_mask <= selected_active_mask;
            end
            if (shared_response_valid && shared_response_ready)
                shared_inflight <= 1'b0;
        end
    end
    // EXECUTE -> WRITEBACK

    always_ff @(posedge clk) begin

        if (reset) begin

            wb_valid      <= 1'b0;
            wb_warp_id    <= 2'd0;
            wb_rd_addr    <= 4'd0;
            wb_write_mask <= 4'b0000;
            wb_result     <= 128'd0;
            wb_is_tensor  <= 1'b0;
            wb_tensor_last <= 1'b0;

        end

        else if (wb_slot_available) begin

            wb_valid <= 1'b0;
            wb_is_tensor <= 1'b0;
            wb_tensor_last <= 1'b0;

            // Priority: FP, tensor row, tensor-memory row, copy ticket/status, shared load, global load/status, integer.
            // Each producer holds its result until it owns a writeback slot.
            if (fp_take) begin
                wb_valid      <= 1'b1;
                wb_warp_id    <= fp_warp_id;
                wb_rd_addr    <= fp_rd_addr;
                wb_write_mask <= fp_write_mask;
                wb_result     <= fp_result;
            end
            else if (tensor_take) begin
                wb_valid      <= 1'b1;
                wb_is_tensor  <= 1'b1;
                wb_tensor_last <= tensor_row_last;
                wb_warp_id    <= tensor_warp_id;
                wb_rd_addr    <= tensor_rd_addr;
                wb_write_mask <= tensor_write_mask;
                wb_result     <= tensor_result;
            end
            else if (tmem_take) begin
                wb_valid      <= 1'b1;
                wb_warp_id    <= tmem_row_warp;
                wb_rd_addr    <= tmem_row_rd;
                wb_write_mask <= tmem_row_mask;
                wb_result     <= tmem_row_data;
            end
            else if (copy_take) begin
                wb_valid      <= 1'b1;
                wb_warp_id    <= copy_row_warp;
                wb_rd_addr    <= copy_row_rd;
                wb_write_mask <= copy_row_mask;
                wb_result     <= copy_row_data;
            end
            else if (shared_take) begin
                wb_valid      <= 1'b1;
                wb_warp_id    <= shared_warp_id;
                wb_rd_addr    <= shared_rd_addr;
                wb_write_mask <= shared_write_mask;
                wb_result     <= shared_load_data;
            end
            else if (global_take) begin
                wb_valid      <= 1'b1;
                wb_warp_id    <= global_warp;
                wb_rd_addr    <= global_rd;
                wb_write_mask <= global_mask;
                wb_result     <= global_data;
            end
            else if (
                run_enable &&
                instruction_execute_accept &&
                is_integer_op
            ) begin

                wb_valid      <= 1'b1;
                wb_warp_id    <= selected_warp_id;
                wb_rd_addr    <= rd_addr;
                wb_write_mask <= selected_active_mask;
                wb_result     <= current_alu_result;

            end

        end

    end
    // WARP STATE CONTROL

    always @(*) begin

        state_update_enable =
            1'b0;

        state_update_warp_id =
            selected_warp_id;

        update_pc_enable =
            1'b0;

        update_pc =
            selected_pc;

        update_mask_enable =
            1'b0;

        update_active_mask =
            selected_active_mask;

        update_valid_enable =
            1'b0;

        update_valid =
            selected_valid;

        update_ready_enable =
            1'b0;

        update_ready =
            selected_ready;

        update_halted_enable =
            1'b0;

        update_halted =
            selected_halted;
        // PRIORITY 1: LAUNCH

        if (launch_enable) begin

            state_update_enable =
                1'b1;

            state_update_warp_id =
                launch_warp_id;

            update_pc_enable =
                1'b1;

            update_pc =
                launch_pc;

            update_mask_enable =
                1'b1;

            update_active_mask =
                launch_active_mask;

            update_valid_enable =
                1'b1;

            update_valid =
                1'b1;

            update_ready_enable =
                1'b1;

            update_ready =
                1'b1;

            update_halted_enable =
                1'b1;

            update_halted =
                1'b0;

        end
        // PRIORITY 2: RECONVERGENCE EVENT

        else if (
            issue_accept &&
            reconvergence_event
        ) begin

            state_update_enable =
                1'b1;

            state_update_warp_id =
                selected_warp_id;

            if (divergence_pc_update_enable) begin

                update_pc_enable =
                    1'b1;

                update_pc =
                    divergence_next_pc;

            end

            if (divergence_mask_update_enable) begin

                update_mask_enable =
                    1'b1;

                update_active_mask =
                    divergence_next_mask;

            end

        end
        // PRIORITY 3: ACCEPTED INSTRUCTION

        else if (
            run_enable &&
            instruction_execute_accept
        ) begin

            state_update_enable =
                1'b1;

            state_update_warp_id =
                selected_warp_id;
            // EXIT

            if (opcode == OP_EXIT) begin

                update_ready_enable =
                    1'b1;

                update_ready =
                    1'b0;

                update_halted_enable =
                    1'b1;

                update_halted =
                    1'b1;

            end
            // BEQ

            else if (is_branch) begin

                if (divergence_pc_update_enable) begin

                    update_pc_enable =
                        1'b1;

                    update_pc =
                        divergence_next_pc;

                end

                if (divergence_mask_update_enable) begin

                    update_mask_enable =
                        1'b1;

                    update_active_mask =
                        divergence_next_mask;

                end

            end
            // EVERYTHING ELSE

            else begin

                update_pc_enable =
                    1'b1;

                update_pc =
                    selected_pc + 32'd4;

            end

        end

    end

endmodule
