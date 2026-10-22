module reconvergence_stack (
    input logic clk,
    input logic reset,
    // WARP SELECTION

    input logic [1:0] warp_id,
    // PUSH

    input logic push_enable,

    input logic [31:0] push_deferred_pc,
    input logic [3:0]  push_deferred_mask,

    input logic [31:0] push_reconverge_pc,
    input logic [3:0]  push_reconverge_mask,
    // POP

    input logic pop_enable,

    // UPDATE TOP ENTRY
    //
    // Used after switching to the deferred path.

    input logic update_pending_enable,
    input logic update_deferred_pending,
    // TOP OF STACK

    output logic [31:0] top_deferred_pc,
    output logic [3:0]  top_deferred_mask,

    output logic [31:0] top_reconverge_pc,
    output logic [3:0]  top_reconverge_mask,

    output logic top_deferred_pending,
    // STATUS

    output logic       empty,
    output logic       full,
    output logic [2:0] depth,

    // Debug/error indicators
    output logic overflow_attempt,
    output logic underflow_attempt
);
    // PARAMETERS

    localparam integer NUM_WARPS   = 4;
    localparam integer STACK_DEPTH = 4;

    // STACK STORAGE
    //
    // [warp][stack entry]

    logic [31:0] deferred_pc_stack
        [0:NUM_WARPS-1][0:STACK_DEPTH-1];

    logic [3:0] deferred_mask_stack
        [0:NUM_WARPS-1][0:STACK_DEPTH-1];

    logic [31:0] reconverge_pc_stack
        [0:NUM_WARPS-1][0:STACK_DEPTH-1];

    logic [3:0] reconverge_mask_stack
        [0:NUM_WARPS-1][0:STACK_DEPTH-1];

    logic deferred_pending_stack
        [0:NUM_WARPS-1][0:STACK_DEPTH-1];

    // STACK POINTERS
    //
    // Number of valid entries:
    //
    // 0 = empty
    // 4 = full

    logic [2:0] stack_depth [0:NUM_WARPS-1];

    integer w;
    integer s;
    // STATUS

    assign depth = stack_depth[warp_id];

    assign empty =
        (stack_depth[warp_id] == 3'd0);

    assign full =
        (stack_depth[warp_id] == 3'd4);
    // TOP-OF-STACK READ

    always @(*) begin

        top_deferred_pc      = 32'd0;
        top_deferred_mask    = 4'b0000;

        top_reconverge_pc    = 32'd0;
        top_reconverge_mask  = 4'b0000;

        top_deferred_pending = 1'b0;

        if (stack_depth[warp_id] != 3'd0) begin

            case (stack_depth[warp_id])

                3'd1: begin

                    top_deferred_pc =
                        deferred_pc_stack[warp_id][0];

                    top_deferred_mask =
                        deferred_mask_stack[warp_id][0];

                    top_reconverge_pc =
                        reconverge_pc_stack[warp_id][0];

                    top_reconverge_mask =
                        reconverge_mask_stack[warp_id][0];

                    top_deferred_pending =
                        deferred_pending_stack[warp_id][0];

                end

                3'd2: begin

                    top_deferred_pc =
                        deferred_pc_stack[warp_id][1];

                    top_deferred_mask =
                        deferred_mask_stack[warp_id][1];

                    top_reconverge_pc =
                        reconverge_pc_stack[warp_id][1];

                    top_reconverge_mask =
                        reconverge_mask_stack[warp_id][1];

                    top_deferred_pending =
                        deferred_pending_stack[warp_id][1];

                end

                3'd3: begin

                    top_deferred_pc =
                        deferred_pc_stack[warp_id][2];

                    top_deferred_mask =
                        deferred_mask_stack[warp_id][2];

                    top_reconverge_pc =
                        reconverge_pc_stack[warp_id][2];

                    top_reconverge_mask =
                        reconverge_mask_stack[warp_id][2];

                    top_deferred_pending =
                        deferred_pending_stack[warp_id][2];

                end

                3'd4: begin

                    top_deferred_pc =
                        deferred_pc_stack[warp_id][3];

                    top_deferred_mask =
                        deferred_mask_stack[warp_id][3];

                    top_reconverge_pc =
                        reconverge_pc_stack[warp_id][3];

                    top_reconverge_mask =
                        reconverge_mask_stack[warp_id][3];

                    top_deferred_pending =
                        deferred_pending_stack[warp_id][3];

                end

                default: begin

                    top_deferred_pc      = 32'd0;
                    top_deferred_mask    = 4'b0000;

                    top_reconverge_pc    = 32'd0;
                    top_reconverge_mask  = 4'b0000;

                    top_deferred_pending = 1'b0;

                end

            endcase

        end

    end
    // ERROR STATUS

    assign overflow_attempt =
        push_enable &&
        full;

    assign underflow_attempt =
        pop_enable &&
        empty;
    // STACK UPDATE

    always_ff @(posedge clk) begin

        if (reset) begin

            for (w = 0; w < NUM_WARPS; w = w + 1) begin

                stack_depth[w] <= 3'd0;

                for (
                    s = 0;
                    s < STACK_DEPTH;
                    s = s + 1
                ) begin

                    deferred_pc_stack[w][s]
                        <= 32'd0;

                    deferred_mask_stack[w][s]
                        <= 4'b0000;

                    reconverge_pc_stack[w][s]
                        <= 32'd0;

                    reconverge_mask_stack[w][s]
                        <= 4'b0000;

                    deferred_pending_stack[w][s]
                        <= 1'b0;

                end

            end

        end

        else begin

            // PUSH
            //
            // Highest priority operation.

            if (
                push_enable &&
                !full
            ) begin

                case (stack_depth[warp_id])

                    3'd0: begin

                        deferred_pc_stack[warp_id][0]
                            <= push_deferred_pc;

                        deferred_mask_stack[warp_id][0]
                            <= push_deferred_mask;

                        reconverge_pc_stack[warp_id][0]
                            <= push_reconverge_pc;

                        reconverge_mask_stack[warp_id][0]
                            <= push_reconverge_mask;

                        deferred_pending_stack[warp_id][0]
                            <= 1'b1;

                    end

                    3'd1: begin

                        deferred_pc_stack[warp_id][1]
                            <= push_deferred_pc;

                        deferred_mask_stack[warp_id][1]
                            <= push_deferred_mask;

                        reconverge_pc_stack[warp_id][1]
                            <= push_reconverge_pc;

                        reconverge_mask_stack[warp_id][1]
                            <= push_reconverge_mask;

                        deferred_pending_stack[warp_id][1]
                            <= 1'b1;

                    end

                    3'd2: begin

                        deferred_pc_stack[warp_id][2]
                            <= push_deferred_pc;

                        deferred_mask_stack[warp_id][2]
                            <= push_deferred_mask;

                        reconverge_pc_stack[warp_id][2]
                            <= push_reconverge_pc;

                        reconverge_mask_stack[warp_id][2]
                            <= push_reconverge_mask;

                        deferred_pending_stack[warp_id][2]
                            <= 1'b1;

                    end

                    3'd3: begin

                        deferred_pc_stack[warp_id][3]
                            <= push_deferred_pc;

                        deferred_mask_stack[warp_id][3]
                            <= push_deferred_mask;

                        reconverge_pc_stack[warp_id][3]
                            <= push_reconverge_pc;

                        reconverge_mask_stack[warp_id][3]
                            <= push_reconverge_mask;

                        deferred_pending_stack[warp_id][3]
                            <= 1'b1;

                    end

                endcase

                stack_depth[warp_id]
                    <= stack_depth[warp_id] + 3'd1;

            end
            // POP

            else if (
                pop_enable &&
                !empty
            ) begin

                stack_depth[warp_id]
                    <= stack_depth[warp_id] - 3'd1;

            end
            // UPDATE TOP DEFERRED-PENDING BIT

            else if (
                update_pending_enable &&
                !empty
            ) begin

                case (stack_depth[warp_id])

                    3'd1:
                        deferred_pending_stack[warp_id][0]
                            <= update_deferred_pending;

                    3'd2:
                        deferred_pending_stack[warp_id][1]
                            <= update_deferred_pending;

                    3'd3:
                        deferred_pending_stack[warp_id][2]
                            <= update_deferred_pending;

                    3'd4:
                        deferred_pending_stack[warp_id][3]
                            <= update_deferred_pending;

                endcase

            end

        end

    end

endmodule
