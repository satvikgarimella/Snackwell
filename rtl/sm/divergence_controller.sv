module divergence_controller (
    // =================================================
    // CURRENT WARP STATE
    // =================================================

    input logic [31:0] current_pc,
    input logic [3:0]  current_active_mask,

    // =================================================
    // CURRENT INSTRUCTION TYPE
    // =================================================

    input logic        branch_valid,

    // =================================================
    // BRANCH UNIT RESULTS
    // =================================================

    input logic [3:0]  taken_mask,
    input logic [3:0]  not_taken_mask,

    input logic        uniform_taken,
    input logic        uniform_not_taken,
    input logic        divergent,

    input logic [31:0] branch_target,
    input logic [31:0] fallthrough_pc,

    // =================================================
    // SAVED RECONVERGENCE TARGET
    // =================================================

    input logic        reconvergence_target_valid,
    input logic [31:0] reconvergence_target_pc,

    // =================================================
    // CURRENT RECONVERGENCE STACK TOP
    // =================================================

    input logic        stack_empty,

    input logic [31:0] stack_deferred_pc,
    input logic [3:0]  stack_deferred_mask,

    input logic [31:0] stack_reconverge_pc,
    input logic [3:0]  stack_reconverge_mask,

    input logic        stack_deferred_pending,

    // =================================================
    // WARP STATE UPDATE OUTPUTS
    // =================================================

    output logic        pc_update_enable,
    output logic [31:0] next_pc,

    output logic        mask_update_enable,
    output logic [3:0]  next_active_mask,

    // =================================================
    // STACK COMMANDS
    // =================================================

    output logic        stack_push_enable,

    output logic [31:0] stack_push_deferred_pc,
    output logic [3:0]  stack_push_deferred_mask,

    output logic [31:0] stack_push_reconverge_pc,
    output logic [3:0]  stack_push_reconverge_mask,

    output logic        stack_pop_enable,

    output logic        stack_update_pending_enable,
    output logic        stack_update_deferred_pending,

    // =================================================
    // STATUS
    // =================================================

    output logic        reconvergence_event,
    output logic        switch_to_deferred_path,
    output logic        restore_reconverged_mask,

    // Invalid divergent branch with no SSY target.
    output logic        missing_reconvergence_target
);


    // =================================================
    // CONTROL
    // =================================================

    always @(*) begin

        // ---------------------------------------------
        // Defaults
        // ---------------------------------------------

        pc_update_enable   = 1'b0;
        next_pc            = current_pc;

        mask_update_enable = 1'b0;
        next_active_mask   = current_active_mask;


        stack_push_enable = 1'b0;

        stack_push_deferred_pc   = 32'd0;
        stack_push_deferred_mask = 4'b0000;

        stack_push_reconverge_pc   = 32'd0;
        stack_push_reconverge_mask = 4'b0000;


        stack_pop_enable = 1'b0;

        stack_update_pending_enable   = 1'b0;
        stack_update_deferred_pending = 1'b0;


        reconvergence_event       = 1'b0;
        switch_to_deferred_path   = 1'b0;
        restore_reconverged_mask  = 1'b0;

        missing_reconvergence_target = 1'b0;


        // =================================================
        // PRIORITY 1:
        // RECONVERGENCE POINT
        //
        // This happens before normal branch processing.
        // =================================================

        if (
            !stack_empty &&
            current_pc == stack_reconverge_pc
        ) begin

            reconvergence_event = 1'b1;


            // -----------------------------------------
            // FIRST ARRIVAL
            //
            // The path currently executing reached JOIN.
            // Switch to the deferred path.
            // -----------------------------------------

            if (stack_deferred_pending) begin

                switch_to_deferred_path = 1'b1;

                pc_update_enable = 1'b1;
                next_pc = stack_deferred_pc;

                mask_update_enable = 1'b1;
                next_active_mask = stack_deferred_mask;


                // Mark alternate path as consumed.
                stack_update_pending_enable = 1'b1;
                stack_update_deferred_pending = 1'b0;

            end


            // -----------------------------------------
            // SECOND ARRIVAL
            //
            // Both paths are complete.
            // Restore original active mask and pop.
            //
            // PC does NOT need to change here.
            // The normal instruction flow can continue
            // from the reconvergence point.
            // -----------------------------------------

            else begin

                restore_reconverged_mask = 1'b1;

                mask_update_enable = 1'b1;
                next_active_mask = stack_reconverge_mask;

                stack_pop_enable = 1'b1;

            end

        end


        // =================================================
        // PRIORITY 2:
        // BRANCH INSTRUCTION
        // =================================================

        else if (branch_valid) begin


            // -----------------------------------------
            // UNIFORM TAKEN
            // -----------------------------------------

            if (uniform_taken) begin

                pc_update_enable = 1'b1;
                next_pc = branch_target;

            end


            // -----------------------------------------
            // UNIFORM NOT TAKEN
            // -----------------------------------------

            else if (uniform_not_taken) begin

                pc_update_enable = 1'b1;
                next_pc = fallthrough_pc;

            end


            // -----------------------------------------
            // DIVERGENT BRANCH
            //
            // Execute taken path first.
            //
            // Save the not-taken path for later.
            // -----------------------------------------

            else if (divergent) begin

                if (reconvergence_target_valid) begin

                    // Execute taken lanes first.

                    pc_update_enable = 1'b1;
                    next_pc = branch_target;

                    mask_update_enable = 1'b1;
                    next_active_mask = taken_mask;


                    // Save not-taken lanes.

                    stack_push_enable = 1'b1;

                    stack_push_deferred_pc =
                        fallthrough_pc;

                    stack_push_deferred_mask =
                        not_taken_mask;

                    stack_push_reconverge_pc =
                        reconvergence_target_pc;

                    stack_push_reconverge_mask =
                        current_active_mask;

                end

                else begin

                    // Divergence without an SSY target is
                    // architecturally invalid in this v0 ISA.

                    missing_reconvergence_target = 1'b1;

                end

            end

        end

    end


endmodule