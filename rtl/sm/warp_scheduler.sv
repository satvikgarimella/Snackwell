module warp_scheduler (
    input  logic       clk,
    input  logic       reset,

    input  logic [3:0] valid_mask,
    input  logic [3:0] ready_mask,
    input  logic [3:0] halted_mask,

    // Warps temporarily blocked by dependencies,
    // execution resources, memory, etc.
    input  logic [3:0] blocked_mask,

    // Only advance round-robin state when the
    // candidate instruction was actually accepted.
    input  logic       issue_accept,

    output logic       issue_valid,
    output logic [1:0] selected_warp_id,
    output logic [3:0] eligible_mask
);

    logic [1:0] rr_pointer;
    // ELIGIBILITY

    assign eligible_mask =
        valid_mask       &
        ready_mask       &
        ~halted_mask     &
        ~blocked_mask;

    // ROUND-ROBIN SELECTION
    //
    // rr_pointer means:
    // "start searching from this warp"

    always @(*) begin

        issue_valid      = 1'b0;
        selected_warp_id = rr_pointer;

        case (rr_pointer)

            // Search:
            // W0 -> W1 -> W2 -> W3
            2'd0: begin

                if (eligible_mask[0]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd0;
                end

                else if (eligible_mask[1]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd1;
                end

                else if (eligible_mask[2]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd2;
                end

                else if (eligible_mask[3]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd3;
                end

            end

            // Search:
            // W1 -> W2 -> W3 -> W0
            2'd1: begin

                if (eligible_mask[1]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd1;
                end

                else if (eligible_mask[2]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd2;
                end

                else if (eligible_mask[3]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd3;
                end

                else if (eligible_mask[0]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd0;
                end

            end

            // Search:
            // W2 -> W3 -> W0 -> W1
            2'd2: begin

                if (eligible_mask[2]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd2;
                end

                else if (eligible_mask[3]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd3;
                end

                else if (eligible_mask[0]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd0;
                end

                else if (eligible_mask[1]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd1;
                end

            end

            // Search:
            // W3 -> W0 -> W1 -> W2
            2'd3: begin

                if (eligible_mask[3]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd3;
                end

                else if (eligible_mask[0]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd0;
                end

                else if (eligible_mask[1]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd1;
                end

                else if (eligible_mask[2]) begin
                    issue_valid      = 1'b1;
                    selected_warp_id = 2'd2;
                end

            end

            default: begin

                issue_valid      = 1'b0;
                selected_warp_id = 2'd0;

            end

        endcase

    end
    // ROUND-ROBIN POINTER UPDATE

    always_ff @(posedge clk) begin

        if (reset) begin

            rr_pointer <= 2'd0;

        end

        else if (
            issue_valid &&
            issue_accept
        ) begin

            // Start next search after the warp that
            // successfully issued.

            case (selected_warp_id)

                2'd0:
                    rr_pointer <= 2'd1;

                2'd1:
                    rr_pointer <= 2'd2;

                2'd2:
                    rr_pointer <= 2'd3;

                2'd3:
                    rr_pointer <= 2'd0;

                default:
                    rr_pointer <= 2'd0;

            endcase

        end

    end

endmodule
