module warp_state_table (
    input  logic        clk,
    input  logic        reset,
    // READ PORT

    input  logic [1:0]  read_warp_id,

    output logic [31:0] read_pc,
    output logic [3:0]  read_active_mask,
    output logic        read_valid,
    output logic        read_ready,
    output logic        read_halted,
    // UPDATE PORT

    input  logic        update_enable,
    input  logic [1:0]  update_warp_id,

    input  logic        update_pc_enable,
    input  logic [31:0] update_pc,

    input  logic        update_mask_enable,
    input  logic [3:0]  update_active_mask,

    input  logic        update_valid_enable,
    input  logic        update_valid,

    input  logic        update_ready_enable,
    input  logic        update_ready,

    input  logic        update_halted_enable,
    input  logic        update_halted,
    // SCHEDULER STATUS VECTORS

    output logic [3:0]  valid_mask,
    output logic [3:0]  ready_mask,
    output logic [3:0]  halted_mask
);
    // STATE STORAGE

    logic [31:0] warp_pc          [0:3];
    logic [3:0]  warp_active_mask [0:3];

    logic        warp_valid       [0:3];
    logic        warp_ready       [0:3];
    logic        warp_halted      [0:3];

    integer i;
    // STATE UPDATE

    always_ff @(posedge clk) begin

        if (reset) begin

            for (i = 0; i < 4; i = i + 1) begin

                warp_pc[i]          <= 32'd0;
                warp_active_mask[i] <= 4'b0000;

                warp_valid[i]       <= 1'b0;
                warp_ready[i]       <= 1'b0;
                warp_halted[i]      <= 1'b0;

            end

        end

        else if (update_enable) begin

            if (update_pc_enable) begin
                warp_pc[update_warp_id] <= update_pc;
            end

            if (update_mask_enable) begin
                warp_active_mask[update_warp_id]
                    <= update_active_mask;
            end

            if (update_valid_enable) begin
                warp_valid[update_warp_id]
                    <= update_valid;
            end

            if (update_ready_enable) begin
                warp_ready[update_warp_id]
                    <= update_ready;
            end

            if (update_halted_enable) begin
                warp_halted[update_warp_id]
                    <= update_halted;
            end

        end

    end
    // SELECTED WARP READ

    assign read_pc =
        warp_pc[read_warp_id];

    assign read_active_mask =
        warp_active_mask[read_warp_id];

    assign read_valid =
        warp_valid[read_warp_id];

    assign read_ready =
        warp_ready[read_warp_id];

    assign read_halted =
        warp_halted[read_warp_id];
    // SCHEDULER MASKS

    assign valid_mask = {
        warp_valid[3],
        warp_valid[2],
        warp_valid[1],
        warp_valid[0]
    };

    assign ready_mask = {
        warp_ready[3],
        warp_ready[2],
        warp_ready[1],
        warp_ready[0]
    };

    assign halted_mask = {
        warp_halted[3],
        warp_halted[2],
        warp_halted[1],
        warp_halted[0]
    };

endmodule
