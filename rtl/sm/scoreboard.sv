module scoreboard (
    input  logic       clk,
    input  logic       reset,

    // Mark destination register busy when an instruction issues
    input  logic       mark_busy_enable,
    input  logic [1:0] mark_busy_warp_id,
    input  logic [3:0] mark_busy_reg,

    // Clear destination register when writeback completes
    input  logic       clear_busy_enable,
    input  logic [1:0] clear_busy_warp_id,
    input  logic [3:0] clear_busy_reg,

    // Dependency check for one candidate instruction
    input  logic [1:0] check_warp_id,
    input  logic [3:0] check_rs1,
    input  logic [3:0] check_rs2,

    output logic       rs1_busy,
    output logic       rs2_busy,
    output logic       sources_ready,

    // Debug/status
    output logic [15:0] selected_warp_busy_mask
);

    logic [15:0] busy [0:3];

    integer w;
    // BUSY STATE UPDATE

    always_ff @(posedge clk) begin

        if (reset) begin

            for (w = 0; w < 4; w = w + 1) begin
                busy[w] <= 16'b0;
            end

        end

        else begin

            // Clear first
            if (clear_busy_enable) begin
                busy[clear_busy_warp_id][clear_busy_reg] <= 1'b0;
            end

            // Mark busy
            //
            // If the same register is cleared and marked busy
            // in the same cycle, the new instruction wins.
            if (mark_busy_enable) begin
                busy[mark_busy_warp_id][mark_busy_reg] <= 1'b1;
            end

        end

    end
    // DEPENDENCY CHECK

    assign rs1_busy =
        busy[check_warp_id][check_rs1];

    assign rs2_busy =
        busy[check_warp_id][check_rs2];

    assign sources_ready =
        ~rs1_busy &
        ~rs2_busy;

    // Useful for GTKWave/debugging
    assign selected_warp_busy_mask =
        busy[check_warp_id];

endmodule
