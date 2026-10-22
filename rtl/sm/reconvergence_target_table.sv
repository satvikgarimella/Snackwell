module reconvergence_target_table (
    input logic clk,
    input logic reset,

    // Write target
    input logic        write_enable,
    input logic [1:0]  write_warp_id,
    input logic [31:0] write_pc,

    // Consume target
    input logic        clear_enable,
    input logic [1:0]  clear_warp_id,

    // Read target
    input  logic [1:0]  read_warp_id,
    output logic [31:0] read_pc,
    output logic        read_valid
);

    logic [31:0] reconvergence_pc [0:3];
    logic        reconvergence_valid [0:3];

    integer w;

    assign read_pc =
        reconvergence_pc[read_warp_id];

    assign read_valid =
        reconvergence_valid[read_warp_id];

    always_ff @(posedge clk) begin

        if (reset) begin

            for (w = 0; w < 4; w = w + 1) begin

                reconvergence_pc[w] <= 32'd0;
                reconvergence_valid[w] <= 1'b0;

            end

        end

        else begin

            // Clear first
            if (clear_enable) begin

                reconvergence_valid[clear_warp_id]
                    <= 1'b0;

            end

            // Write wins if both happen to target same cycle.
            if (write_enable) begin

                reconvergence_pc[write_warp_id]
                    <= write_pc;

                reconvergence_valid[write_warp_id]
                    <= 1'b1;

            end

        end

    end

endmodule
