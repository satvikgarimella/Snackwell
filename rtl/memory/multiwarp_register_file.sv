module multiwarp_register_file (
    input  logic         clk,
    input  logic         reset,

    // ------------------------------------------------
    // READ CONTEXT
    // ------------------------------------------------

    input  logic [1:0]   read_warp_id,

    input  logic [3:0]   rs1_addr,
    input  logic [3:0]   rs2_addr,

    output logic [127:0] rs1_data,
    output logic [127:0] rs2_data,

    // ------------------------------------------------
    // WRITEBACK CONTEXT
    // ------------------------------------------------

    input  logic         write_enable,
    input  logic [1:0]   write_warp_id,
    input  logic [3:0]   rd_addr,
    input  logic [127:0] write_data,
    input  logic [3:0]   write_mask,

    // ------------------------------------------------
    // DEBUG PRELOAD
    // ------------------------------------------------

    input  logic         debug_write_enable,
    input  logic [1:0]   debug_warp_id,
    input  logic [1:0]   debug_lane_id,
    input  logic [3:0]   debug_reg_addr,
    input  logic [31:0]  debug_write_data
);

    logic [31:0] registers [0:3][0:3][0:15];

    integer w;
    integer l;
    integer r;


    // =================================================
    // RESET + WRITEBACK
    // =================================================

    always_ff @(posedge clk) begin

        if (reset) begin

            for (w = 0; w < 4; w = w + 1) begin
                for (l = 0; l < 4; l = l + 1) begin
                    for (r = 0; r < 16; r = r + 1) begin
                        registers[w][l][r] <= 32'd0;
                    end
                end
            end

        end

        else begin

            // Debug preload gets priority.
            if (debug_write_enable) begin

                registers
                    [debug_warp_id]
                    [debug_lane_id]
                    [debug_reg_addr]
                    <= debug_write_data;

            end

            else if (write_enable) begin

                if (write_mask[0])
                    registers[write_warp_id][0][rd_addr]
                        <= write_data[31:0];

                if (write_mask[1])
                    registers[write_warp_id][1][rd_addr]
                        <= write_data[63:32];

                if (write_mask[2])
                    registers[write_warp_id][2][rd_addr]
                        <= write_data[95:64];

                if (write_mask[3])
                    registers[write_warp_id][3][rd_addr]
                        <= write_data[127:96];

            end

        end

    end


    // =================================================
    // COMBINATIONAL READS
    // =================================================

    assign rs1_data = {
        registers[read_warp_id][3][rs1_addr],
        registers[read_warp_id][2][rs1_addr],
        registers[read_warp_id][1][rs1_addr],
        registers[read_warp_id][0][rs1_addr]
    };


    assign rs2_data = {
        registers[read_warp_id][3][rs2_addr],
        registers[read_warp_id][2][rs2_addr],
        registers[read_warp_id][1][rs2_addr],
        registers[read_warp_id][0][rs2_addr]
    };


endmodule