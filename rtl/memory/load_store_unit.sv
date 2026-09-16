module load_store_unit (
    // ------------------------------------------------
    // PER-LANE BASE ADDRESSES
    // ------------------------------------------------

    input  logic [127:0] base_addresses,

    // Shared immediate offset
    input  logic [31:0]  offset,

    // Store source values, one per lane
    input  logic [127:0] store_data,

    input  logic [3:0]   active_mask,

    input  logic         is_load,
    input  logic         is_store,

    // ------------------------------------------------
    // OUTPUTS
    // ------------------------------------------------

    output logic [127:0] lane_addresses,
    output logic [127:0] store_write_data,

    output logic [3:0]   load_enable_mask,
    output logic [3:0]   store_enable_mask
);

    // =================================================
    // PER-LANE ADDRESS GENERATION
    //
    // address[lane] = base[lane] + offset
    // =================================================

    assign lane_addresses[31:0] =
        base_addresses[31:0] + offset;

    assign lane_addresses[63:32] =
        base_addresses[63:32] + offset;

    assign lane_addresses[95:64] =
        base_addresses[95:64] + offset;

    assign lane_addresses[127:96] =
        base_addresses[127:96] + offset;


    // =================================================
    // STORE DATA
    // =================================================

    assign store_write_data = store_data;


    // =================================================
    // ACTIVE-LANE REQUEST MASKS
    // =================================================

    assign load_enable_mask =
        is_load
            ? active_mask
            : 4'b0000;


    assign store_enable_mask =
        is_store
            ? active_mask
            : 4'b0000;


endmodule