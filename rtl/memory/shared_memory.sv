module shared_memory (
    input  logic         clk,
    input  logic         reset,

    // ------------------------------------------------
    // FOUR LANE REQUESTS
    // ------------------------------------------------

    input  logic [127:0] lane_addresses,

    input  logic [127:0] store_data,

    input  logic [3:0]   load_enable_mask,
    input  logic [3:0]   store_enable_mask,

    // ------------------------------------------------
    // LOAD RESPONSES
    // ------------------------------------------------

    output logic [127:0] load_data,

    // ------------------------------------------------
    // BANK CONFLICT STATUS
    // ------------------------------------------------

    output logic [3:0]   conflict_mask,
    output logic         bank_conflict
);

    // =================================================
    // 4 KB SHARED MEMORY
    //
    // 4 banks
    // 256 x 32-bit words per bank
    //
    // 4 * 256 * 32 = 32768 bits = 4096 bytes
    // =================================================

    logic [31:0] bank0 [0:255];
    logic [31:0] bank1 [0:255];
    logic [31:0] bank2 [0:255];
    logic [31:0] bank3 [0:255];


    // =================================================
    // PER-LANE ADDRESS DECODE
    // =================================================

    logic [31:0] addr0;
    logic [31:0] addr1;
    logic [31:0] addr2;
    logic [31:0] addr3;

    logic [1:0] bank_sel0;
    logic [1:0] bank_sel1;
    logic [1:0] bank_sel2;
    logic [1:0] bank_sel3;

    logic [7:0] row0;
    logic [7:0] row1;
    logic [7:0] row2;
    logic [7:0] row3;


    assign addr0 = lane_addresses[31:0];
    assign addr1 = lane_addresses[63:32];
    assign addr2 = lane_addresses[95:64];
    assign addr3 = lane_addresses[127:96];


    // Byte addresses:
    //
    // bits [1:0] = byte offset inside 32-bit word
    // bits [3:2] = bank number
    // bits [11:4] = row
    //
    // Total addressable range = 4 KB.

    assign bank_sel0 = addr0[3:2];
    assign bank_sel1 = addr1[3:2];
    assign bank_sel2 = addr2[3:2];
    assign bank_sel3 = addr3[3:2];

    assign row0 = addr0[11:4];
    assign row1 = addr1[11:4];
    assign row2 = addr2[11:4];
    assign row3 = addr3[11:4];


    // =================================================
    // REQUEST ACTIVITY
    // =================================================

    logic [3:0] request_mask;

    assign request_mask =
        load_enable_mask |
        store_enable_mask;


    // =================================================
    // BANK CONFLICT DETECTION
    //
    // A lane conflicts if another active lane requests
    // the same bank during this cycle.
    // =================================================

    always @(*) begin

        conflict_mask = 4'b0000;


        if (
            request_mask[0] &&
            request_mask[1] &&
            bank_sel0 == bank_sel1
        ) begin
            conflict_mask[0] = 1'b1;
            conflict_mask[1] = 1'b1;
        end


        if (
            request_mask[0] &&
            request_mask[2] &&
            bank_sel0 == bank_sel2
        ) begin
            conflict_mask[0] = 1'b1;
            conflict_mask[2] = 1'b1;
        end


        if (
            request_mask[0] &&
            request_mask[3] &&
            bank_sel0 == bank_sel3
        ) begin
            conflict_mask[0] = 1'b1;
            conflict_mask[3] = 1'b1;
        end


        if (
            request_mask[1] &&
            request_mask[2] &&
            bank_sel1 == bank_sel2
        ) begin
            conflict_mask[1] = 1'b1;
            conflict_mask[2] = 1'b1;
        end


        if (
            request_mask[1] &&
            request_mask[3] &&
            bank_sel1 == bank_sel3
        ) begin
            conflict_mask[1] = 1'b1;
            conflict_mask[3] = 1'b1;
        end


        if (
            request_mask[2] &&
            request_mask[3] &&
            bank_sel2 == bank_sel3
        ) begin
            conflict_mask[2] = 1'b1;
            conflict_mask[3] = 1'b1;
        end

    end


    assign bank_conflict =
        |conflict_mask;


    // =================================================
    // STORES
    //
    // For now, conflicting stores are NOT committed.
    //
    // Later we can serialize/replay them.
    // =================================================

    integer i;

    always_ff @(posedge clk) begin

        if (reset) begin

            for (i = 0; i < 256; i = i + 1) begin

                bank0[i] <= 32'd0;
                bank1[i] <= 32'd0;
                bank2[i] <= 32'd0;
                bank3[i] <= 32'd0;

            end

        end

        else begin

            // -----------------------------------------
            // Lane 0
            // -----------------------------------------

            if (
                store_enable_mask[0] &&
                !conflict_mask[0]
            ) begin

                case (bank_sel0)

                    2'd0:
                        bank0[row0] <= store_data[31:0];

                    2'd1:
                        bank1[row0] <= store_data[31:0];

                    2'd2:
                        bank2[row0] <= store_data[31:0];

                    2'd3:
                        bank3[row0] <= store_data[31:0];

                endcase

            end


            // -----------------------------------------
            // Lane 1
            // -----------------------------------------

            if (
                store_enable_mask[1] &&
                !conflict_mask[1]
            ) begin

                case (bank_sel1)

                    2'd0:
                        bank0[row1] <= store_data[63:32];

                    2'd1:
                        bank1[row1] <= store_data[63:32];

                    2'd2:
                        bank2[row1] <= store_data[63:32];

                    2'd3:
                        bank3[row1] <= store_data[63:32];

                endcase

            end


            // -----------------------------------------
            // Lane 2
            // -----------------------------------------

            if (
                store_enable_mask[2] &&
                !conflict_mask[2]
            ) begin

                case (bank_sel2)

                    2'd0:
                        bank0[row2] <= store_data[95:64];

                    2'd1:
                        bank1[row2] <= store_data[95:64];

                    2'd2:
                        bank2[row2] <= store_data[95:64];

                    2'd3:
                        bank3[row2] <= store_data[95:64];

                endcase

            end


            // -----------------------------------------
            // Lane 3
            // -----------------------------------------

            if (
                store_enable_mask[3] &&
                !conflict_mask[3]
            ) begin

                case (bank_sel3)

                    2'd0:
                        bank0[row3] <= store_data[127:96];

                    2'd1:
                        bank1[row3] <= store_data[127:96];

                    2'd2:
                        bank2[row3] <= store_data[127:96];

                    2'd3:
                        bank3[row3] <= store_data[127:96];

                endcase

            end

        end

    end


    // =================================================
    // LOADS
    //
    // Combinational for now.
    //
    // Conflicting lanes return zero.
    // Later they'll be serialized/replayed.
    // =================================================

    always @(*) begin

        load_data = 128'd0;


        // -----------------------------------------
        // Lane 0
        // -----------------------------------------

        if (
            load_enable_mask[0] &&
            !conflict_mask[0]
        ) begin

            case (bank_sel0)

                2'd0:
                    load_data[31:0] = bank0[row0];

                2'd1:
                    load_data[31:0] = bank1[row0];

                2'd2:
                    load_data[31:0] = bank2[row0];

                2'd3:
                    load_data[31:0] = bank3[row0];

            endcase

        end


        // -----------------------------------------
        // Lane 1
        // -----------------------------------------

        if (
            load_enable_mask[1] &&
            !conflict_mask[1]
        ) begin

            case (bank_sel1)

                2'd0:
                    load_data[63:32] = bank0[row1];

                2'd1:
                    load_data[63:32] = bank1[row1];

                2'd2:
                    load_data[63:32] = bank2[row1];

                2'd3:
                    load_data[63:32] = bank3[row1];

            endcase

        end


        // -----------------------------------------
        // Lane 2
        // -----------------------------------------

        if (
            load_enable_mask[2] &&
            !conflict_mask[2]
        ) begin

            case (bank_sel2)

                2'd0:
                    load_data[95:64] = bank0[row2];

                2'd1:
                    load_data[95:64] = bank1[row2];

                2'd2:
                    load_data[95:64] = bank2[row2];

                2'd3:
                    load_data[95:64] = bank3[row2];

            endcase

        end


        // -----------------------------------------
        // Lane 3
        // -----------------------------------------

        if (
            load_enable_mask[3] &&
            !conflict_mask[3]
        ) begin

            case (bank_sel3)

                2'd0:
                    load_data[127:96] = bank0[row3];

                2'd1:
                    load_data[127:96] = bank1[row3];

                2'd2:
                    load_data[127:96] = bank2[row3];

                2'd3:
                    load_data[127:96] = bank3[row3];

            endcase

        end

    end


endmodule