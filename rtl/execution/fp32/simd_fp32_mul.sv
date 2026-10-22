module simd_fp32_mul (
    input  logic         clk,
    input  logic         rst,

    input  logic         start,

    input  logic [127:0] operand_a,
    input  logic [127:0] operand_b,

    input  logic [3:0]   active_mask,

    output logic         busy,
    output logic         done,

    output logic [127:0] result
);

    logic [3:0] lane_start;
    logic [3:0] lane_busy;
    logic [3:0] lane_done;

    logic [31:0] lane_result [0:3];

    logic [3:0] started_mask;
    logic [3:0] completed_mask;

    integer i;
    // START ONLY ACTIVE LANES

    always_comb begin
        for (i = 0; i < 4; i = i + 1) begin
            lane_start[i] = start && !busy && active_mask[i];
        end
    end
    // FOUR FP32 MULTIPLIER LANES

    fp32_mul_wrapper lane0 (
        .clk       (clk),
        .rst       (rst),
        .start     (lane_start[0]),
        .operand_a (operand_a[31:0]),
        .operand_b (operand_b[31:0]),
        .busy      (lane_busy[0]),
        .done      (lane_done[0]),
        .result    (lane_result[0])
    );

    fp32_mul_wrapper lane1 (
        .clk       (clk),
        .rst       (rst),
        .start     (lane_start[1]),
        .operand_a (operand_a[63:32]),
        .operand_b (operand_b[63:32]),
        .busy      (lane_busy[1]),
        .done      (lane_done[1]),
        .result    (lane_result[1])
    );

    fp32_mul_wrapper lane2 (
        .clk       (clk),
        .rst       (rst),
        .start     (lane_start[2]),
        .operand_a (operand_a[95:64]),
        .operand_b (operand_b[95:64]),
        .busy      (lane_busy[2]),
        .done      (lane_done[2]),
        .result    (lane_result[2])
    );

    fp32_mul_wrapper lane3 (
        .clk       (clk),
        .rst       (rst),
        .start     (lane_start[3]),
        .operand_a (operand_a[127:96]),
        .operand_b (operand_b[127:96]),
        .busy      (lane_busy[3]),
        .done      (lane_done[3]),
        .result    (lane_result[3])
    );
    // PACK RESULTS

    always_comb begin
        result = {
            lane_result[3],
            lane_result[2],
            lane_result[1],
            lane_result[0]
        };
    end
    // TRACK WHICH LANES ARE FINISHED

    always_ff @(posedge clk) begin

        if (rst) begin

            started_mask   <= 4'b0000;
            completed_mask <= 4'b0000;

            busy <= 1'b0;
            done <= 1'b0;

        end else begin

            done <= 1'b0;

            // Start a SIMD operation
            if (start && !busy) begin

                started_mask   <= active_mask;
                completed_mask <= ~active_mask;

                busy <= 1'b1;

            end

            // Record completion of each lane
            if (busy) begin

                completed_mask <=
                    completed_mask |
                    lane_done;

                // All active lanes complete
                if (
                    (completed_mask | lane_done) ==
                    4'b1111
                ) begin

                    busy <= 1'b0;
                    done <= 1'b1;

                    started_mask <= 4'b0000;

                end

            end

        end

    end

endmodule
