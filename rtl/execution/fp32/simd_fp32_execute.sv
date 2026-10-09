// One outstanding SIMD FP instruction. All active lanes must finish before
// done pulses; inactive lanes never start. Results remain stable until reuse.
module simd_fp32_execute (
    input  logic         clk,
    input  logic         rst,
    input  logic         start,
    input  logic [5:0]   opcode,
    input  logic [127:0] operand_a,
    input  logic [127:0] operand_b,
    input  logic [3:0]   active_mask,
    output logic         busy,
    output logic         done,
    output logic [127:0] result
);
    import snackwell_pkg::*;
    logic use_mul;
    logic [3:0] completed_mask, lane_done;
    logic [127:0] mul_result, add_result;
    logic [3:0] mul_done, add_done;
    wire accept = start && !busy &&
        (opcode == snackwell_pkg::OP_FMUL || opcode == OP_FADD || opcode == snackwell_pkg::OP_FSUB);

    for (genvar lane = 0; lane < 4; lane = lane + 1) begin : lanes
        fp32_mul_wrapper mul (
            .clk(clk), .rst(rst),
            .start(accept && active_mask[lane] && opcode == snackwell_pkg::OP_FMUL),
            .operand_a(operand_a[lane*32 +: 32]),
            .operand_b(operand_b[lane*32 +: 32]),
            .busy(), .done(mul_done[lane]),
            .result(mul_result[lane*32 +: 32])
        );
        fp32_addsub addsub (
            .clk(clk), .rst(rst),
            .start(accept && active_mask[lane] && opcode != snackwell_pkg::OP_FMUL),
            .subtract(opcode == snackwell_pkg::OP_FSUB),
            .operand_a(operand_a[lane*32 +: 32]),
            .operand_b(operand_b[lane*32 +: 32]),
            .busy(), .done(add_done[lane]),
            .result(add_result[lane*32 +: 32])
        );
    end

    assign lane_done = use_mul ? mul_done : add_done;
    assign result = use_mul ? mul_result : add_result;

    always_ff @(posedge clk) begin
        if (rst) begin
            use_mul <= 1'b0;
            completed_mask <= 4'b0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (accept) begin
                use_mul <= opcode == snackwell_pkg::OP_FMUL;
                completed_mask <= ~active_mask;
                busy <= 1'b1;
            end else if (busy) begin
                completed_mask <= completed_mask | lane_done;
                if (&(completed_mask | lane_done)) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule
