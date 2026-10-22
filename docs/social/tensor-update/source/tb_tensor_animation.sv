`timescale 1ns/1ps
module tb_tensor_animation;
logic clk=0, reset=1, request_valid=0;
logic [511:0] a_tile, b_tile;
wire request_ready, busy, result_valid, step_advance;
wire [511:0] result_tile;
wire [3:0] step_index;
integer cycles=0;
always #5 clk=~clk;
always @(posedge clk) if (!reset) cycles=cycles+1;
systolic_array_4x4 dut (.clk(clk),.reset(reset),.enable(1'b1),
.request_valid(request_valid),.request_ready(request_ready),.a_tile(a_tile),.b_tile(b_tile),
.busy(busy),.result_valid(result_valid),.result_ready(1'b0),.result_tile(result_tile),
.step_index(step_index),.step_advance(step_advance),.pe_done_mask());
for(genvar r=0;r<4;r=r+1)begin:rs
for(genvar c=0;c<4;c=c+1)begin:cs
always @(posedge clk) if(!reset && step_advance)
$display("TRACE,%0d,%0d,%0d,%0d,%h,%h,%b,%b,%h", cycles,step_index,r,c,
dut.rows[r].columns[c].pe.saved_a,dut.rows[r].columns[c].pe.saved_b,
dut.rows[r].columns[c].pe.saved_a_valid,dut.rows[r].columns[c].pe.saved_b_valid,
dut.rows[r].columns[c].pe.accumulator);
end end
initial begin
a_tile=512'h400000003f800000000000003f80000040400000400000003f800000000000003f800000000000003f800000400000004080000040400000400000003f800000;
b_tile=512'h000000003f800000400000003f8000003f800000000000003f80000040000000400000003f8000003f800000000000003f80000040000000000000003f800000;
repeat(3) @(negedge clk); reset=0;
@(negedge clk); request_valid=1;
@(negedge clk); request_valid=0;
wait(result_valid); #1;
if(result_tile !== 512'h400000004080000040a0000040a0000040800000408000004110000040e000004080000040c00000404000004040000041000000410000004150000041300000) $fatal(1,"Animation result mismatch");
$display("PASS exact animation matrices through RTL");
$finish;
end
initial begin #100000; $fatal(1,"Animation trace timeout"); end
endmodule
