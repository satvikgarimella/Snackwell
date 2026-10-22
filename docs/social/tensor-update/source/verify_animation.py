"""Simulate the animation's exact matrices and verify every PE at every step."""
from pathlib import Path
import json, struct, subprocess, shutil
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
data=json.loads((ROOT/'source/animation-data.json').read_text())
bit=lambda n:struct.unpack('>I',struct.pack('>f',float(n)))[0]
def tile(name):
    return "512'h"+''.join(f'{bit(v):08x}' for row in reversed(data[name]) for v in reversed(row))
bench='''`timescale 1ns/1ps
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
A_ASSIGN
B_ASSIGN
repeat(3) @(negedge clk); reset=0;
@(negedge clk); request_valid=1;
@(negedge clk); request_valid=0;
wait(result_valid); #1;
if(result_tile !== EXPECTED) $fatal(1,"Animation result mismatch");
$display("PASS exact animation matrices through RTL");
$finish;
end
initial begin #100000; $fatal(1,"Animation trace timeout"); end
endmodule
'''.replace('A_ASSIGN','a_tile='+tile('A')+';').replace('B_ASSIGN','b_tile='+tile('B')+';').replace('EXPECTED',tile('C'))
p=ROOT/'source/tb_tensor_animation.sv';p.write_text(bench)
compiled=ROOT/'source/animation-sim.vvp'
package=REPO/'rtl/snackwell_pkg.sv'
sources=[package,*sorted(p for p in (REPO/'rtl').rglob('*.sv') if p != package),*sorted((REPO/'rtl').rglob('*.v'))]
r=subprocess.run([shutil.which('iverilog') or '/opt/homebrew/bin/iverilog','-g2012','-s','tb_tensor_animation','-o',str(compiled),*map(str,sources),str(p)],capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
r=subprocess.run([shutil.which('vvp') or '/opt/homebrew/bin/vvp',str(compiled)],capture_output=True,text=True,timeout=30)
(ROOT/'source/animation-rtl.log').write_text(r.stdout+r.stderr)
assert r.returncode==0,r.stdout+r.stderr
records=[]
for line in r.stdout.splitlines():
 if not line.startswith('TRACE,'):continue
 _,clock,t,r,c,a,b,av,bv,acc=line.split(',')
 clock,t,r,c=map(int,(clock,t,r,c));k=t-r-c
 expected_acc=sum(data['A'][r][j]*data['B'][j][c] for j in range(min(4,max(0,k+1))))
 assert int(acc,16)==bit(expected_acc),(t,r,c,acc,expected_acc)
 valid=0<=k<4
 assert (av=='1')==valid and (bv=='1')==valid
 if valid:
  assert int(a,16)==bit(data['A'][r][k]) and int(b,16)==bit(data['B'][k][c])
 records.append(dict(clock=clock,step=t,row=r,column=c,k=k if valid else None,accumulator=expected_acc))
assert len(records)==160 and sum(r['k'] is not None for r in records)==64
records.sort(key=lambda r:(r['step'],r['row'],r['column']))
(ROOT/'source/animation-trace.json').write_text(json.dumps(records,indent=2)+'\n')
print('PASS: animation verified against RTL: 10 logical steps, 160 PE snapshots, 64 MACs, 16 outputs.')
print('Advance clocks:', sorted(set(r['clock'] for r in records)))
compiled.unlink()
