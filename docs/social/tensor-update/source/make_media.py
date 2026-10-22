"""Rebuild Snackwell social diagrams, editable Excalidraw scenes, and flow video.

Uses Pillow and a locally compiled macOS AVFoundation encoder. Artifacts
visualize the logical-step protocol; animation speed is not hardware timing.
"""
from pathlib import Path
import json, math, struct, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
OUT.mkdir(exist_ok=True)
QA=ROOT/'source/qa';QA.mkdir(exist_ok=True)
W, H = 1600, 1000
BG='#fffefa'; INK='#22262f'; MUTED='#606674'; LINE='#bac0c9'
BLUE='#1971c2'; BLUE_LIGHT='#e7f5ff'; ORANGE='#c56a0a'; ORANGE_LIGHT='#fff4e6'
PURPLE='#7048c6'; PURPLE_LIGHT='#f3f0ff'; GREEN='#2b8a3e'; GREEN_LIGHT='#ebfbee'
SANS='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
MONO='/System/Library/Fonts/Menlo.ttc'
FONTS={}
def font(size, bold=False, mono=False):
    key=(size,bold,mono)
    if key not in FONTS: FONTS[key]=ImageFont.truetype(MONO if mono else BOLD if bold else SANS,size)
    return FONTS[key]

def label(draw, xy, text, size=28, fill=INK, bold=False, anchor=None, mono=False):
    draw.text(xy,str(text),font=font(size,bold,mono),fill=fill,anchor=anchor,spacing=8)

def rounded(draw, xy, fill=BG, outline=INK, width=2, radius=14):
    draw.rounded_rectangle(xy,radius=radius,fill=fill,outline=outline,width=width)

def arrow(draw, pts, color=INK, width=3, head=11):
    draw.line(pts,fill=color,width=width,joint='curve')
    x,y=pts[-1]; x0,y0=pts[-2]; a=math.atan2(y-y0,x-x0)
    draw.polygon([(x,y),(x-head*math.cos(a-.45),y-head*math.sin(a-.45)),
                  (x-head*math.cos(a+.45),y-head*math.sin(a+.45))],fill=color)

class Scene:
    """One scene shared by the raster export and editable Excalidraw JSON."""
    def __init__(self): self.elements=[]; self.image=Image.new('RGB',(W,H),BG); self.d=ImageDraw.Draw(self.image)
    def base(self,t,x,y,w,h,color=INK,bg='transparent',sw=2):
        i=len(self.elements); e=dict(id=f'snackwell-{i:04}',type=t,x=x,y=y,width=w,height=h,angle=0,
            strokeColor=color,backgroundColor=bg,fillStyle='solid',strokeWidth=sw,strokeStyle='solid',
            roughness=0,opacity=100,groupIds=[],frameId=None,roundness=None,seed=10000+i,
            version=1,versionNonce=50000+i,isDeleted=False,boundElements=None,updated=1,
            link=None,locked=False)
        self.elements.append(e); return e
    def text(self,x,y,s,size=28,color=INK,bold=False,center=False,mono=False):
        # Excalidraw uses Arial/Helvetica (fontFamily 2) or monospace (3).
        f=font(size,bold,mono); lines=s.split('\n'); widths=[f.getlength(a) for a in lines]
        width=max(widths)+2; height=size*1.25*len(lines)
        if center: x-=width/2
        e=self.base('text',x,y,width,height,color,sw=1)
        e.update(text=s,originalText=s,fontSize=size,fontFamily=3 if mono else 2,
            textAlign='left',verticalAlign='top',containerId=None,autoResize=True,lineHeight=1.25)
        label(self.d,(x,y),s,size,color,bold,mono=mono)
        return e
    def rect(self,x,y,w,h,color=INK,fill=BG,sw=2,dash=False):
        e=self.base('rectangle',x,y,w,h,color,fill,sw);e['roundness']={'type':3}
        if dash:e['strokeStyle']='dashed'
        rounded(self.d,(x,y,x+w,y+h),fill,color,sw)
        return e
    def line(self,pts,color=INK,sw=3,head=True):
        x,y=pts[0]; coords=[[a-x,b-y] for a,b in pts]
        e=self.base('arrow' if head else 'line',x,y,max(p[0] for p in coords)-min(p[0] for p in coords),
            max(p[1] for p in coords)-min(p[1] for p in coords),color,sw=sw)
        e.update(points=coords,startBinding=None,endBinding=None,startArrowhead=None,
            endArrowhead='arrow' if head else None,lastCommittedPoint=None)
        if head: arrow(self.d,pts,color,sw)
        else:self.d.line(pts,fill=color,width=sw)
    def save(self,name):
        self.image.save(OUT/f'{name}.png')
        (OUT/f'{name}.excalidraw').write_text(json.dumps(dict(type='excalidraw',version=2,
            source='https://excalidraw.com',elements=self.elements,
            appState={'viewBackgroundColor':BG,'gridSize':None},files={}),indent=2)+'\n')
    def heading(self,title,subtitle):
        self.text(70,44,'SNACKWELL  /  BUILD LOG 03',22,MUTED,bold=True)
        self.text(70,91,title,55,bold=True)
        self.text(72,162,subtitle,27,MUTED)
    def footer(self,right):
        self.line([(70,931),(1530,931)],LINE,1,False)
        self.text(70,950,'@satvikgari  ·  SystemVerilog',20,MUTED)
        self.text(1010,950,right,20,MUTED)

# 1. Integration overview.
s=Scene();s.heading('Snackwell can do matrix math now.', 'FP32 execution + a 4×4 systolic array, connected to the SM.')
s.rect(70,240,380,95,INK,'#f5f5f1');s.text(260,259,'4 resident warps',31,bold=True,center=True);s.text(260,298,'PC · active mask · registers',22,MUTED,center=True)
s.line([(450,288),(535,288)],INK)
s.rect(535,240,405,95,INK,'#f5f5f1');s.text(737,259,'Scheduler + issue checks',29,bold=True,center=True);s.text(737,299,'Dependencies + unit availability',22,MUTED,center=True)
s.line([(940,288),(1015,288)],INK)
s.rect(1015,240,515,95,INK,'#f5f5f1');s.text(1272,259,'Register file / operand gather',29,bold=True,center=True);s.text(1272,299,'4 lanes · 16 registers per lane',22,MUTED,center=True)
xs=[70,435,800,1165];width=335
for i,(title,sub,c,fill) in enumerate([
    ('Integer SIMD','ADD · SUB · AND · OR · XOR',INK,'#f5f5f1'),
    ('FP32 SIMD','FMUL · FADD · FSUB',BLUE,BLUE_LIGHT),
    ('Tensor unit','TMUL · 16 FP32 MAC PEs',PURPLE,PURPLE_LIGHT),
    ('Shared-memory LSU','Per-lane loads + stores',ORANGE,ORANGE_LIGHT)]):
    s.rect(xs[i],465,width,155,c,fill)
    s.text(xs[i]+width/2,487,title,30,c,bold=True,center=True)
    s.text(xs[i]+width/2,533,sub,21,c,center=True)
    s.text(xs[i]+width/2,572,'4×4 tile · one job at a time' if i==2 else 'Four SIMD lanes' if i<2 else '4 KB · four banks',21,MUTED,center=True)
# Operand bus from RF, labeled and routed clear of boxes.
s.line([(1272,335),(1272,392)],INK,2,False)
s.line([(237,392),(1332.5,392)],INK,2,False)
for x in xs:s.line([(x+width/2,392),(x+width/2,465)],INK,2)
s.text(77,355,'OPERANDS',19,MUTED,bold=True)
# Completion paths, merging before WB.
for x in xs:s.line([(x+width/2,620),(x+width/2,697)],INK,2,False)
s.line([(237,697),(1332,697)],INK,2,False);s.line([(800,697),(800,749)],INK,2)
s.rect(320,749,960,115,INK,'#f5f5f1');s.text(800,769,'Shared writeback → register file',32,bold=True,center=True)
s.text(800,817,'Saved warp + destination + mask · tensor rows stay reserved until final commit',23,MUTED,center=True)
s.text(800,891,'After tensor operand gathering, independent integer and FP work can continue.',25,MUTED,center=True)
s.footer('Implemented + tested in simulation');s.save('01-sm-architecture')

# 2. One processing element, non-fused feedback path.
s=Scene();s.heading('One PE. One output element.', '16 copies work together. Each PE keeps its own running sum.')
s.rect(200,276,1200,545,PURPLE,PURPLE_LIGHT)
s.text(255,305,'PE [row, column]',28,PURPLE,bold=True)
s.text(88,436,'A',44,BLUE,bold=True);s.line([(140,458),(494,458)],BLUE,4)
s.text(782,218,'B',44,ORANGE,bold=True);s.line([(800,265),(800,381),(594,381),(594,409)],ORANGE,4)
s.rect(494,409,250,120,BLUE,BLUE_LIGHT);s.text(619,431,'FP32 multiply',28,BLUE,bold=True,center=True);s.text(619,475,'product = A × B',24,BLUE,center=True)
s.line([(744,470),(876,470)],INK,3)
s.rect(876,409,250,120,PURPLE,BG);s.text(1001,431,'FP32 add',29,PURPLE,bold=True,center=True);s.text(1001,475,'C + product',25,PURPLE,center=True)
s.line([(1126,470),(1230,470),(1230,641),(1085,641)],PURPLE,3)
s.rect(746,585,339,110,GREEN,GREEN_LIGHT);s.text(915,605,'Accumulator C[r,c]',27,GREEN,bold=True,center=True);s.text(915,649,'Starts at +0',23,GREEN,center=True)
s.line([(746,641),(667,641),(667,563),(1001,563),(1001,529)],PURPLE,3)
s.text(265,716,'Four matching A/B pairs arrive, in k order: 0 → 1 → 2 → 3',29,bold=True)
s.text(265,761,'Round after multiply, then round after add. This MAC is non-fused.',24,MUTED)
s.line([(235,458),(235,858),(1450,858)],BLUE,3);s.text(360,869,'A forwarded right',23,BLUE)
s.line([(800,285),(1460,285),(1460,888)],ORANGE,3);s.text(1130,889,'B forwarded down',23,ORANGE)
s.footer('A moves →    B moves ↓    C stays');s.save('02-processing-element')

# 3. Register mapping and writeback lifecycle.
s=Scene();s.heading('A matrix instruction inside the SM.', 'TMUL R8, R0, R4    →    C = A × B')
for x,base,name,color,fill in [(75,0,'A',BLUE,BLUE_LIGHT),(595,4,'B',ORANGE,ORANGE_LIGHT),(1115,8,'C',PURPLE,PURPLE_LIGHT)]:
    s.text(x,246,f'{name} tile',36,color,bold=True)
    for c in range(4):s.text(x+140+c*82,300,f'L{c}',21,MUTED,center=True)
    for r in range(4):
        s.text(x,355+r*69,f'R{base+r}',26,bold=True)
        for c in range(4):
            bx=x+100+c*82;s.rect(bx,342+r*69,75,58,color,fill,1)
            s.text(bx+37,358+r*69,f'{r},{c}',24,color,center=True)
    s.text(x,641,'One register = one row',23,MUTED)
s.text(555,449,'×',40,MUTED,center=True);s.text(1075,449,'=',40,MUTED,center=True)
steps=[('1  GATHER','Snapshot both tiles','4 register-read cycles'),('2  EXECUTE','Feed the systolic array','A →  B ↓  C stays'),('3  WRITE BACK','Commit four result rows','Release the tile after the last row')]
for i,(title,sub,note) in enumerate(steps):
    x=75+i*520;s.rect(x,735,455,139,INK,'#f5f5f1')
    s.text(x+22,750,title,25,bold=True);s.text(x+22,791,sub,25);s.text(x+22,836,note,20,MUTED)
    if i<2:s.line([(x+455,803),(x+505,803)],INK,2)
s.footer('Lane mask selects output columns');s.save('03-tmul-registers')

# 4. Real regression output, typeset without fabricated terminal chrome.
raw=(ROOT/'source/regression.txt').read_text(); passed=[line for line in raw.splitlines() if line.startswith('PASS ')]
assert len(passed)==35 and '35/35 testbenches passed.' in raw
meta=json.loads((ROOT/'source/regression-metadata.json').read_text());assert meta['exit_code']==0
im=Image.new('RGB',(W,H),'#111821');d=ImageDraw.Draw(im)
label(d,(70,48),'SNACKWELL  /  SIMULATION REGRESSION',24,'#a6b1c0',True)
label(d,(70,99),'35 / 35 testbenches passed',55,'#e7f5ff',True)
label(d,(73,176),'$ python3 scripts/run_tests.py',28,'#83d5a4',mono=True)
d.line([(70,234),(1530,234)],fill='#354151',width=2)
for i,line in enumerate(passed):
    col=i//18;row=i%18;x=70+col*760;y=260+row*30
    label(d,(x,y),'PASS',21,'#83d5a4',mono=True)
    label(d,(x+73,y),line[5:],21,'#e0e5eb',mono=True)
d.line([(70,828),(1530,828)],fill='#354151',width=2)
label(d,(70,856),'185 full-SM matrix cases · 2,960 checked output elements',30,'#e7f5ff',True)
label(d,(70,901),'RAW/WAW hazards · all 16 masks · four warps · divergence · reset · backpressure',23,'#a6b1c0')
label(d,(70,951),f"Captured run: {meta['timestamp'][:10]} UTC · Icarus Verilog · simulation only",20,'#a6b1c0')
im.save(OUT/'05-test-results.png')

# Exact toy matrices. All values/products/sums are small integers representable
# exactly in FP32. Valid zero is included intentionally.
A=[[1,2,3,4],[2,1,0,1],[0,1,2,3],[1,0,1,2]]
B=[[1,0,2,1],[0,1,1,2],[2,1,0,1],[1,2,1,0]]
C=[[sum(A[r][k]*B[k][c] for k in range(4)) for c in range(4)] for r in range(4)]
# Confirm the animation's arithmetic against the project's independent oracle.
sys.path.insert(0,str(ROOT.parents[2]))
from snackwell.reference.tensor import matmul
bits=lambda x:struct.unpack('>I',struct.pack('>f',float(x)))[0]
assert matmul([bits(x) for row in A for x in row],[bits(x) for row in B for x in row])==[bits(x) for row in C for x in row]
( ROOT/'source/animation-data.json').write_text(json.dumps({'A':A,'B':B,'C':C,'logical_steps':10,'valid_MACs':64},indent=2)+'\n')

# 4x4 array: generously separated paths and cells for mobile viewing.
X,Y=382,355;DX,DY=204,118;PW,PH=145,84
base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(base)
label(d,(70,44),'SNACKWELL  /  TENSOR DATAFLOW',22,MUTED,True)
label(d,(70,91),'A moves right. B moves down. C stays.',48,INK,True)
label(d,(73,161),'A 4×4 output-stationary systolic array · 16 processing elements',27,MUTED)
# A/B legend includes direction, and all cells stay visible throughout.
label(d,(73,241),'A operands →',28,BLUE,True)
label(d,(382,241),'B operands ↓',28,ORANGE,True)
label(d,(1215,241),'OUTPUT C',24,GREEN,True)

for r in range(4):
    for c in range(4):
        cx=X+c*DX;cy=Y+r*DY
        arrow(d,[(cx-60,cy+56),(cx,cy+56)],BLUE,2,8)
        arrow(d,[(cx+PW-15,cy-34),(cx+PW-15,cy)],ORANGE,2,8)
        if c==3: arrow(d,[(cx+PW,cy+56),(cx+PW+30,cy+56)],BLUE,2,8)
        if r==3: arrow(d,[(cx+PW-15,cy+PH),(cx+PW-15,cy+PH+23)],ORANGE,2,8)
# Legend: one value token per active link. Source input tables small but useful.
label(d,(72,340),'A',27,BLUE,True)
for r,row in enumerate(A): label(d,(72,389+r*48),'  '.join(map(str,row)),25,BLUE,mono=True)
label(d,(72,637),'B',27,ORANGE,True)
for r,row in enumerate(B): label(d,(72,681+r*39),'  '.join(map(str,row)),24,ORANGE,mono=True)
d.line([(70,926),(1530,926)],fill=LINE,width=1)
label(d,(70,948),'Logical steps, not clock cycles. Each step waits for every PE to finish.',22,MUTED)

# Neutral skeleton is also exported as an editable Excalidraw overview.
s=Scene();s.heading('A moves right. B moves down. C stays.', '4×4 systolic array · 16 PEs · 64 multiply-accumulates per tile')
for r in range(4):
    for c in range(4):
        cx=360+c*230;cy=350+r*135
        s.line([(cx-67,cy+42),(cx,cy+42)],BLUE,3)
        s.line([(cx+115,cy-49),(cx+115,cy)],ORANGE,3)
        s.rect(cx,cy,180,95,PURPLE,PURPLE_LIGHT)
        s.text(cx+90,cy+13,f'PE {r},{c}',23,PURPLE,center=True)
        s.text(cx+90,cy+47,f'C[{r},{c}]',30,PURPLE,bold=True,center=True)
s.text(87,369,'A rows →',29,BLUE,bold=True);s.text(645,270,'B columns ↓',29,ORANGE,bold=True)
s.text(790,873,'Each PE accumulates four products; the result stays local.',27,MUTED,center=True)
s.footer('10 logical steps · variable latency');s.save('04-systolic-array')

def render(step,phase,final=False):
    """Hold tokens for arithmetic, then advance both streams in lockstep."""
    im=base.copy();d=ImageDraw.Draw(im)
    done=final
    label(d,(1215,280),'Final result' if done else 'Partial sums',23,MUTED)
    label(d,(1215,341),'COMPLETE' if done else f'STEP {step} / 9',26,GREEN if done else PURPLE,True)
    phase_text='Result tile ready' if done else 'Multiply, then add' if phase<.72 else 'All PEs done → advance'
    label(d,(382,839),phase_text,27,INK,True)
    total_macs=0
    for r in range(4):
        for c in range(4):
            cx=X+c*DX;cy=Y+r*DY;k=step-r-c
            valid=0<=k<4 and not done
            include=k+1 if phase>=.72 else k
            n=4 if done else min(4,max(0,include))
            val=sum(A[r][j]*B[j][c] for j in range(n))
            total_macs+=n
            complete=n==4
            fill=GREEN_LIGHT if complete else PURPLE_LIGHT if valid else '#f5f5f1'
            color=GREEN if complete else PURPLE if valid else LINE
            rounded(d,(cx,cy,cx+PW,cy+PH),fill,color,3 if valid else 2,12)
            label(d,(cx+11,cy+7),f'C{r}{c}',18,MUTED)
            label(d,(cx+PW/2,cy+34),str(val),32,GREEN if complete else INK,True,anchor='mt')
            if valid:
                move=max(0,(phase-.72)/.28)
                move=move*move*(3-2*move)
                ax=cx+15+(DX if c<3 else PW+10)*move;ay=cy+56
                bx=cx+PW-15;by=cy+56+(DY if r<3 else PH-25)*move
                for tx,ty,value,fc,bc in [(ax,ay,A[r][k],BLUE,BLUE_LIGHT),(bx,by,B[k][c],ORANGE,ORANGE_LIGHT)]:
                    d.ellipse((tx-14,ty-14,tx+14,ty+14),fill=bc,outline=fc,width=2)
                    label(d,(tx,ty),str(value),18,fc,True,anchor='mm')
                if phase<.72:
                    label(d,(cx+PW-8,cy+7),f'k={k}',17,PURPLE,anchor='rt')
            gx=1245+c*71;gy=419+r*72
            rounded(d,(gx-27,gy-25,gx+29,gy+25),GREEN_LIGHT if complete else BG,GREEN if complete else LINE,1,6)
            label(d,(gx,gy),str(val),25,GREEN if complete else INK,True,anchor='mm')
    if phase>=.72 and not done:
        move=(phase-.72)/.28;move=move*move*(3-2*move)
        # Stagger the new edge tokens as the previous ones move to neighbors.
        for i in range(4):
            nk=step+1-i
            if 0<=nk<4:
                for tx,ty,v,fc,bc in [
                    (X-DX+15+DX*move,Y+i*DY+56,A[i][nk],BLUE,BLUE_LIGHT),
                    (X+i*DX+PW-15,Y-DY+56+DY*move,B[nk][i],ORANGE,ORANGE_LIGHT)]:
                    d.ellipse((tx-14,ty-14,tx+14,ty+14),fill=bc,outline=fc,width=2)
                    label(d,(tx,ty),str(v),18,fc,True,anchor='mm')
    label(d,(1215,746),f'{total_macs} / 64 MACs',26,GREEN if done else INK,True)
    label(d,(1215,791),'C = A × B',29,GREEN,True)
    # Step markers make fill/drain visible without pretending one step=one cycle.
    for t in range(10):
        xx=387+t*73
        d.ellipse((xx,889,xx+12,901),fill=PURPLE if t<=step or done else LINE)
    label(d,(1215,868),'FP32 · non-fused MAC',20,MUTED)
    return im

def render_social(step,phase,final=False):
    im=Image.new('RGB',(1080,1080),BG);d=ImageDraw.Draw(im)
    label(d,(48,33),'SNACKWELL  /  BUILD LOG 03',21,MUTED,True)
    label(d,(48,78),'Inside the 4×4 tensor array',48,INK,True)
    label(d,(48,155),'A moves right →',29,BLUE,True)
    label(d,(567,155),'B moves down ↓',29,ORANGE,True)
    label(d,(48,209),'Each PE keeps one output value.',29,INK)
    macs=sum(min(4,max(0,step-r-c+(1 if phase>=.72 else 0))) for r in range(4) for c in range(4))
    if final:macs=64
    label(d,(48,268),'RESULT READY' if final else f'LOGICAL STEP {step} / 9',23,GREEN if final else PURPLE,True)
    label(d,(780,268),f'{macs} / 64 MACs',23,GREEN if final else MUTED,True)
    core=render(step,phase,final).crop((310,278,1190,825)).resize((1000,622),Image.Resampling.LANCZOS)
    im.paste(core,(40,308))
    label(d,(48,953),'C = A × B' if final else 'Multiply, then add' if phase<.72 else 'All PEs done → advance',33,GREEN if final else INK,True)
    label(d,(48,1015),'10 logical steps · variable clock latency · non-fused FP32',23,MUTED)
    return im

if '--static-only' not in sys.argv:
    FPS=24; INTRO=1.5; STEP_TIME=1.4; OUTRO=3.0
    frames=int((INTRO+10*STEP_TIME+OUTRO)*FPS)
    frame_dir=Path('/private/tmp/snackwell-video-frames');frame_dir.mkdir(exist_ok=True)
    gif=[]
    for f in range(frames):
        time=f/FPS
        if time<INTRO: im=render_social(0,0)
        elif time<INTRO+10*STEP_TIME:
            logical=(time-INTRO)/STEP_TIME;step=min(9,int(logical));phase=logical-step
            im=render_social(step,phase)
        else: im=render_social(9,1,True)
        im.save(frame_dir/f'{f:05}.png',compress_level=2)
        if f%2==0:
            small=im.resize((720,720),Image.Resampling.LANCZOS)
            gif.append(small.quantize(colors=128,method=Image.Quantize.MEDIANCUT))
    gif[0].save(OUT/'tensor-flow.gif',save_all=True,append_images=gif[1:],duration=83,loop=0,optimize=False,disposal=2)
    render_social(3,.5).save(OUT/'tensor-flow-poster.png')
    render_social(0,.8).save(QA/'animation-check-step0.png')
    render_social(4,.8).save(QA/'animation-check-step4.png')
    render_social(9,1,True).save(QA/'animation-check-final.png')
    subprocess.run(['/private/tmp/snackwell-encode-video',str(frame_dir),str(OUT/'tensor-flow.mp4'),'1080','1080','24'],check=True)
print('Media written to',OUT)
print('C =',C)
