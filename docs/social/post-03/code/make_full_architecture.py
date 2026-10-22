"""Render the implemented Snackwell SM as editable Excalidraw, SVG, and PNG.

Run with Python 3 + Pillow. Fonts use the macOS Arial installation, matching
Excalidraw's Helvetica/Arial family. No external services are used.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import math
import re
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[2]
W, H = 3820, 2810
BG = '#fffefa'
INK = '#242830'
MUTED = '#626975'
LINE = '#bdc4cc'
BLUE, BL = '#1971c2', '#e7f5ff'
PURPLE, PL = '#7048c6', '#f3f0ff'
ORANGE, OL = '#c56a0a', '#fff4e6'
GREEN, GL = '#2b8a3e', '#ebfbee'
TEAL, TL = '#087f8c', '#edf9f8'
GRAY = '#f5f5f2'
FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
fonts = {}


def font(size):
    if size not in fonts:
        fonts[size] = ImageFont.truetype(FONT, size)
    return fonts[size]


class Diagram:
    def __init__(self):
        self.elements = []
        self.nodes = {}
        self.connections = []
        self.containment = []

    def base(self, typ, ident, x, y, w, h, color=INK, fill='transparent', sw=3, group=None):
        n = len(self.elements)
        e = dict(id=ident, type=typ, x=x, y=y, width=w, height=h, angle=0,
                 strokeColor=color, backgroundColor=fill, fillStyle='solid',
                 strokeWidth=sw, strokeStyle='solid', roughness=0, opacity=100,
                 groupIds=[group] if group else [], frameId=None, roundness=None,
                 seed=18000+n, version=1, versionNonce=48000+n, isDeleted=False,
                 boundElements=[], updated=1, link=None, locked=False)
        self.elements.append(e)
        return e

    def text(self, x, y, s, size=29, color=INK, ident=None, group=None, parent=None):
        ident = ident or f'text-{len(self.elements)}'
        lines = s.split('\n')
        width = max(font(size).getlength(line) for line in lines) + 3
        height = size*1.30*len(lines)
        e = self.base('text', ident, x, y, width, height, color, sw=1, group=group)
        e.update(text=s, originalText=s, fontSize=size, fontFamily=2,
                 textAlign='left', verticalAlign='top', containerId=None,
                 autoResize=True, lineHeight=1.30)
        if parent:
            self.containment.append((ident, parent))
        return e

    def rect(self, ident, x, y, w, h, color=INK, fill=GRAY, title=None,
             body=None, module=None, group=False, size=35, body_size=28):
        gid = ident if group else None
        e = self.base('rectangle', ident, x, y, w, h, color, fill, group=gid)
        e['roundness'] = {'type': 3}
        e['customData'] = {'role': 'container' if group else 'block'}
        if module:
            e['customData']['rtl'] = module
        self.nodes[ident] = e
        if title:
            self.text(x+26, y+22, title, size, color, group=gid, parent=ident)
        if body:
            self.text(x+26, y+(66 if h <= 125 else 77), body, body_size, MUTED, group=gid, parent=ident)
        return e

    def edge(self, ident, pts, color=INK, dashed=False, start=None, end=None,
             both=False, head=True, width=3):
        xs, ys = zip(*pts)
        x, y = pts[0]
        e = self.base('arrow' if head else 'line', ident, x, y,
                      max(xs)-min(xs), max(ys)-min(ys), color, sw=width)
        e.update(points=[[a-x, b-y] for a, b in pts],
                 startBinding=None, endBinding=None,
                 startArrowhead='arrow' if both else None,
                 endArrowhead='arrow' if head else None,
                 lastCommittedPoint=None)
        if dashed:
            e['strokeStyle'] = 'dashed'
        for field, node_id in [('startBinding', start), ('endBinding', end)]:
            if node_id:
                e[field] = {'elementId': node_id, 'focus': 0, 'gap': 0}
                self.nodes[node_id]['boundElements'].append({'id': ident, 'type': 'arrow'})
        self.connections.append({'id': ident, 'start': start, 'end': end})
        return e

    def dot(self, x, y, color=INK):
        self.base('ellipse', f'junction-{len(self.elements)}', x-5, y-5, 10, 10, color, color, 1)

    def save(self):
        # Group each block with its text, and nest arithmetic blocks inside their
        # execution-unit group. The components remain individually editable.
        parents = dict(self.containment)
        containers = [e for e in self.nodes.values() if e['customData']['role']=='container']
        for e in self.elements:
            target = self.nodes.get(parents.get(e['id'],e['id']))
            if target:
                e['groupIds'] = ['group-'+target['id']]
                for c in containers:
                    if c is not target and c['x'] <= target['x'] and c['y'] <= target['y'] and \
                       target['x']+target['width'] <= c['x']+c['width'] and \
                       target['y']+target['height'] <= c['y']+c['height']:
                        e['groupIds'].append('group-'+c['id'])
            elif e['type'] in ('arrow','line'):
                pts=[(e['x']+a,e['y']+b) for a,b in e['points']]
                for c in containers:
                    if all(c['x'] <= a <= c['x']+c['width'] and c['y'] <= b <= c['y']+c['height'] for a,b in pts):
                        e['groupIds']=['group-'+c['id']]
        # Keep container backgrounds behind internal links and components.
        layers = [[], [], [], [], []]
        for e in self.elements:
            i = 0 if e.get('customData', {}).get('role') == 'container' else \
                1 if e['type'] in ('arrow', 'line') else \
                4 if e['type'] == 'text' else 3 if e['type'] == 'ellipse' else 2
            layers[i].append(e)
        ordered = sum(layers, [])
        ids = {e['id'] for e in ordered}
        assert len(ids) == len(ordered)
        for e in ordered:
            for key in ('startBinding', 'endBinding'):
                if e.get(key):
                    assert e[key]['elementId'] in ids
        by_id = {e['id']: e for e in ordered}
        for tid, pid in self.containment:
            t, p = by_id[tid], by_id[pid]
            assert t['x'] >= p['x'] and t['y'] >= p['y'], (tid, pid)
            assert t['x']+t['width'] <= p['x']+p['width']-15, (t['text'], 'too wide', p['id'])
            assert t['y']+t['height'] <= p['y']+p['height']-9, (t['text'], 'too tall', p['id'])
        payload = dict(type='excalidraw', version=2, source='https://excalidraw.com',
                       elements=ordered, appState={'viewBackgroundColor': BG, 'gridSize': None}, files={})
        (OUT/'snackwell-full-architecture.excalidraw').write_text(json.dumps(payload, indent=2)+'\n')
        self.render(ordered)
        (OUT/'code/diagram-manifest.json').write_text(json.dumps({
            'canvas': [W,H], 'elements': len(ordered),
            'blocks': {k: v.get('customData', {}) for k,v in self.nodes.items()},
            'connections': self.connections,
        }, indent=2)+'\n')

    def render(self, elements):
        im = Image.new('RGB', (W,H), BG)
        d = ImageDraw.Draw(im)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
               f'<rect width="{W}" height="{H}" fill="{BG}"/>']

        def seg(a,b,color,width,dashed):
            if not dashed:
                d.line([a,b], fill=color, width=width)
                return
            length = math.dist(a,b)
            for pos in range(0, math.ceil(length), 19):
                t0, t1 = pos/max(1,length), min(length,pos+10)/max(1,length)
                p = (a[0]+(b[0]-a[0])*t0,a[1]+(b[1]-a[1])*t0)
                q = (a[0]+(b[0]-a[0])*t1,a[1]+(b[1]-a[1])*t1)
                d.line([p,q], fill=color, width=width)

        def arrowhead(tip, prev, color, width):
            angle = math.atan2(tip[1]-prev[1],tip[0]-prev[0])
            wings = [(tip[0]-17*math.cos(angle+s*.43), tip[1]-17*math.sin(angle+s*.43)) for s in [-1,1]]
            d.line([wings[0],tip,wings[1]], fill=color, width=width)
            svg.append(f'<polyline points="{wings[0][0]},{wings[0][1]} {tip[0]},{tip[1]} {wings[1][0]},{wings[1][1]}" fill="none" stroke="{color}" stroke-width="{width}"/>')

        for e in elements:
            x,y,w,h = [e[k] for k in ('x','y','width','height')]
            c, fill, sw = e['strokeColor'], e['backgroundColor'], e['strokeWidth']
            if e['type'] == 'rectangle':
                d.rounded_rectangle((x,y,x+w,y+h), radius=28, fill=fill, outline=c, width=sw)
                svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{fill}" stroke="{c}" stroke-width="{sw}"/>')
            elif e['type'] == 'ellipse':
                d.ellipse((x,y,x+w,y+h), fill=c)
                svg.append(f'<circle cx="{x+w/2}" cy="{y+h/2}" r="{w/2}" fill="{c}"/>')
            elif e['type'] == 'text':
                sz = e['fontSize']
                for n, line in enumerate(e['text'].split('\n')):
                    yy = y+n*sz*1.30
                    d.text((x,yy),line,font=font(sz),fill=c,anchor='lt')
                    svg.append(f'<text x="{x}" y="{yy}" dominant-baseline="text-before-edge" font-family="Arial,Helvetica,sans-serif" font-size="{sz}" fill="{c}">{escape(line)}</text>')
            elif e['type'] in ('arrow','line'):
                pts = [(x+a,y+b) for a,b in e['points']]
                dashed = e['strokeStyle']=='dashed'
                for a,b in zip(pts,pts[1:]):
                    seg(a,b,c,sw,dashed)
                dash = ' stroke-dasharray="10 9"' if dashed else ''
                svg.append(f'<polyline points="'+ ' '.join(f'{a},{b}' for a,b in pts) +f'" fill="none" stroke="{c}" stroke-width="{sw}"{dash}/>')
                if e.get('endArrowhead'):
                    arrowhead(pts[-1],pts[-2],c,sw)
                if e.get('startArrowhead'):
                    arrowhead(pts[0],pts[1],c,sw)
        svg.append('</svg>')
        im.save(OUT/'snackwell-full-architecture.png')
        im.resize((1910,1405),Image.Resampling.LANCZOS).save(OUT/'code/review-preview.png')
        (OUT/'snackwell-full-architecture.svg').write_text('\n'.join(svg))


s = Diagram()
s.text(100,52,'Snackwell — full SM architecture',77)
s.text(104,153,'Current implemented RTL  ·  4 resident warps  ·  4 lanes per warp  ·  32-bit instructions and data',32,MUTED)
s.rect('interfaces',100,230,3620,110,LINE,BG)
s.text(126,248,'Inputs: clock / reset, run enable, warp launch (ID, PC, mask), debug register preload',29)
s.text(126,290,'Status: issue / stalls, warp state, register operands / ALU result, bank conflicts, branch masks and reconvergence',27,MUTED)

# Selection, instruction fetch, decode, and issue.
s.rect('state',100,460,570,200,TEAL,TL,'Warp state table',
       '4 × PC, active mask\nValid / ready / halted flags',module='rtl/sm/warp_state_table.sv')
s.rect('scheduler',800,460,520,200,INK,GRAY,'Warp scheduler',
       'Round-robin eligible warps\nSkip blocked / halted warps',module='rtl/sm/warp_scheduler.sv')
s.rect('fetch',1450,460,550,200,INK,GRAY,'Instruction fetch',
       'Shared instruction memory\n16 × 32-bit words; PC[5:2]',module='rtl/frontend/instruction_memory.sv')
s.rect('decode',2130,460,570,200,INK,GRAY,'Decode + opcode control',
       'Opcode, Rd, Rs1, Rs2\n14-bit immediate → 32 bits',module=['rtl/frontend/instruction_decoder.sv','rtl/sm/multiwarp_execution_core.sv'])
s.rect('issue',2850,460,870,240,INK,GRAY,'Issue control + acceptance',
       'Source / destination dependencies + unit availability\nBlock or retry a candidate; issue only when ready\nAccept also advances scheduler and warp control',
       module=['rtl/sm/issue_control.sv','rtl/sm/issue_stage.sv','rtl/sm/warp_control_unit.sv'],body_size=28)
s.edge('launch',[(250,340),(250,460)],TEAL,True,end='state')
s.text(265,406,'launch',23,TEAL)
s.edge('run-control',[(3440,340),(3440,460)],MUTED,True,end='issue')
s.text(3455,409,'run / debug',23,MUTED)
s.edge('eligible',[(670,550),(800,550)],TEAL,True,start='state',end='scheduler')
s.text(688,510,'status',23,TEAL)
s.edge('select-state',[(930,660),(930,727),(585,727),(585,660)],TEAL,True,start='scheduler',end='state')
s.text(615,685,'selected warp ID',24,TEAL)
s.edge('selected-pc',[(440,460),(440,400),(1725,400),(1725,460)],TEAL,True,start='state',end='fetch')
s.text(1110,362,'selected PC',25,TEAL)
s.edge('instruction',[(2000,550),(2130,550)],start='fetch',end='decode')
s.text(2027,510,'32-bit',23,MUTED)
s.edge('decode-issue',[(2700,550),(2850,550)],MUTED,True,start='decode',end='issue')
s.text(2727,510,'fields',23,MUTED)

# Dependency state, physical register storage and core-level resource checks.
s.rect('scoreboard',100,865,570,275,GREEN,GL,'Scoreboard + blocked state',
       '16 busy bits per warp\nCheck reads + destination writes\nMark on issue; clear on commit\nWake / retry blocked warps',
       module=['rtl/sm/scoreboard.sv','rtl/sm/issue_control.sv'],size=33,body_size=27)
s.rect('rf',1040,845,1610,245,INK,GRAY,'Multiwarp register file + read-port selection',
       '4 warps × 4 lanes × 16 registers × 32 bits = 1 KB\nTwo 128-bit reads; one masked 128-bit write\nNormal reads use the selected warp; TMUL owns both read ports for four gather cycles\nDebug preload has priority over architectural writes',
       module=['rtl/memory/multiwarp_register_file.sv','rtl/sm/multiwarp_execution_core.sv'],body_size=27)
s.rect('checks',2850,865,870,240,INK,GRAY,'Unit availability + tensor dependency checks',
       'FP in-flight · tensor ready / gather · writeback space\nCheck all A / B / C rows against pending destinations\nEXIT waits until the warp has no pending writes\nRejoin handling takes priority over the fetched operation',
       module='rtl/sm/multiwarp_execution_core.sv',size=33,body_size=28)
s.edge('scoreboard-issue',[(670,950),(710,950),(710,782),(3040,782),(3040,700)],GREEN,True,start='scoreboard',end='issue',both=True)
s.text(1120,740,'register checks / destination allocation on issue',25,GREEN)
s.edge('checks-issue',[(3320,865),(3320,700)],MUTED,True,start='checks',end='issue')
s.text(3340,797,'ready + tile dependencies',24,MUTED)
s.edge('read-selectors',[(2415,660),(2415,845)],MUTED,True,start='decode',end='rf')
s.text(2435,705,'read addresses',23,MUTED)
s.edge('rf-warp-select',[(1070,660),(1070,690),(1385,690),(1385,845)],TEAL,True,start='scheduler',end='rf')
s.text(1405,704,'warp select',23,TEAL)
s.edge('debug-preload',[(2770,340),(2770,920),(2650,920)],MUTED,True,end='rf')
s.text(2657,944,'debug',23,MUTED)

# Common functional dispatch/data paths. This is not a pipeline register drawing.
s.edge('dispatch',[(2850,640),(2800,640),(2800,1205)],MUTED,True,start='issue',head=False)
s.edge('rf-operands',[(1845,1090),(1845,1205)],start='rf',head=False)
s.edge('operand-bus',[(385,1205),(3445,1205)],head=False)
s.text(1060,1157,'OPERANDS + ISSUED OPERATION  ·  opcode, warp, active mask, PC and immediate',28,MUTED)
for ident,x in [('integer',385),('fp32',1100),('tensor',1930),('lsu',2750),('branch',3445)]:
    s.edge('dispatch-'+ident,[(x,1205),(x,1290)])
    s.dot(x,1205)
s.dot(1845,1205)
s.dot(2800,1205)

# Arithmetic execution units.
s.rect('integer',100,1290,570,440,INK,GRAY,'Integer SIMD',
       'Four parallel 32-bit lanes',module='rtl/execution/simd_integer_execute.sv',group=True)
s.rect('integer-lanes',130,1445,510,170,INK,BG,'4 × integer ALU',
       'ADD · SUB · AND · OR · XOR\nCombinational lane results',module='rtl/execution/integer_alu.sv',size=31,body_size=27)
s.text(126,1660,'Active mask applied at writeback',27,MUTED,parent='integer')

s.rect('fp32',800,1290,600,570,BLUE,BL,'FP32 SIMD',
       'FMUL · FADD · FSUB\nFour lanes, each containing:',module='rtl/execution/fp32/simd_fp32_execute.sv',group=True)
s.rect('fp-mul',830,1460,255,155,BLUE,BG,'Multiply',
       'FP32 wrapper\n+ multiplier',module=['rtl/execution/fp32/fp32_mul_wrapper.sv','rtl/execution/fp32/multiplier.v'],size=30,body_size=25)
s.rect('fp-add',1115,1460,255,155,BLUE,BG,'Add / subtract',
       'Iterative FP32\nnormalize / round',module='rtl/execution/fp32/fp32_addsub.sv',size=29,body_size=24)
s.rect('fp-completion',830,1670,540,125,BLUE,BG,'Lane completion + saved tag',
       'Wait for active lanes; retain result + context',module='rtl/sm/multiwarp_execution_core.sv',size=29,body_size=24)
s.edge('fp-mul-result',[(955,1615),(955,1670)],BLUE,start='fp-mul',end='fp-completion')
s.edge('fp-add-result',[(1240,1615),(1240,1670)],BLUE,start='fp-add',end='fp-completion')
s.text(826,1812,'One SIMD FP instruction in flight',27,BLUE,parent='fp32')

# Complete TMUL hierarchy, with tile gathering and output packing in the SM adapter.
s.rect('tensor',1530,1290,800,830,PURPLE,PL,'Tensor execution · TMUL',
       'Fixed 4×4 FP32 · one tensor job in flight',module='rtl/sm/sm_tensor_unit.sv',group=True)
s.rect('tensor-adapter',1560,1430,710,190,PURPLE,BG,'SM tensor adapter',
       'Gather A + B in 4 register-read cycles\nSave warp / mask; reserve 4 destination rows\nPack the 16 output words into 4 vector writes',
       module='rtl/sm/sm_tensor_unit.sv',size=32,body_size=26)
s.rect('tensor-interface',1560,1670,710,110,PURPLE,BG,'Tile buffers + word interface',
       'Load 32 input words; stream 16 result words',module='rtl/execution/tensor/tensor_core.sv',size=29,body_size=26)
s.rect('array',1560,1830,710,220,PURPLE,BG,'4×4 systolic array · 16 FP32 PEs',
       'Feeder staggers the row / column inputs\nEach PE: multiply → add → local C register\nA moves right; B moves down; C stays\nAdvance only when every PE finishes',
       module=['rtl/execution/tensor/systolic_array_4x4.sv','rtl/execution/tensor/systolic_feeder_4x4.sv','rtl/execution/tensor/fp32_mac_pe.sv'],size=31,body_size=25)
s.edge('load-words',[(1910,1620),(1910,1670)],PURPLE,start='tensor-adapter',end='tensor-interface')
s.edge('launch-array',[(1910,1780),(1910,1830)],PURPLE,start='tensor-interface',end='array')
s.edge('array-results',[(2270,2005),(2300,2005),(2300,1740),(2270,1740)],PURPLE,start='array',end='tensor-interface')
s.edge('tensor-results',[(2270,1710),(2300,1710),(2300,1550),(2270,1550)],PURPLE,start='tensor-interface',end='tensor-adapter')
s.text(1556,2072,'Non-fused MAC · release destinations on final row commit',25,PURPLE,parent='tensor')

# Memory execution: separate LSU and shared-memory storage.
s.rect('lsu',2460,1290,580,200,ORANGE,OL,'Load / store unit',
       'LD_SHARED · ST_SHARED\nPer-lane base + immediate\nActive load / store masks',module='rtl/memory/load_store_unit.sv',size=34,body_size=27)
s.rect('shared-memory',2460,1580,580,335,ORANGE,OL,'Shared memory',
       '4 KB · 4 banks\n256 × 32-bit words per bank\nBank = address[3:2]\nRow = address[11:4]\nBank-conflict detection\nConflicting lanes suppressed; no replay',
       module='rtl/memory/shared_memory.sv',size=34,body_size=25)
s.edge('memory-request',[(2750,1490),(2750,1580)],ORANGE,start='lsu',end='shared-memory')
s.text(2480,1520,'addresses / store data / lane masks',24,ORANGE)

# Branch and reconvergence subsystem.
s.rect('branch',3170,1290,550,190,TEAL,TL,'Branch unit · BEQ',
       'Compare active lanes\nTaken / not-taken masks\nTarget PC + fallthrough PC',module='rtl/execution/branch_unit.sv',size=32,body_size=26)
s.rect('divergence',3170,1590,550,210,TEAL,TL,'Divergence controller',
       'Choose path + active mask\nSwitch deferred path / rejoin\nHandle rejoin before fetched op',module='rtl/sm/divergence_controller.sv',size=32,body_size=26)
s.rect('target-table',3170,1880,550,150,TEAL,TL,'Reconvergence target table',
       'One target PC per warp\nSSY writes; accepted BEQ clears',module='rtl/sm/reconvergence_target_table.sv',size=30,body_size=24)
s.rect('reconv-stack',3170,2110,550,185,TEAL,TL,'Reconvergence stacks',
       '4 entries per warp\nDeferred + rejoin PC / mask\nPush / pop + pending-path state',module='rtl/sm/reconvergence_stack.sv',size=32,body_size=25)
s.edge('branch-control',[(3445,1480),(3445,1590)],TEAL,start='branch',end='divergence')
s.edge('target-read',[(3170,1955),(3140,1955),(3140,1760),(3170,1760)],TEAL,True,start='target-table',end='divergence')
s.edge('stack-control',[(3720,1740),(3750,1740),(3750,2190),(3720,2190)],TEAL,True,start='divergence',end='reconv-stack',both=True)
s.edge('ssy-save',[(3720,600),(3775,600),(3775,1950),(3720,1950)],TEAL,True,start='issue',end='target-table')
s.edge('rejoin-event',[(3720,1640),(3740,1640),(3740,1135),(3600,1135),(3600,1105)],TEAL,True,start='divergence',end='checks')

# Completions and the one architectural writeback slot.
s.edge('integer-result',[(620,1730),(620,2230)],start='integer',head=False)
s.edge('fp-result',[(1100,1860),(1100,2230)],BLUE,start='fp32',head=False)
s.edge('tensor-row-result',[(1930,2120),(1930,2230)],PURPLE,start='tensor',head=False)
s.edge('load-result',[(2750,1915),(2750,2230)],ORANGE,start='shared-memory',head=False)
s.text(2770,2035,'load data',25,ORANGE)
s.edge('completion-bus',[(620,2230),(2750,2230)],head=False)
for x in [620,1100,1930,2750]:
    s.dot(x,2230)
s.rect('writeback',800,2320,2240,210,GREEN,GL,'Shared writeback arbitration + result register',
       'Priority: completed FP → tensor row → integer result / shared-memory load\nOne 128-bit result + saved warp, destination register and active-lane mask\nCommit when the register port is available; hold pending results during debug writes',
       module='rtl/sm/multiwarp_execution_core.sv',size=38,body_size=29)
s.edge('completion-to-wb',[(1930,2230),(1930,2320)],start=None,end='writeback')
s.text(1290,2260,'FP / tensor hold results; integer / loads retry when writeback is busy',26,GREEN)
s.edge('commit-to-rf',[(2395,2320),(2395,1090)],GREEN,True,start='writeback',end='rf',width=4)
s.text(1780,2555,'REGISTER COMMIT: write active lanes · clear busy state · release the final tensor row',27,GREEN)
s.edge('commit-to-scoreboard',[(800,2450),(730,2450),(730,1040),(670,1040)],GREEN,True,start='writeback',end='scoreboard',width=4)
s.edge('final-row-release',[(2395,2100),(2330,2100)],GREEN,True,end='tensor',width=4)
s.dot(2395,2100,GREEN)

# Unit status feeds core-level availability logic; tensor masks also feed dependencies.
s.edge('fp-status',[(1400,1350),(1460,1350),(1460,1130)],MUTED,True,start='fp32',head=False)
s.edge('tensor-status',[(2330,1350),(2370,1350),(2370,1130)],MUTED,True,start='tensor',head=False)
s.edge('wb-space',[(3040,2440),(3080,2440),(3080,1130)],MUTED,True,start='writeback',head=False)
s.edge('status-bus',[(1460,1130),(3315,1130),(3315,1105)],MUTED,True,end='checks')
for x in [1460,2370,3080]:
    s.dot(x,1130,MUTED)

# Warp state update and issue acceptance feedback (rather than a separate PC register).
s.rect('state-update',100,2320,570,290,TEAL,TL,'Warp state update logic',
       'Accepted instruction: PC + 4\nBranch / rejoin: new PC + mask\nEXIT: halt after pending writes\nLaunch: initialize the warp\nUpdate priority: launch → rejoin → op',
       module='rtl/sm/multiwarp_execution_core.sv',size=32,body_size=26)
s.edge('branch-state-update',[(3170,1685),(3110,1685),(3110,2660),(385,2660),(385,2610)],TEAL,True,start='divergence',end='state-update')
s.text(1525,2615,'branch / reconvergence PC + active-mask updates',27,TEAL)
s.edge('accepted-state-update',[(3630,700),(3630,760),(3795,760),(3795,2720),(230,2720),(230,2610)],MUTED,True,start='issue',end='state-update')
s.text(1525,2682,'accepted instruction / EXIT',25,MUTED)
s.edge('update-state',[(100,2440),(42,2440),(42,600),(100,600)],TEAL,True,start='state-update',end='state',width=4)
s.edge('scheduler-accept',[(2850,680),(2730,680),(2730,817),(1220,817),(1220,660)],MUTED,True,start='issue',end='scheduler')
s.text(1520,788,'accept / blocked masks',22,MUTED)

# Explanatory key in the natural space beneath the integer lane block.
s.text(125,1870,'Reading the connections',31)
s.text(125,1923,'Solid: instructions / operands / results\nDashed: control and availability\nGreen: dependencies / writeback\nDots connect shared paths',25,MUTED)
s.text(125,2084,'Clock / reset apply across the SM.',25,MUTED)
s.text(104,2760,'Functional block diagram of snackwell_sm / multiwarp_execution_core. Connections show data and control flow; they do not imply one pipeline stage per box.',24,MUTED)

# Small whitespace masks keep crossing lines out of connection labels.
for prefix in ['OPERANDS + ISSUED', 'addresses / store data', 'fields']:
    label=next(e for e in s.elements if e['type']=='text' and e['text'].startswith(prefix))
    s.base('rectangle','label-bg-'+label['id'],label['x']-8,label['y']-3,
           label['width']+16,label['height']+6,BG,BG,1)

# These nodes are defined after their shared input bus, so bind them now.
for ident in ['integer','fp32','tensor','lsu','branch']:
    e=next(e for e in s.elements if e['id']=='dispatch-'+ident)
    e['endBinding']={'elementId':ident,'focus':0,'gap':0}
    s.nodes[ident]['boundElements'].append({'id':e['id'],'type':'arrow'})

s.save()

# Audit the diagram against every module reachable from the actual SM top.
rtl_files=[*sorted((REPO/'rtl').rglob('*.sv')),*sorted((REPO/'rtl').rglob('*.v'))]
module_files={}
module_text={}
for path in rtl_files:
    src=path.read_text()
    names=re.findall(r'\bmodule\s+(\w+)',src)
    if names:
        module_files[names[0]]=path
        module_text[names[0]]=src
reachable=set()
def walk(name):
    if name in reachable:return
    reachable.add(name)
    for candidate in module_files:
        if re.search(r'\b'+re.escape(candidate)+r'\s+(?:#\s*\([^;]*?\)\s*)?\w+\s*\(',module_text[name]):
            walk(candidate)
walk('snackwell_sm')
covered={'rtl/sm/snackwell_sm.sv'}
for node in s.nodes.values():
    refs=node.get('customData',{}).get('rtl',[])
    covered.update([refs] if isinstance(refs,str) else refs)
assert all(str(module_files[n].relative_to(REPO)) in covered for n in reachable), sorted(reachable)
audit={name:{'source':str(module_files[name].relative_to(REPO)),
             'sha256':hashlib.sha256(module_files[name].read_bytes()).hexdigest()}
       for name in sorted(reachable)}
(OUT/'code/rtl-coverage.json').write_text(json.dumps(audit,indent=2)+'\n')
print(f'Coverage: all {len(reachable)} reachable module types are represented.')
print(f'Wrote {len(s.elements)} editable elements to {OUT}')
