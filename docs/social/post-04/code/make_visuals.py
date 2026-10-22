#!/usr/bin/env python3
"""Build editable Excalidraw scenes and matching local previews for build log 04.

Requires Pillow. Uses the same palette and Arial text as build log 03.
All illustrations use editable text, rectangles and arrows; no embedded images.
The results card reads captured verification evidence, never invented results.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import struct
import zipfile

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[2]
EVIDENCE = OUT / "code/evidence"
PREVIEW = OUT / "previews"
STEPS = OUT / "animation/steps"
W, H = 1600, 1000
BG, INK, MUTED, LINE = "#fffefa", "#22262f", "#606674", "#bac0c9"
GRAY = "#f5f5f1"
BLUE, BL = "#1971c2", "#e7f5ff"
ORANGE, OL = "#c56a0a", "#fff4e6"
PURPLE, PL = "#7048c6", "#f3f0ff"
GREEN, GL = "#2b8a3e", "#ebfbee"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"
FONTS = {}


def font(size, mono=False):
    key = size, mono
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(MONO if mono else FONT, size)
    return FONTS[key]


class Scene:
    def __init__(self, name):
        self.name = name
        self.elements = []
        self.containment = []

    def base(self, typ, x, y, w, h, color=INK, fill="transparent", sw=2, group=None):
        i = len(self.elements)
        e = dict(id=f"{self.name}-{i:03d}", type=typ, x=x, y=y, width=w,
                 height=h, angle=0, strokeColor=color, backgroundColor=fill,
                 fillStyle="solid", strokeWidth=sw, strokeStyle="solid",
                 roughness=0, opacity=100, groupIds=[group] if group else [],
                 frameId=None, roundness=None, seed=41000+i, version=1,
                 versionNonce=81000+i, isDeleted=False, boundElements=[],
                 updated=1, link=None, locked=False)
        self.elements.append(e)
        return e

    def text(self, x, y, text, size=28, color=INK, center=False, mono=False,
             group=None, parent=None):
        text = str(text)
        lines = text.split("\n")
        w = max(font(size, mono).getlength(line) for line in lines) + 4
        if mono:
            w *= 1.06  # Allow for Excalidraw's bundled monospace font.
        h = size * 1.25 * len(lines)
        e = self.base("text", x-w/2 if center else x, y, w, h, color, sw=1, group=group)
        e.update(text=text, originalText=text, fontSize=size, fontFamily=3 if mono else 2,
                 textAlign="center" if center else "left", verticalAlign="top",
                 containerId=None, autoResize=True, lineHeight=1.25)
        if parent:
            self.containment.append((e, parent))
        return e

    def rect(self, x, y, w, h, color=LINE, fill=BG, sw=2, group=None):
        e = self.base("rectangle", x, y, w, h, color, fill, sw, group)
        e["roundness"] = {"type": 3}
        return e

    def arrow(self, pts, color=INK, sw=3, head=True):
        x, y = pts[0]
        coords = [[a-x, b-y] for a, b in pts]
        xs, ys = zip(*coords)
        e = self.base("arrow" if head else "line", x, y, max(xs)-min(xs),
                      max(ys)-min(ys), color, sw=sw)
        e.update(points=coords, startBinding=None, endBinding=None,
                 startArrowhead=None, endArrowhead="arrow" if head else None,
                 lastCommittedPoint=None)
        return e

    def box(self, x, y, w, h, title, sub="", color=INK, fill=GRAY, size=30, subsize=23):
        group = f"{self.name}-group-{len(self.elements)}"
        p = self.rect(x, y, w, h, color, fill, group=group)
        self.text(x+w/2, y+21, title, size, color, True, group=group, parent=p)
        if sub:
            self.text(x+w/2, y+67, sub, subsize, MUTED, True, group=group, parent=p)
        return p

    def heading(self, title, subtitle):
        self.text(70, 44, "SNACKWELL  /  BUILD LOG 04", 22, MUTED)
        self.text(70, 92, title, 54)
        self.text(72, 166, subtitle, 27, MUTED)

    def footer(self, text):
        self.arrow([(70, 930), (1530, 930)], LINE, 1, False)
        self.text(70, 950, "@satvikgari", 20, MUTED)
        self.text(340, 950, text, 20, MUTED)

    def validate(self):
        ids = [e["id"] for e in self.elements]
        assert len(ids) == len(set(ids))
        assert not any(e["type"] == "image" for e in self.elements)
        for e in self.elements:
            if e["type"] in ("arrow", "line"):
                pts = [(e["x"]+x, e["y"]+y) for x, y in e["points"]]
            else:
                pts = [(e["x"], e["y"]), (e["x"]+e["width"], e["y"]+e["height"])]
            assert all(0 <= x <= W and 0 <= y <= H for x, y in pts), (self.name, e)
        for t, p in self.containment:
            assert t["x"] >= p["x"]+9 and t["y"] >= p["y"]+8, t["text"]
            assert t["x"]+t["width"] <= p["x"]+p["width"]-9, t["text"]
            assert t["y"]+t["height"] <= p["y"]+p["height"]-8, t["text"]
        texts = [e for e in self.elements if e["type"] == "text"]
        for i, a in enumerate(texts):
            for b in texts[i+1:]:
                overlap_x = min(a["x"]+a["width"], b["x"]+b["width"]) - max(a["x"], b["x"])
                overlap_y = min(a["y"]+a["height"], b["y"]+b["height"]) - max(a["y"], b["y"])
                assert overlap_x <= 0 or overlap_y <= 0, (self.name, "overlapping text", a["text"], b["text"])

    def render(self):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        for e in self.elements:
            x, y, w, h = (e[k] for k in ("x", "y", "width", "height"))
            c, fill, sw = (e[k] for k in ("strokeColor", "backgroundColor", "strokeWidth"))
            if e["type"] == "rectangle":
                d.rounded_rectangle((x, y, x+w, y+h), radius=16, fill=fill, outline=c, width=sw)
            elif e["type"] == "text":
                sz, mono = e["fontSize"], e["fontFamily"] == 3
                for n, line in enumerate(e["text"].split("\n")):
                    xx = x + (w-font(sz, mono).getlength(line))/2 if e["textAlign"] == "center" else x
                    d.text((xx, y+n*sz*1.25), line, font=font(sz, mono), fill=c, anchor="lt")
            elif e["type"] in ("arrow", "line"):
                pts = [(x+a, y+b) for a, b in e["points"]]
                d.line(pts, fill=c, width=sw, joint="curve")
                if e.get("endArrowhead"):
                    (px, py), (tx, ty) = pts[-2:]
                    a = math.atan2(ty-py, tx-px)
                    wings = [(tx-15*math.cos(a+s*.43), ty-15*math.sin(a+s*.43)) for s in (-1, 1)]
                    d.line([wings[0], (tx, ty), wings[1]], fill=c, width=sw)
        return im

    def save(self, directory=OUT, preview=True):
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(type="excalidraw", version=2, source="https://excalidraw.com",
                       elements=self.elements, appState={"viewBackgroundColor": BG,
                       "gridSize": None, "theme": "light", "exportBackground": True}, files={})
        (directory / (self.name+".excalidraw")).write_text(json.dumps(payload, indent=2)+"\n")
        im = self.render()
        if preview:
            PREVIEW.mkdir(exist_ok=True)
            im.save(PREVIEW / (self.name+".png"))
        return im


def read_evidence():
    metadata = json.loads((EVIDENCE / "run-metadata.json").read_text())
    assert "completed_utc" in metadata and all(c["exit_code"] == 0 for c in metadata["commands"])
    regression = (EVIDENCE / "regression.txt").read_text()
    python = (EVIDENCE / "python-tests.txt").read_text()
    rtl_passed, rtl_total = map(int, re.search(r"(\d+)/(\d+) testbenches passed", regression).groups())
    methods = int(re.search(r"Ran (\d+) tests", python).group(1))
    assert rtl_passed == rtl_total and python.rstrip().endswith("OK")
    results = json.loads((EVIDENCE / "workload-results.json").read_text())
    assert results["verified"] and all(r["verified"] for r in results["workloads"].values())
    vector = json.loads((EVIDENCE / "vector_add-result.json").read_text())
    attention = json.loads((EVIDENCE / "attention-result.json").read_text())
    tile8 = json.loads((EVIDENCE / "tiled_matmul-result.json").read_text())
    tile16 = json.loads((EVIDENCE / "tiled_matmul_16x16-result.json").read_text())
    for d in (vector, attention, tile8, tile16):
        assert d["verified"]
    assert tile8["stages"][0]["profile"]["opcode_counts"]["TMUL"] == 8
    assert tile16["stages"][0]["profile"]["opcode_counts"]["TMUL"] == 64
    assert [s["stage"] for s in attention["stages"]] == ["qk", "scale_mask", "softmax", "pv"]
    return metadata, rtl_total, methods, results["workloads"], vector, attention


def opening_scene(workloads, vector):
    """A cover with independent workload examples, not a hardware pipeline."""
    s = Scene("00-kernel-overview")
    s.text(70, 48, "SNACKWELL  /  BUILD LOG 04", 23, MUTED)
    s.text(70, 111, "Snackwell has kernels now.", 72)
    s.text(74, 214, "Shared memory, an assembler, and programs running on the SM.", 31, MUTED)

    # Four real vector-add lanes from the captured run.
    p = s.rect(70, 333, 460, 388, BLUE, BL)
    s.text(300, 360, "Vector operations", 37, BLUE, True, parent=p)
    values = lambda key: [struct.unpack(">f", bytes.fromhex(word))[0] for word in vector[key]][:4]
    a, b, c = [values(key) for key in ("a_bits", "b_bits", "c_bits")]
    for i in range(4):
        x = 156+i*96
        s.text(x, 438, f"{a[i]:g}", 35, BLUE, True, parent=p)
        s.text(x, 493, f"{b[i]:g}", 35, BLUE, True, parent=p)
        q = s.rect(x-37, 568, 74, 64, GREEN, GL)
        s.text(x, 579, f"{c[i]:g}", 35, GREEN, True, parent=q)
    s.text(93, 491, "+", 37, BLUE, parent=p)
    s.arrow([(106, 546), (494, 546)], BLUE, 2, False)
    s.text(300, 667, "FP32 add + multiply", 27, BLUE, True, parent=p)

    # An 8x8 matrix divided into four 4x4 tiles.
    p = s.rect(570, 333, 460, 388, PURPLE, PL)
    s.text(800, 360, "Tiled matmul", 37, PURPLE, True, parent=p)
    cell = 22
    for tile_r in range(2):
        for tile_c in range(2):
            x, y = 703+tile_c*104, 431+tile_r*104
            s.rect(x, y, cell*4, cell*4, PURPLE, BG, sw=2)
            for j in range(1, 4):
                s.arrow([(x+j*cell, y), (x+j*cell, y+4*cell)], PURPLE, 1, False)
                s.arrow([(x, y+j*cell), (x+4*cell, y+j*cell)], PURPLE, 1, False)
    s.text(800, 667, "8×8 and 16×16 matrices", 27, PURPLE, True, parent=p)

    # This is the allowed-position mask, not fabricated attention probabilities.
    p = s.rect(1070, 333, 460, 388, GREEN, GL)
    s.text(1300, 360, "Single-head attention", 35, GREEN, True, parent=p)
    for r in range(4):
        for c in range(4):
            allowed = c <= r
            s.rect(1204+c*49, 431+r*49, 43, 43,
                   GREEN if allowed else LINE, GREEN if allowed else BG, sw=1)
    s.text(1300, 667, "Causal mask shown", 27, GREEN, True, parent=p)

    p = s.rect(370, 804, 860, 82, GREEN, GL)
    s.text(800, 829, f"{len(workloads)} / {len(workloads)} workload cases verified", 32, GREEN, True, parent=p)
    s.footer("Implemented and checked in RTL simulation")
    return s


def assembler_scene(vector):
    s = Scene("01-assembly-to-sm")
    s.heading("Writing a program for Snackwell", "The assembler turns readable instructions into machine code.")
    boxes = [(70, "Assembly source", "vector_add.asm", BLUE, BL),
             (620, "Machine code", "32-bit instructions", PURPLE, PL),
             (1170, "Snackwell SM", "Runs the program", GREEN, GL)]
    for i, (x, title, sub, color, fill) in enumerate(boxes):
        s.box(x, 280, 360, 135, title, sub, color, fill, size=34, subsize=27)
        if i < 2:
            s.arrow([(x+363, 347), (x+535, 347)])
            s.text(x+449, 304, "assemble" if i == 0 else "load", 24, MUTED, True)
    p = s.rect(70, 512, 720, 340, BLUE, BL)
    s.text(100, 537, "Load → add → store", 34, BLUE, parent=p)
    source = (REPO / "kernels/vector_add.asm").read_text()
    code = [re.split(r";|//|#", line)[0].strip() for line in source.splitlines()]
    code = [line for line in code if line]
    s.text(102, 615, "\n".join(code[5:9]), 31, mono=True, parent=p)
    s.text(102, 794, "Address setup omitted from this excerpt.", 25, MUTED, parent=p)
    p = s.rect(845, 512, 685, 340, GREEN, GL)
    s.text(880, 537, "Vector-add example", 34, GREEN, parent=p)
    for y, label, numbers, color in [(619, "A", "1    2    3    4", BLUE),
                                     (675, "B", "10   20   30   40", ORANGE),
                                     (750, "C", "11   22   33   44", GREEN)]:
        s.text(885, y, label, 32, color, parent=p)
        s.text(975, y, numbers, 32, color, mono=True, parent=p)
    s.arrow([(880, 730), (1490, 730)], LINE, 1, False)
    assert vector["instructions"] == len(code) == 10
    s.footer("RTL simulation · first four values shown · all 16 outputs checked.")
    return s


def memory_scene():
    s = Scene("02-shared-memory")
    s.heading("Shared memory takes turns when needed", "Four lanes share four memory banks.")
    for side, origin in enumerate((70, 820)):
        p = s.rect(origin, 245, 710, 574, LINE, BG)
        title = "Different banks" if side == 0 else "Same bank"
        s.text(origin+355, 277, title, 37, BLUE if side == 0 else ORANGE, True, parent=p)
        for i in range(4):
            x = origin+28+i*169
            q = s.rect(x, 374, 147, 78, BLUE, BL)
            s.text(x+73, 397, f"Lane {i}", 28, BLUE, True, parent=q)
        if side == 0:
            for i in range(4):
                x = origin+101+i*169
                s.arrow([(x, 454), (x, 550)], BLUE)
                q = s.rect(origin+28+i*169, 558, 147, 89, ORANGE, OL)
                s.text(x, 585, f"Bank {i}", 28, ORANGE, True, parent=q)
        else:
            # The bank serves the captured lanes in ascending lane order.
            for i in range(4):
                x = origin+101+i*169
                s.arrow([(x, 454), (x, 499)], ORANGE, 2, False)
            s.arrow([(origin+101, 499), (origin+608, 499)], ORANGE, 2, False)
            s.arrow([(origin+355, 499), (origin+355, 550)], ORANGE)
            q = s.rect(origin+186, 558, 338, 89, ORANGE, OL)
            s.text(origin+355, 585, "Bank 0", 31, ORANGE, True, parent=q)
        if side == 0:
            s.text(origin+355, 699, "All four served together", 32, GREEN, True, parent=p)
            s.text(origin+355, 751, "1 service cycle", 29, MUTED, True, parent=p)
        else:
            s.text(origin+355, 699, "Lane 0 → 1 → 2 → 3", 32, ORANGE, True, parent=p)
            s.text(origin+355, 751, "4 service cycles", 29, MUTED, True, parent=p)
    s.text(800, 862, "The load returns after every active lane has been served.", 31, center=True)
    s.footer("Service cycles shown · request acceptance and register writeback add time.")
    return s


def vector_scene(vector):
    s = Scene("03-vector-kernel")
    s.heading("One lane, one vector element", "C[i] = A[i] + B[i]     ·     warp 0 shown below")
    vals = lambda key: [struct.unpack(">f", bytes.fromhex(word))[0] for word in vector[key]]
    a, b, c = [vals(key) for key in ("a_bits", "b_bits", "c_bits")]
    assert all(x+y == z for x, y, z in zip(a, b, c))
    for lane in range(4):
        x = 70+lane*380
        p = s.rect(x, 290, 320, 340, BLUE, BL)
        s.text(x+160, 319, f"Lane {lane}", 32, BLUE, True, parent=p)
        s.text(x+160, 395, f"{a[lane]:g} + {b[lane]:g}", 45, center=True, parent=p)
        s.arrow([(x+160, 465), (x+160, 519)], PURPLE)
        s.text(x+160, 549, f"{c[lane]:g}", 50, GREEN, True, parent=p)
    s.text(800, 688, "Four warps × four lanes = sixteen output elements", 34, center=True)
    stages = [(210, "Load A[i] and B[i]", BLUE, BL), (655, "FADD", PURPLE, PL), (1100, "Store C[i]", GREEN, GL)]
    for i, (x, title, color, fill) in enumerate(stages):
        p = s.rect(x, 823, 335, 71, color, fill)
        s.text(x+167, 843, title, 27, color, True, parent=p)
        if i < 2:
            s.arrow([(x+337, 858), (x+432, 858)])
    s.footer(f"Vector-add example: {vector['cycles']} simulated execution cycles · all 16 outputs match.")
    return s


def tile_grid(s, x, name, color, fill, highlights=None, done=None, y=280):
    s.text(x+174, y-55, f"{name}  ·  8×8", 34, color, True)
    for r in range(2):
        for c in range(2):
            active = highlights is None or (r, c) in highlights
            complete = done is not None and (r, c) in done
            cc = GREEN if complete else color if active else LINE
            ff = GL if complete else fill if active else GRAY
            p = s.rect(x+c*180, y+r*157, 168, 145, cc, ff, 3 if active else 1)
            s.text(x+c*180+84, y+r*157+47, f"{name}{r}{c}", 42, cc, True, parent=p)


def tiling_scene(step=None):
    s = Scene("04-tiled-matmul" if step is None else f"step-{step+1:02d}")
    s.heading("8×8 matmul on a 4×4 array", "Each box is a 4×4 tile. Two tile products make one output tile.")
    r, c, inner = (0, 0, None) if step is None else (step//4, (step//2) % 2, step % 2)
    active_a = {(r, j) for j in range(2)} if inner is None else {(r, inner)}
    active_b = {(j, c) for j in range(2)} if inner is None else {(inner, c)}
    done = {(rr, cc) for rr in range(2) for cc in range(2) if step is not None and rr*2+cc < step//2}
    if inner == 1:
        done.add((r, c))
    tile_grid(s, 100, "A", BLUE, BL, active_a)
    tile_grid(s, 625, "B", ORANGE, OL, active_b)
    tile_grid(s, 1150, "C", PURPLE, PL, {(r, c)}, done)
    s.text(540, 405, "×", 55, MUTED, True)
    s.text(1065, 405, "=", 55, MUTED, True)
    if step is None:
        formula = "C00 = A00 × B00 + A01 × B10"
        caption = "Repeat for the other three output tiles."
    else:
        formula = f"C{r}{c} += A{r}{inner} × B{inner}{c}"
        caption = f"Pair {step+1} / 8  ·  " + (f"First product for C{r}{c}" if inner == 0 else f"Second product completes C{r}{c}")
    s.text(800, 626, formula, 41, PURPLE, True)
    s.text(800, 697, caption, 27, MUTED, True)
    for i, (title, color, fill) in enumerate([("Load tiles", BLUE, BL), ("Multiply (TMUL)", PURPLE, PL), ("Add to C + store", GREEN, GL)]):
        x = 165+i*465
        p = s.rect(x, 774, 345, 91, color, fill)
        s.text(x+172, 803, title, 29, color, True, parent=p)
        if i < 2:
            s.arrow([(x+347, 819), (x+451, 819)])
    s.footer("8×8: 8 TMULs · 16×16: 64 TMULs" if step is None else "Illustrated kernel order · each pictured tile is 4×4 · animation speed is not clock timing.")
    return s


def more_kernels_scene(workloads):
    """Independent operation examples for the additional-kernels tweet."""
    names = ("bias_add", "residual_add", "row_sum", "row_max", "relu", "gelu",
             "layernorm", "softmax", "linear")
    assert all(workloads[name]["verified"] for name in names)
    s = Scene("07-more-kernels")
    s.heading("More kernels running on Snackwell", "Each one runs as assembly on the SM.")

    def card(x, y, title, subtitle, color, fill):
        p = s.rect(x, y, 460, 300, color, fill)
        s.text(x+230, y+24, title, 35, color, True, parent=p)
        s.text(x+230, y+82, subtitle, 25, MUTED, True, parent=p)
        return p

    def down(x, y, color):
        s.arrow([(x, y+185), (x, y+215)], color, 2)

    p = card(70, 250, "Bias + residual", "Add values element by element.", BLUE, BL)
    s.text(300, 383, "Bias: X + b", 34, BLUE, True, parent=p)
    s.text(300, 433, "b repeats across rows", 24, MUTED, True, parent=p)
    s.text(300, 486, "Residual: X + R", 34, BLUE, True, parent=p)

    p = card(570, 250, "Row sum / max", "Reduce each row to one value.", ORANGE, OL)
    s.text(800, 386, "[1, 2, 3, 4]", 33, ORANGE, True, mono=True, parent=p)
    down(800, 250, ORANGE)
    for x, label in ((606, "sum = 10"), (819, "max = 4")):
        q = s.rect(x, 476, 175, 56, ORANGE, BG)
        s.text(x+87.5, 486, label, 28, ORANGE, True, parent=q)

    p = card(1070, 250, "ReLU / GELU", "Two activation functions.", PURPLE, PL)
    # Label the finite-input example as ReLU: GELU has a different result.
    s.text(1300, 386, "ReLU: [-2, -1, 1, 2]", 28, PURPLE, True, mono=True, parent=p)
    down(1300, 250, PURPLE)
    s.text(1300, 486, "[0, 0, 1, 2]", 33, PURPLE, True, mono=True, parent=p)

    p = card(70, 590, "LayerNorm", "Normalize each row.", PURPLE, PL)
    # Normalization includes epsilon; gamma and beta follow it.
    s.text(300, 726, "Center + normalize", 32, PURPLE, True, parent=p)
    down(300, 590, PURPLE)
    s.text(300, 826, "Scale + bias", 34, PURPLE, True, parent=p)

    p = card(570, 590, "Softmax", "Turn scores into weights.", GREEN, GL)
    s.text(800, 726, "[0, 0, 0, 0]", 33, GREEN, True, mono=True, parent=p)
    down(800, 590, GREEN)
    s.text(800, 829, "[0.25, 0.25, 0.25, 0.25]", 26, GREEN, True, mono=True, parent=p)

    p = card(1070, 590, "Linear layer", "Matmul + bias in two launches.", BLUE, BL)
    s.text(1300, 726, "X × W", 37, BLUE, True, parent=p)
    down(1300, 590, BLUE)
    s.text(1300, 826, "+ bias", 37, BLUE, True, parent=p)

    s.footer("Implemented and checked in RTL simulation · individual operation examples shown")
    return s


def attention_scene(attention):
    s = Scene("05-attention-kernels")
    s.heading("Attention in four kernel launches", "Single head · Q, K and V are supplied as inputs")
    stages = [("1  Scores", "Q × Kᵀ", "Compare queries and keys", BLUE, BL),
              ("2  Scale + mask", "Scores / √d", "Mask future tokens", ORANGE, OL),
              ("3  Softmax", "Scores → P", "Convert to weights", PURPLE, PL),
              ("4  Output", "P × V", "Combine value rows", GREEN, GL)]
    for i, (title, formula, sub, color, fill) in enumerate(stages):
        x = 70+i*380
        p = s.rect(x, 303, 320, 272, color, fill)
        s.text(x+160, 338, title, 31, color, True, parent=p)
        s.text(x+160, 414, formula, 39, color, True, parent=p)
        s.text(x+160, 514, sub, 23, MUTED, True, parent=p)
        if i < 3:
            s.arrow([(x+323, 442), (x+365, 442)])
    s.text(800, 642, "Each stage runs on the SM and writes its result to shared memory.", 31, center=True)
    s.text(800, 712, "The simulation runner passes that memory into the next launch.", 29, MUTED, True)
    p = s.rect(335, 794, 930, 87, GREEN, GL)
    s.text(800, 819, "Intermediate results and final output checked", 31, GREEN, True, parent=p)
    assert attention["launches"] == 4
    s.footer("Four separate simulator launches · d = head width · a complete transformer block is next.")
    return s


def results_scene(metadata, rtl, methods, workloads, attention):
    s = Scene("06-verified-results")
    s.heading("Checking the results", "Every workload runs through the SM in RTL simulation.")
    for x, count, label in [(70, rtl, "RTL testbenches"), (570, methods, "Python test methods"), (1070, len(workloads), "Runnable workloads")]:
        p = s.rect(x, 243, 460, 149, GREEN, GL)
        s.text(x+230, 264, f"{count} / {count}", 50, GREEN, True, parent=p)
        s.text(x+230, 338, label, 27, GREEN, True, parent=p)
    s.box(95, 489, 430, 138, "SM output memory", "What the kernel wrote", BLUE, BL, size=33, subsize=26)
    s.arrow([(534, 558), (664, 558)])
    s.box(675, 489, 830, 138, "Compare with an independent reference", "Check the actual numerical results", PURPLE, PL, size=32, subsize=26)
    s.text(800, 712, "Matmul: bit-for-bit FP32 checks", 33, center=True)
    s.text(800, 772, "GELU, LayerNorm, softmax and attention: checked within tolerances", 29, center=True)
    s.text(800, 845, "Also checked: memory outside the output and each stage of attention.", 27, MUTED, True)
    s.footer(f"Captured {metadata['completed_utc'][:10]} UTC · logs and full measurements saved with these diagrams.")
    return s


def main():
    metadata, rtl, methods, workloads, vector, attention = read_evidence()
    scenes = [opening_scene(workloads, vector), assembler_scene(vector), memory_scene(), vector_scene(vector),
              tiling_scene(), more_kernels_scene(workloads), attention_scene(attention),
              results_scene(metadata, rtl, methods, workloads, attention)]
    images = [s.save() for s in scenes]
    # Preserve each animation pose as a normal, independently editable scene.
    frames = [tiling_scene(i).save(STEPS, preview=False) for i in range(8)]
    (OUT / "animation").mkdir(exist_ok=True)
    frames[0].save(OUT / "animation/tiled-matmul.gif", save_all=True,
                   append_images=frames[1:], duration=[1600]*7+[2600], loop=0,
                   optimize=False, disposal=2)
    frames[-1].save(PREVIEW / "animation-last-step.png")
    # All preview thumbnails come from the same native-element scene data.
    sheet = Image.new("RGB", (1640, ((len(images)+1)//2)*520+10), "#e9e9e5")
    for i, im in enumerate(images):
        sheet.paste(im.resize((800, 500), Image.Resampling.LANCZOS), (10+(i%2)*820, 10+(i//2)*520))
    sheet.save(PREVIEW / "all-diagrams.png")
    manifest = {"canvas": [W, H], "style_reference": "docs/social/tensor-update/assets",
                "verified_utc": metadata["completed_utc"],
                "editable_scenes": [{"file": s.name+".excalidraw", "elements": len(s.elements)} for s in scenes],
                "animation_steps": 8,
                "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for root in ("rtl", "tb", "snackwell", "scripts", "tests", "kernels")
                                  for p in sorted((REPO/root).rglob("*"))
                                  if p.suffix in (".sv", ".v", ".py", ".asm") and p.is_file()}}
    (OUT / "code/manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(OUT / "snackwell-post-4-excalidraw.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.glob("*.excalidraw")) + sorted(STEPS.glob("*.excalidraw")):
            z.write(p, p.relative_to(OUT))
    tweets = json.loads((OUT / "tweets.json").read_text())
    draft = ["# Snackwell build log 04 — thread draft", "",
             "Draft only. Each numbered section below is one tweet; media notes are not tweet text.", ""]
    for i, tweet in enumerate(tweets, 1):
        assert tweet["number"] == i
        assert tweet["text"].isascii() and len(tweet["text"]) <= 280
        draft += [f"## {i}. {tweet['label']}", "", tweet["text"], ""]
        paths = tweet["media"] + ([tweet["editable"]] if "editable" in tweet else [])
        for path in paths:
            assert (OUT/path).exists(), path
        media = ", ".join(f"[{Path(p).name}]({p})" for p in tweet["media"]) or "None"
        draft += [f"**Attach:** {media}", ""]
        if "editable" in tweet:
            draft += [f"**Editable:** [{tweet['editable']}]({tweet['editable']})", ""]
        if "still_alternative" in tweet:
            path = tweet["still_alternative"]
            assert (OUT/path).exists()
            draft += [f"**Still alternative:** [{Path(path).name}]({path})", "", tweet["still_text_change"], ""]
        draft += [f"Text length: {len(tweet['text'])} ASCII characters (URL shortening only reduces this).", ""]
    (OUT / "thread-draft.md").write_text("\n".join(draft))
    print(f"Created {len(scenes)} editable diagrams, 8 editable animation steps, GIF and previews in {OUT}")


if __name__ == "__main__":
    main()
