"""Build the en_ch_mixed before/after panel, matching figures/multilingual/panel_*.pdf.

Geometry is copied from panel_zh_a.pdf:
  canvas 496.8 pt wide; left box x 36.7-263.5, right box x 270.0-496.8, top y 14.4;
  grey 0.5pt frame on the raw crop, blue 0.9pt frame on the compiled output;
  8pt Times-Bold column headers; rotated 8pt label + 6.4pt score in the left gutter.
Left half is a raster crop of the source page; right half is the compiled
main.pdf embedded as a vector Form XObject (show_pdf_page), so the panel stays
fully vector on the output side.
"""
import os
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(OUT, "cand", "c06.png")
CMP = str(ROOT / "outputs" / "c06_output" / "main.pdf")

LABEL = "English\u2013Chinese"
SCORE = "char F1 0.986"

# ── crop windows (raw px / compiled pt), chosen to show the same content ──
RAW_CLIP = fitz.Rect(140, 115, 1400, 787)       # header band + both columns → "opposite"/"path"
CMP_CLIP = fitz.Rect(34, 38, 578, 323.4)

# ── panel constants, from panel_zh_a.pdf ──
W = 496.8
LX0, LX1 = 36.7, 263.5
RX0, RX1 = 270.0, 496.8
TOP = 14.4
BOT_PAD = 2.85
GREY = (0.6901, 0.6902, 0.6901)
BLUE = (0.0, 0.44706, 0.69804)
HDR_SIZE = 8.0
LBL_SIZE = 8.0
SCR_SIZE = 6.4
BOXW = LX1 - LX0                                # 226.8

# ── heights follow each side's own aspect ratio ──
raw_doc = fitz.open(RAW)
raw_pg = raw_doc[0]
raw_h = BOXW * (RAW_CLIP.height / RAW_CLIP.width)
cmp_h = BOXW * (CMP_CLIP.height / CMP_CLIP.width)
H = TOP + max(raw_h, cmp_h) + BOT_PAD

out = fitz.open()
page = out.new_page(width=W, height=H)
page.draw_rect(fitz.Rect(0, 0, W, H), color=None, fill=(1, 1, 1))

lbox = fitz.Rect(LX0, TOP, LX1, TOP + raw_h)
rbox = fitz.Rect(RX0, TOP, RX1, TOP + cmp_h)

# left: raster crop of the source page, rendered at ~600 dpi equivalent
page.draw_rect(lbox, color=None, fill=(1, 1, 1))
zoom = 8.0 * BOXW / RAW_CLIP.width              # target ≈ 8 px per output pt
pix = raw_pg.get_pixmap(clip=RAW_CLIP, matrix=fitz.Matrix(zoom, zoom))
page.insert_image(lbox, pixmap=pix)
page.draw_rect(lbox, color=GREY, width=0.5)

# right: the compiled PDF region, kept as vector
page.draw_rect(rbox, color=None, fill=(1, 1, 1))
cmp_doc = fitz.open(CMP)
page.show_pdf_page(rbox, cmp_doc, 0, clip=CMP_CLIP)
page.draw_rect(rbox, color=BLUE, width=0.9)

# ── column headers ──
for text, box in (("raw page (crop)", lbox), ("PRISM output, compiled", rbox)):
    w = fitz.get_text_length(text, fontname="tibo", fontsize=HDR_SIZE)
    page.insert_text(fitz.Point((box.x0 + box.x1) / 2 - w / 2, 9.0),
                     text, fontname="tibo", fontsize=HDR_SIZE)

# ── rotated label + score in the left gutter, centred on the raw box ──
cy = (lbox.y0 + lbox.y1) / 2
lw = fitz.get_text_length(LABEL, fontname="tibo", fontsize=LBL_SIZE)
page.insert_text(fitz.Point(8.6, cy + lw / 2), LABEL,
                 fontname="tibo", fontsize=LBL_SIZE, rotate=90)
sw = fitz.get_text_length(SCORE, fontname="tiro", fontsize=SCR_SIZE)
page.insert_text(fitz.Point(22.6, cy + sw / 2), SCORE,
                 fontname="tiro", fontsize=SCR_SIZE, rotate=90)

FIGDIR = str(ROOT / "figures" / "multilingual")
dst = os.path.join(FIGDIR, "panel_mx_en_ch_p20.pdf")
out.save(dst, garbage=4, deflate=True)
print("wrote", dst)
print("canvas %.2f x %.2f pt  (%.3f x %.3f in)" % (W, H, W / 72, H / 72))
print("raw box h=%.2f  cmp box h=%.2f" % (raw_h, cmp_h))

done = fitz.open(dst)[0]
done.get_pixmap(dpi=200).save(os.path.join(FIGDIR, "panel_mx_en_ch_p20_200dpi.png"))
proof = done.get_pixmap(dpi=300)
proof.save(os.path.join(OUT, "panel_mixed_proof.png"))
print("proof", proof.width, "x", proof.height)
