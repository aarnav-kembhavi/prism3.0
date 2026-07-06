"""
formula_v2.py
-------------
Formula recall fix (opt-in: PRISM_FML_V2=1). Measured leaks, no new models:

1. Multi-formula blocks: PP-DocLayout boxes stacked/side-by-side display
   equations as ONE Formula region; GT annotates per equation. One merged pred
   matches at most one GT (the rest score 0 "empty pred"), and merged crops
   truncate at Texo's 256-token cap. -> split Formula crops into rows (ink
   bands) and columns (wide gaps), one det per equation.
2. Confidence gate: coverage recall is 92.4% at conf 0.30 vs 87% at the old
   0.50 (PP-StructureV3 runs this detector at 0.30). -> callers lower the
   detect conf for MATH_CLASSES; this module guards the extra low-conf boxes.
3. Formula/Text overlap: a recovered display equation is usually ALSO inside a
   Text det. -> inline-vs-display decided by INK BESIDE the formula in the
   text crop (a display equation owns its horizontal band; an inline fragment
   has prose on the same rows), Text dets fully covered by Formula dets are
   dropped, partial overlaps get the formula region masked out of the text
   crop, and detection_postprocess no longer deletes Formulas inside Text.
4. Equation numbers: "(8)" / "(A.1)" at the line margin enters the crop and
   its tokens count against CDM on every matched formula. -> the column pass
   trims small margin chunks.

V2 of the splitter (first A/B showed 280 still-merged + 105 guard-killed):
Otsu binarization (scan noise defeated the fixed <200 ink test), a vertical
ink-bridge test replaces the absolute emptiness rule (matrix brackets/big
parens crossing a gap block the split), no height floor, and the column pass.
"""

import os

import numpy as np
from PIL import Image


def enabled() -> bool:
    return os.environ.get('PRISM_FML_V2', '1') != '0'


# ── geometry helpers ──────────────────────────────────────────────────────────

def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _inter_frac(a, b):
    """Fraction of a's area inside b."""
    aa = _area(a)
    return _inter(a, b) / aa if aa > 0 else 0.0


def _binarize(img: Image.Image) -> np.ndarray:
    """Otsu ink mask (True = ink). Robust to tinted/noisy scan backgrounds."""
    import cv2
    arr = np.asarray(img.convert('L'))
    thr, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2 binarizes as (src > thr); ink is the complement, so <= (for a pure
    # 0/255 image Otsu returns thr=0.0 and a strict < would see no ink at all)
    return arr <= thr


# ── 1. inline-FP guard (ink-beside test) ──────────────────────────────────────

_GUARD_MARGIN_FRAC = 0.12   # ignore line-number/eq-number margins of the text det
_GUARD_INK_FRAC = 0.02      # >2% ink beside the formula's band => inline


def _drop_inline_fps(detections):
    """Drop Formula dets that are INLINE math inside a paragraph, at ANY conf.

    PP-DocLayout's formula class fires on inline fragments (x_2, m_{rp}, ...)
    with conf up to ~0.9. A display equation owns its horizontal band of the
    paragraph (blank left/right of it, except the eq-number margin); an inline
    fragment has prose ink on the same rows. Decided on the text det's crop.
    """
    texts = [d for d in detections if d['class_name'] == 'Text' and d.get('crop') is not None]
    fml_boxes = [d['bbox'] for d in detections if d['class_name'] == 'Formula']
    # A Text det whose INK is mostly formulas is a MATH-DENSE GRID (textbook
    # answer keys: dozens of tiny numbered expressions in columns), not a
    # paragraph — the "ink beside" there is other formulas, and the guard was
    # deleting 58/71 real formulas on such pages. Exempt those hosts.
    # Counting formulas alone is NOT enough: a PPT slide's page-wide Text det
    # can contain 4+ formula chips amid dominant prose (measured: +0.9 text
    # edit on such slides when they were exempted) — so require >=50% of the
    # host's ink to fall inside formula boxes.
    dense_hosts = set()
    for t in texts:
        inside = [f for f in fml_boxes if _inter_frac(f, t['bbox']) >= 0.5]
        if len(inside) < 4:
            continue
        ink = _binarize(t['crop'])
        h, w = ink.shape
        total = ink.sum()
        if total == 0:
            continue
        tb = t['bbox']
        sx = w / max(tb[2] - tb[0], 1.0)
        sy = h / max(tb[3] - tb[1], 1.0)
        covered = np.zeros_like(ink)
        for f in inside:
            gx1 = max(0, int((f[0] - tb[0]) * sx)); gy1 = max(0, int((f[1] - tb[1]) * sy))
            gx2 = min(w, int((f[2] - tb[0]) * sx)); gy2 = min(h, int((f[3] - tb[1]) * sy))
            covered[gy1:gy2, gx1:gx2] = True
        if (ink & covered).sum() >= 0.5 * total:
            dense_hosts.add(id(t))
    ink_cache = {}
    out = []
    for d in detections:
        if d['class_name'] != 'Formula':
            out.append(d)
            continue
        b = d['bbox']
        host = None
        for t in texts:
            if _inter_frac(b, t['bbox']) >= 0.8 and _area(b) < 0.9 * _area(t['bbox']):
                host = t
                break
        # Dets converted from V3's inline_formula class don't get the
        # dense-grid free pass: their own count is what tips a paragraph
        # over the density threshold, and waving them through eats the
        # paragraph's text (measured +0.004 full-set text edit). The
        # ink-blanking below already keeps legitimate side-by-side grids.
        if host is None or (id(host) in dense_hosts and not d.get('from_inline')):
            out.append(d)
            continue
        tb = host['bbox']
        if (b[2] - b[0]) >= 0.6 * (tb[2] - tb[0]):
            out.append(d)          # spans most of the paragraph width: display
            continue
        key = id(host)
        if key not in ink_cache:
            hink = _binarize(host['crop'])
            # other Formula dets' ink is not prose: blank their regions so
            # side-by-side equations don't read as "text beside"
            hh, hw = hink.shape
            tb0 = host['bbox']
            sx0 = hw / max(tb0[2] - tb0[0], 1.0)
            sy0 = hh / max(tb0[3] - tb0[1], 1.0)
            for fb in fml_boxes:
                if _inter(fb, tb0) <= 0:
                    continue
                gx1 = max(0, int((fb[0] - tb0[0]) * sx0) - 2)
                gy1 = max(0, int((fb[1] - tb0[1]) * sy0) - 2)
                gx2 = min(hw, int((fb[2] - tb0[0]) * sx0) + 2)
                gy2 = min(hh, int((fb[3] - tb0[1]) * sy0) + 2)
                if gx2 > gx1 and gy2 > gy1:
                    hink[gy1:gy2, gx1:gx2] = False
            ink_cache[key] = hink
        ink = ink_cache[key]
        h, w = ink.shape
        sx = w / max(tb[2] - tb[0], 1.0)
        sy = h / max(tb[3] - tb[1], 1.0)
        fy1 = max(0, int((b[1] - tb[1]) * sy))
        fy2 = min(h, int((b[3] - tb[1]) * sy))
        fx1 = max(0, int((b[0] - tb[0]) * sx))
        fx2 = min(w, int((b[2] - tb[0]) * sx))
        if fy2 <= fy1:
            out.append(d)
            continue
        m = int(_GUARD_MARGIN_FRAC * w)
        pad = 6
        band = ink[fy1:fy2]
        sides = []
        if fx1 - pad > m:
            sides.append(band[:, m:fx1 - pad])
        if fx2 + pad < w - m:
            sides.append(band[:, fx2 + pad:w - m])
        if not sides:
            out.append(d)
            continue
        beside = np.concatenate([s.reshape(-1) for s in sides])
        if beside.size and beside.mean() > _GUARD_INK_FRAC:
            continue               # prose on the same rows: inline -> drop
        out.append(d)
    return out


# ── 2. block split: rows (ink bands) then columns (wide gaps) ─────────────────

def _row_bands(ink: np.ndarray, min_gap_h: int = 6, min_band_h: int = 10,
               rel_gap: float = 0.25):
    """(top, bottom) bands = equation lines of a stacked formula block.

    Split candidates at runs of >= min_gap_h near-empty rows (relative to the
    densest row, so scan noise doesn't mask gaps). A candidate is rejected if
    a vertical ink run BRIDGES it (matrix bracket, big paren/integral): some
    column has ink through >=70% of the gap and touches both sides. A thin
    segment WIDER than both neighbours is a fraction bar: numerator, bar and
    denominator fuse into one band. Other tiny segments (accents, stray
    marks) merge into their nearest neighbour but do NOT count toward the
    line-height estimate — the rel_gap merge (gaps < rel_gap x median CORE
    height are intra-formula spacing) uses heights measured before any merge,
    otherwise inflated bands cascade into one giant band.
    """
    h, w = ink.shape
    rowd = ink.sum(axis=1)
    if rowd.max() == 0:
        return []
    empty = rowd <= max(2, int(0.05 * rowd.max()))

    # ink segments separated by empty runs >= min_gap_h
    segs, gaps = [], []          # gaps[i] between segs[i] and segs[i+1]
    start, gap_start = None, None
    for r in range(h):
        if empty[r]:
            if start is not None:
                if gap_start is None:
                    gap_start = r
        else:
            if start is None:
                start = r
            elif gap_start is not None:
                if r - gap_start >= min_gap_h:
                    segs.append([start, gap_start])
                    gaps.append((gap_start, r))
                    start = r
                gap_start = None
    if start is not None:
        segs.append([start, gap_start if gap_start is not None else h])
    if len(segs) <= 1:
        return [tuple(s) for s in segs]

    # reject gaps crossed by a vertical ink bridge
    merged_segs = [segs[0]]
    for i, (g0, g1) in enumerate(gaps):
        gap_cols = ink[g0:g1].sum(axis=0) >= 0.7 * (g1 - g0)
        above = ink[max(0, g0 - 4):g0].any(axis=0)
        below = ink[g1:min(h, g1 + 4)].any(axis=0)
        if (gap_cols & above & below).any():
            merged_segs[-1][1] = segs[i + 1][1]      # bridged: merge across
        else:
            merged_segs.append(segs[i + 1])
    segs = merged_segs
    if len(segs) <= 1:
        return [tuple(s) for s in segs]

    # line-height estimate from CORE segments only, before any merging
    core = sorted(s[1] - s[0] for s in segs if s[1] - s[0] >= min_band_h)
    med_h = core[len(core) // 2] if core else min_band_h

    def _extent(s):
        cols = np.where(ink[s[0]:s[1]].any(axis=0))[0]
        return (cols[0], cols[-1] + 1) if len(cols) else (0, 0)

    # tiny segments: fraction bar (wider than both neighbours) fuses the
    # trio; anything else merges into the closer neighbour
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s[1] - s[0] >= min_band_h:
                continue
            if 0 < i < len(segs) - 1:
                sl, sr = _extent(s)
                pl, pr = _extent(segs[i - 1])
                nl, nr = _extent(segs[i + 1])
                if sl <= min(pl, nl) + 3 and sr >= max(pr, nr) - 3:
                    segs[i - 1][1] = segs[i + 1][1]      # fraction: fuse trio
                    del segs[i:i + 2]
                    changed = True
                    break
            gap_prev = s[0] - segs[i - 1][1] if i > 0 else None
            gap_next = segs[i + 1][0] - s[1] if i + 1 < len(segs) else None
            if gap_prev is not None and (gap_next is None or gap_prev <= gap_next):
                segs[i - 1][1] = s[1]
            else:
                segs[i + 1][0] = s[0]
            del segs[i]
            changed = True
            break

    # merge across gaps that are small relative to line height
    merged = [segs[0]]
    for s in segs[1:]:
        if s[0] - merged[-1][1] < rel_gap * med_h:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [tuple(s) for s in merged]


def _col_chunks(ink: np.ndarray, top: int, bot: int,
                gap_factor: float = 1.4, min_gap_w: int = 28):
    """(left, right) chunks of one band; trims equation-number margin chunks.

    Splits at horizontal whitespace >= max(min_gap_w, gap_factor x band
    height) — GT annotates 'eq, qualifier' pairs separately, and equation
    numbers sit >=1 line-height away at the margin. Intra-equation spacing
    stays below that. A small first/last chunk (<= max(50px, 8% of width))
    across such a gap is an equation number: dropped, not emitted.
    """
    band = ink[top:bot]
    h = bot - top
    w = band.shape[1]
    cold = band.sum(axis=0)
    if cold.max() == 0:
        return []
    empty = cold <= max(1, int(0.05 * cold.max()))

    runs, start = [], None
    for c in range(w):
        if not empty[c]:
            if start is None:
                start = c
        else:
            if start is not None:
                runs.append([start, c])
                start = None
    if start is not None:
        runs.append([start, w])
    if not runs:
        return []

    gap_thr = max(min_gap_w, int(gap_factor * h))
    chunks = [runs[0]]
    for r in runs[1:]:
        if r[0] - chunks[-1][1] < gap_thr:
            chunks[-1][1] = r[1]
        else:
            chunks.append(r)

    # trim equation-number chunks at either margin
    num_w = max(50, int(0.08 * w))
    if len(chunks) >= 2 and (chunks[-1][1] - chunks[-1][0]) <= num_w:
        chunks = chunks[:-1]
    if len(chunks) >= 2 and (chunks[0][1] - chunks[0][0]) <= num_w:
        chunks = chunks[1:]
    return [tuple(c) for c in chunks]


_MAX_SUBDETS = 12   # more than this in one region: not an equation stack


def _split_formula_dets(detections):
    """Replace each multi-equation Formula det with one det per equation."""
    out = []
    for d in detections:
        crop = d.get('crop')
        if d['class_name'] != 'Formula' or crop is None or crop.height < 20:
            out.append(d)
            continue
        ink = _binarize(crop)
        bands = _row_bands(ink)
        if not bands:
            out.append(d)
            continue
        x1, y1, x2, y2 = d['bbox']
        sy = (y2 - y1) / crop.height
        sx = (x2 - x1) / crop.width
        subs = []
        for top, bot in bands:
            chunks = _col_chunks(ink, top, bot)
            if not chunks:
                chunks = [(0, crop.width)]
            for lft, rgt in chunks:
                t = max(0, top - 3)
                b = min(crop.height, bot + 3)
                l = max(0, lft - 4)
                r = min(crop.width, rgt + 4)
                sub = dict(d)
                sub['crop'] = crop.crop((l, t, r, b))
                sub['bbox'] = [x1 + l * sx, y1 + t * sy, x1 + r * sx, y1 + b * sy]
                subs.append(sub)
        if not subs or len(subs) > _MAX_SUBDETS:
            out.append(d)
        elif len(subs) == 1:
            out.append(subs[0])    # single equation, possibly eq-number-trimmed
        else:
            out.extend(subs)
    return out


# ── 3. formula/text dedup + masking ───────────────────────────────────────────

def _resolve_text_overlaps(detections):
    """A Text det almost fully covered by Formula dets duplicates them: drop it.
    Partial overlaps (equation inside a paragraph box) get the formula region
    whited out of the text crop so text OCR doesn't garble it into the output.
    """
    fml = [d['bbox'] for d in detections if d['class_name'] == 'Formula']
    if not fml:
        return detections
    out = []
    for d in detections:
        if d['class_name'] != 'Text':
            out.append(d)
            continue
        t = d['bbox']
        overlapping = [f for f in fml if _inter(f, t) > 0]
        if not overlapping:
            out.append(d)
            continue
        covered = sum(_inter(f, t) for f in overlapping) / max(_area(t), 1.0)
        if covered >= 0.85:
            continue  # text det IS the formula(s); Formula dets own this region
        crop = d.get('crop')
        if crop is not None and crop.width > 1:
            sx = crop.width / max(t[2] - t[0], 1.0)
            sy = crop.height / max(t[3] - t[1], 1.0)
            masked = None
            for f in overlapping:
                if _inter(f, t) < 0.3 * _area(f):
                    continue  # graze — leave it
                if masked is None:
                    masked = crop.copy()
                mx1 = int((max(f[0], t[0]) - t[0]) * sx)
                my1 = int((max(f[1], t[1]) - t[1]) * sy)
                mx2 = int((min(f[2], t[2]) - t[0]) * sx)
                my2 = int((min(f[3], t[3]) - t[1]) * sy)
                if mx2 > mx1 and my2 > my1:
                    fill = 255 if masked.mode == 'L' else (255, 255, 255)
                    masked.paste(fill, (mx1, my1, mx2, my2))
            if masked is not None:
                d = dict(d)
                d['crop'] = masked
        out.append(d)
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def apply_formula_v2(detections, img_width: int = None, img_height: int = None):
    """Full pass: giant-FP drop -> inline-FP guard -> text dedup/masking ->
    block split.

    Dedup/masking runs BEFORE the split so coverage is judged against the full
    formula block box (splitting keeps only the ink bands, undercounting it).
    """
    if img_width and img_height:
        # a "Formula" covering most of the page is a background FP (seen on
        # PPT slides: whole-slide boxes at conf ~0.3 that consume the real
        # formula chips inside via same-class containment)
        page_area = float(img_width) * float(img_height)
        detections = [d for d in detections
                      if d['class_name'] != 'Formula'
                      or _area(d['bbox']) <= 0.5 * page_area]
    detections = _drop_inline_fps(detections)
    detections = _resolve_text_overlaps(detections)
    detections = _split_formula_dets(detections)
    return detections
