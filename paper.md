# PRISM — experiment log (what we tried, what happened, what we kept)

> Running log of every approach tried on the way to the current pipeline
> (Overall 78.46 on OmniDocBench v1.6). Each entry: what we tried, what we
> measured, why it was kept or removed. Code for rejected approaches is
> deleted from the repo (this file is the record); model files may remain on
> disk but nothing references them.

## Layout detection

- **EasyOCR-era prototype (pre-benchmark)** — first text stack before the
  current architecture. Replaced wholesale by RapidOCR PP-OCRv4 (faster,
  smaller, better multilingual). Historical only.
- **Two-YOLO combo: YOLOv11n-DocLayNet (11 MB) + DocLayout-YOLO (72 MB)** —
  the original detector pair; DocLayout-YOLO boosted formulas/tables the nano
  model missed. Replaced by a single **PP-DocLayout_plus-L (RT-DETR, 124 MB
  ONNX, imgsz 800)** after a head-to-head showed it beats the combo on every
  OmniDocBench category for net +42 MB. Fallback code removed 2026-07-04
  (was: `PRISM_USE_PPDL_LAYOUT=0`, `pipeline/yolo_onnx.py`).
- **MFD (opendatalab YOLOv8l formula detector, 167 MB)** — wired as an
  opt-in second formula detector. Measured on the 313 formula pages:
  **+3.3 CDM for +167 MB and +2 s/page**, and its detection recall on the
  hard wide single-line equations (51%) was WORSE than PP-DocLayout's.
  Superseded by the zero-model FML_V2 fixes (+21.2 CDM). Wiring removed.
- **DocLayout-YOLO / PPDL-cache / FML-BOOST benchmark boost blocks** — layered
  recall boosts from the two-YOLO era. Instrumented the v10 full run: **none
  of them fired on any of 1651 pages** (no-ops under PP-DocLayout). Removed.

## Formula recognition & recall

- **pix2tex** — first math OCR; replaced by **Texo-distill** (20M-param
  distilled UniMERNet/PP-FormulaNet-S family) for quality. Worker removed.
- **Texo vs PP-FormulaNet-S head-to-head** (930 crops, edit distance): tie
  (~0.26 edit on clean crops) at **1/3 the size** → kept Texo. Recognition
  was proven NOT to be the formula bottleneck.
- **PP-FormulaNet_plus-M as recognizer upgrade** — would fix CJK-in-formula
  (Zh-BLEU 89.8 vs Texo's ~none) but is **617 MB and 1.3 s/formula on GPU**;
  CPU latency kills it. plus-S is only +1.7 En-BLEU over what Texo already
  ties, with weak CJK (53.3). Rejected; documented as the known ZH ceiling.
- **Content-based math re-routing** (run RapidOCR on text lines, classify
  output as math, re-route to Texo — the "cheap Marker") — **failed**: on OCR
  content alone the classifier gets 19% recall at usable precision because
  RapidOCR garbles math before any signal survives. (The 84.7% recall
  version of the classifier was an aspect-ratio cropping artifact — caught
  in a self-check.) Never shipped.
- **"Formula detection ceiling" (~58% recall) — a measurement artifact.**
  Strict one-to-one IoU>0.5 at conf 0.50 hid that: coverage recall of the
  same detector is 87→96% at 0.5→0.05; 32% of "missed" formulas sat inside
  ONE oversized det boxing a whole multi-equation stack (GT is per line);
  and `resolve_overlaps` silently deleted Formulas ≥80% contained in Text.
  Fix (kept, now default): formula conf 0.30 (math only), structure-aware
  block split (Otsu + vertical-ink-bridge for matrices + fraction-bar fusion
  + column split), equation-number trim, ink-beside inline guard,
  formula-region masking out of text crops. **CDM 56.90 → 78.11.**
- **`tex_to_md` row-separator bug** — the markdown converter ran text-mode
  `\\ → newline` over the WHOLE document, deleting array/matrix row
  separators from every multi-row formula pred in every benchmark run ever.
  Display math is now quarantined through conversion. Part of the +21 CDM.
- **Formula conf 0.15 arm** — measured: CDM identical to 0.30 (77.83), text
  +0.64pp worse, reading order worse. **0.30 is the operating point.** (Also
  surfaced that `filter_by_confidence` had a hidden 0.30 floor.)

## Tables

- **TATR PyTorch worker** — replaced by ONNX INT8 (115 MB safetensors → 30 MB,
  ~1.5× faster on CPU, no torch in the process tree). Torch worker removed.
- **Spanning cells** — TATR detects `table spanning cell`; the worker used to
  throw them away while **39% of GT tables contain spans** (those scored
  0.603 vs 0.723 unspanned). Now emitted as `\multicolumn`/`\multirow`,
  converted to colspan/rowspan HTML. Spanned tables 0.603 → 0.647;
  TEDS overall 69.96 → 71.46.
- **Token handling bugs (fixed, kept)**: OCR tokens straddling column
  boundaries were dumped whole into one cell (now split by character
  interpolation); tokens wider than 80% of the crop were deleted outright
  (killed full-width header cells); empty rows were dropped (GT keeps
  `<tr></tr>`); single-column tables were nuked by a 0.85-width filter.
- **Coordinate-heuristic table builder** — kept only as TATR-failure
  fallback.

- **Table confidence gate 0.30** (mirroring the formula gate) — measured on
  the 458 table pages and **rejected**: detection recall does rise 91.7→94.4%
  at 0.13 FP/page, but TEDS went 71.46→71.25 (the marginal dets produce
  low-scoring fragments and can steal the matcher from the real pred) and the
  9 zero-TEDS newspaper tables it targeted sit below conf 0.30 anyway.
  Default stays 0.50 (`PRISM_PPDL_TBL_CONF` knob kept).
- **RapidTable / SLANet-plus replaces TATR as primary (2026-07-05)** — the
  v10 per-table breakdown showed 102/665 tables below 0.3 TEDS (93 lost
  table-units): 40 detection misses, 46 TATR structure collapses (dense
  newspaper stats tables: GT 15 rows → TATR predicted 53 mostly-empty rows),
  and a persistent 10pp full-vs-structure gap (68.9 vs 79.2) from cell
  content, i.e. token-to-grid assignment. Stratified 60-table A/B (GT crops,
  OmniDocBench TEDS impl): catastrophic tables **-0.01 → 0.568**, mid
  0.640 → 0.666, good 0.931 → 0.923 (unchanged), median 1.34 s/table
  including its own PP-OCRv6-small cell OCR. Extrapolated per-table TEDS
  68.9 → ~78. SLANet-plus is 7.4 MB (TATR INT8 was 30 MB, stays as
  fallback only). Integrated as a stdio child process in `.venv_rtable`
  (`pipeline/rtable_worker.py` + `rtable_child.py`, `PRISM_RTABLE` kill
  switch); RapidTable emits final `<table>` HTML which now passes through
  `tex_to_md` under quarantine (the %-comment stripper truncated raw HTML at
  the first literal `%` — same class of bug as the math `\\` newline one).
- **Table gate 0.30 retested WITH RapidTable (2026-07-05) — still rejected.**
  458-page subset, both arms with RapidTable + PP-OCRv6: gate 0.50 page-TEDS
  78.83 / per-table 74.91 / text 0.0850; gate 0.30 78.63 / 74.49 / 0.0876.
  The marginal detections lose more via matcher-stealing and text damage than
  the recovered tables gain, regardless of the structure model. 0.50 stays.
- **RapidTable subset result (gate 0.50)**: page-TEDS 71.40 → **78.83**
  (+7.4), per-table 68.90 → 74.91 (333 improved / 139 regressed, no cheap
  guard found — empty-cell fraction does not separate regressions), text on
  table pages 0.098 → 0.085 (v6 OCR), RO neutral. Table-subset run: median
  4.93 s/pg, peak 2.32 GB.
- **Newspaper table detection at 800px** — tiled 2×2 detection (12% overlap)
  on broadsheets recovers some missed tables (Chicago Tribune 10/13 → 13/13,
  Boston Globe 14/26 → 18/26 with union) but WSJ agate stock listings are
  invisible to PPDL at any conf/tiling (0/6). Times UK's "missed" table was
  actually detected at conf 0.30 and rejected by the 0.50 table gate —
  retest the 0.30 gate now that RapidTable handles marginal tables (the
  original rejection was TATR-specific: marginal dets produced garbage
  fragments).
- **Answer-key pages (dense math grids)** — two textbook answer-key pages
  alone held 124 zero-CDM formulas (40.8% of ALL formula loss was EN books).
  The detector finds all of them (71/72 after postprocess); the inline guard
  was deleting 58 because the "prose ink beside" a formula was just OTHER
  formulas on the same row. Fix: (a) other Formula dets' ink is blanked
  before the beside-test; (b) a Text det with ≥4 contained formulas AND ≥50%
  of its INK inside formula boxes is a math-dense grid — guard skipped.
  The ink-fraction condition matters: exempting on count alone put +0.9 text
  edit on PPT slides (page-wide Text dets with 4+ chips amid dominant prose).
  A/B on the 313 formula pages: CDM 77.83→**78.96**, zeros 243→**110**, text
  +0.4pp on formula pages only, reading order best of all arms (0.1624).

- **Formula render failures (CDM=0 despite low edit distance)** — 34
  formulas compiled to nothing because of KaTeX-isms Texo emits that
  pdflatex rejects (`\infin`, `\gt`, `\lt`) and display delimiters nested
  inside the pipeline's own display wrapper (`\[` inside `\[`). `_sanitize`
  in the math worker now rewrites `\infin→\infty`, `\gt/\lt→>/<` and strips
  interior `\[ \] $$` (2026-07-05).

## Reading order

- **Flat "full-width + N columns" model for 3+ column pages** — scrambled
  newspapers (RO edit 0.595): their real structure is vertical regions each
  with its own columns. Replaced by **recursive XY-cut** ordering
  (`layout_utils.xycut_order`). First cut applied it only when the gutter
  detector said 3+ columns → newspapers went 0.595→0.543, but half of them
  are misdetected as "1 column" (touching boxes hide gutters). Extended to
  busy 1-column pages (≥8 dets; safe — XY-cut degenerates to top-down on a
  true single column): **newspapers 0.595→0.390, newspaper text 0.153→0.139**
  (67 improved / 2 regressed). Regression check on 60 book/academic/report
  pages: RO 0.279→0.264, text flat — no damage. 2-column pages keep the
  validated DAG/paracol path.
- **Formula split side-effect** — per-equation dets (instead of merged
  blocks) alone improved RO 0.323 → 0.258 on formula pages before XY-cut.

## Normalization (Stage 1)

- **v4: full unconditional Stage 1** (white balance → rectification → shadow
  → glare → moiré → CLAHE → DPI resize) — baseline.
- **v5: adaptive gating on detection scores** — abandoned (table regression).
- **v6: skip Stage 1 entirely** (deskew + modality only) — **won every
  metric** on OmniDocBench (+11.6pp formula EN vs v4): the corrections
  destroy clean digital renders, and the benchmark is mostly digital.
- **v7: formula crops from pre-CLAHE fidelity image** — only +0.6pp; the
  damage was DPI-resize + white balance too, not just CLAHE. Abandoned.
- **Kept**: three-way gate — screenshot (entropy < 0.55) → skip; phone-photo
  but pure-white background present (white_frac ≥ 0.02) → skip; genuine
  camera capture → full Stage 1. The earlier moiré/glare defect gate fired
  on ~100% of pages (white paper trips the same metrics) and was replaced by
  the white-fraction test (100% defect recall, ~10% FP on the labeled sweep).
- **Camera-branch overhaul (2026-07-05, defect samples in test_images/real;
  clean/screenshot paths untouched — benchmark unaffected):**
  - **Shadow removal scaled ratio×128, not ×255** → every camera capture
    collapsed to flat mid-gray (out_mean ~130, std ~15; paper rendered gray).
    Fixed to ×255: paper white, text dark (receipts mean 130→235).
  - **Glare inpainting ran AFTER shadow flattening** → with paper now
    properly white, the L>230 glare mask covered 99.7% of the page and Telea
    inpainted the entire document into mush. Reordered glare BEFORE shadow,
    plus a >20%-mask guard (real specular glare is local; a page-sized mask
    means the detector latched onto paper).
  - **Rectification had no content validation**: on a receipt-on-dark-wood
    photo it warped a wood-grain quad across the canvas and destroyed the
    page. Added convexity + corner-angle (50–130°) checks and a
    document-containment gate (the quad must hold ≥55% of the bright-region
    mass and be ≥35% bright inside) — background quads now rejected, photo
    falls back to the unwarped original.
  - **Moiré FFT notch ghosted text** — on paper photos the strongest
    high-frequency spikes ARE the text-line periodicity, so the filter
    partially erased the text it was supposed to protect. Now opt-in
    (`PRISM_MOIRE=1`) for photographed screens only.
  - **CLAHE made conditional** (skip when grayscale std ≥ 45): after the
    fixed shadow divide most captures are already well-separated and CLAHE
    only re-amplified wood grain / paper noise.
  - **EXIF myth resolved**: cv2.imdecode APPLIES the JPEG orientation tag
    (verified tag-6 portrait receipt loads upright) — an explicit transpose
    we briefly added double-rotated and was removed.
  - **Perf**: pre-cap camera captures to 1800px shorter side BEFORE the
    correction stack (was capped only at the end, so inpaint/FFT/Hough ran
    on 9MP), and the 250-DPI resize that UPSCALED small captures 2.6×
    replaced by a mild ≤1.5× floor for <900px. Defect-sample latency:
    receipts 42s→16s/10s, glare pages 12.5s→1.0s.
  - **Product empty-output rescue**: the whole-page-OCR fallback existed only
    in the benchmark runner; orchestrate emitted an EMPTY document when
    layout found nothing (receipts). Ported (`PRISM_PAGE_OCR_FALLBACK=1`):
    the receipt now yields its full itemization end-to-end.
- **Fidelity image** — all Picture/formula crops come from a pre-destructive
  copy so OCR sees original pixels. Kept.

- **Defect-set round 2 (2026-07-05, 44 user-supplied defect images)** — three
  camera-branch fixes, benchmark pinned byte-identical:
  - **Routing rescue**: shadowed captures with >2% pure white (bright paper
    beside the shadow) took the clean-digital shortcut and got ZERO
    corrections (8 of 44 images, e.g. heavy corner shadows at white_frac
    0.022–0.42). New `_paper_shadow_dim` metric (4×4 grid, per-cell p95
    gray of low-saturation bright pixels; shadows dim PAPER, page design
    does not) reroutes at dim>0.05. **Cannot be made benchmark-safe as a
    global default** — OmniDocBench PPT gradients hit dim 0.6 — so the
    benchmark runner pins `PRISM_NORM_STRICT=1` (original routing,
    verified on the 169-page shortcut population) and the product default
    gets the rescue. Fired on exactly the 8 shadowed defect images.
  - **Shadow removal no longer erases text**: the divide-by-Gaussian-blur
    background contained the text ink, so dense paragraphs under a shadow
    were brightened text-and-all (hand-shadow sample lost strokes).
    Now: morphological CLOSE (kernel ≈ min-side/50, > stroke width) +
    median blur = text-free illumination field; gain capped at 3× so
    shadow cores don't amplify noise. Equal-or-better on all samples.
  - **Glare inpaint per-blob gate**: L>230 caught page-scale bright-paper
    patches on matte photos and Telea-inpainted ghost smears through the
    text. True specular glare is compact: keep only blobs ≤1.5% of page
    each. Smears gone on the hand-shadow/lamp samples.
  - Remaining known misses (logged, not fixed): one bottom-edge shadow
    with dim=0 (16-cell grid too coarse at the extreme edge); tilted
    calibration-chart photos on dark backgrounds still not rectified
    (quad validation rejects them; low priority for document workloads).
- **Recognition-verified normalization + the invariance finding
  (2026-07-05)** — `normalization/verified.py`: every camera-branch
  correction is a proposal gated by a DBNet probe (PP-OCRv6 det @640px,
  ~120ms; accept only on ≥2% confident-text-surface gain). Synthetic-defect
  study (40 pages EN/ZH, GT block boxes valid, three protocols):
  **PP-OCRv6 is nearly invariant to photometric defects** — a 0.18×
  corner shadow + noise costs only +0.004 block edit; open-loop correction
  taxes EVERY condition including clean (0.177→0.184); verified gating
  tracks the no-correction baseline at page level (0.600 vs open 0.603
  clean; 0.604 vs 0.608 glare). Conclusion adopted in the paper:
  photometric normalization = verification-gated RESCUE (real defect set:
  8 routing rescues, stroke-erasure rejections, receipts readable), not a
  default enhancement. Three experiment iterations logged: block-level
  mild (no signal), block-level harsh + strict accept (invariance
  finding), page-level (verified≈none<open).

## OCR / language routing

- **GT-hint language routing** in the benchmark (standard for pipeline
  systems); the product probes with both engines and counts CJK codepoints.
- **traditional_chinese fell through to the ENGLISH engine** (text edit
  0.913, table TEDS 44 on those pages) — routing fixed 2026-07-04 (0.913 →
  0.379).
- **GOT-OCR2 "high-quality mode"** (1.4 GB VLM, full-page) — an early
  quality escape hatch; GPU-only in practice, opposite of the efficiency
  thesis. Removed 2026-07-04.

- **PP-OCRv6-small swap (2026-07-05)** — block-level A/B on 1354 GT text
  blocks (notes + en_ch_mixed + EN/ZH controls), engines built with the
  pipeline's det params (limit 1280, box_thresh 0.3):

  | arm | EN | ZH | mixed | notes |
  |---|---|---|---|---|
  | pipeline v4 (en-v4 rec for EN) | 0.101 | 0.101 | 0.190 | 0.200 |
  | rapidocr 3.9 engine, defaults | 0.096 | 0.165 | 0.257 | 0.240 |
  | **v6 models in 1.4.4 engine** | **0.059** | **0.086** | **0.155** | **0.132** |

  Two findings: (a) the models are the win, the 3.x engine's default params
  (shorter det side, different thresholds) throw half of it away — so we swap
  model files only (v6 det+rec read their charset from ONNX metadata, which
  rapidocr_onnxruntime 1.4.4 supports via `get_character_list`); (b) the v6
  unified charset beats even the dedicated EN v4 rec by 42%, so the EN/CJK
  engine split collapses to one model pair. +20 MB net weights. Kill switch
  `PRISM_OCR_V6=0`. Also ~2× faster rec.

## Empty pages / detector pathologies

- **Whole-page Picture/Formula boxes** consumed every real det via
  containment suppression → 56 pages emitted nothing (33 PPT slides).
  Fixed: near-page Pictures don't consume; giant Formulas dropped before
  overlap resolution; an outer Formula containing ≥2 higher-conf Formulas
  loses to the chips; whole-page OCR fallback when output < 30 chars.
  Text on those pages 0.749 → 0.308. The 19 unrescued are stylized textbook
  covers where even raw RapidOCR reads zero lines.

## Milestones (official OmniDocBench v1.6 full, 1651 pages)

| run | date | Overall | text↓ | CDM | TEDS | RO↓ |
|---|---|---|---|---|---|---|
| v9 (baseline of record) | 2026-07-01 | 70.37 | 0.1575 | 56.90 | 69.96 | 0.3234 |
| v10 (FML_V2+TBL_V2+rescues) | 2026-07-04 | 78.46 | 0.1419 | 78.11 | 71.46 | 0.2864 |
| v13 (fml_v4 answer-key + XY-cut + marginalia + plain emphasis + norm overhaul) | 2026-07-05 | 80.43 | 0.1358 | 83.46 | 71.40 | 0.2409 |
| **v14 (RapidTable + PP-OCRv6 + formula sanitizer)** | 2026-07-05 | **83.55** | **0.1203** | **83.84** | **78.83** | **0.2383** |

v13 deltas vs v10: CDM +5.35 (answer-key rule recovered the 124-formula pages),
RO −0.046 (XY-cut), text −0.006; TEDS flat (no table change in v13 — RapidTable
and PP-OCRv6 land in v14).

v14 deltas vs v13: TEDS +7.43 (RapidTable — exactly the subset prediction),
text −0.0155 (PP-OCRv6), CDM +0.38 (sanitizer), RO −0.003. Perf IMPROVED:
median 3.04 s/pg (was 4.71 in v10), peak RAM 2.61 GB (dual-worker benchmark
config). Weights ~283 MB (PPDL 124 + OCRv6 31 + Texo 79 + SLANet-plus 7.4 +
TATR-fallback 30; en-v4 rec no longer loaded). Leaderboard position: between
POINTS-Reader-3B (83.37) and Nanonets-OCR-s-3B (83.61); MinerU-Pipeline 86.47
remains ahead. v9→v14 = +13.2 Overall in five days.

**v14 on the v1.5 cut: 85.73** (text 0.1080, CDM 86.33, TEDS 81.67, RO
0.2305; directional — v1.6/1.7 matcher on the v1.5 subset): second among all
pipelines there, 1.0 behind PP-StructureV3 (8 GB / 58 s-page on CPU), ahead
of Nanonets-OCR-s 85.59, MinerU2-VLM 85.56, GPT-5.2 85.50.

Marker 78.44 / MinerU-Pipeline 85.75 on the same set. PRISM: 245 MB weights,
CPU-only, median 4.71 s/page, peak RAM 2.24 GB.
