# PRISM — Full Project Context

> The single reference for what PRISM is, how every component works, why each
> decision was made, and what was measured. Companion docs: `paper.md` (log of
> every abandoned approach), `docs/paperresults.md` (results vs. SOTA).
> Current state: **OmniDocBench v1.6 Overall 83.55** (v14 — parity with the
> 3B-VLM class: POINTS-Reader 83.37 < PRISM < Nanonets-OCR-s 83.61; v1.5-cut
> 85.73, 2nd pipeline behind PP-StructureV3) at ~283 MB weights, CPU-only,
> **3.0 s/page median**, 2.6 GB peak. Last updated 2026-07-05 (v14: RapidTable
> primary tables, PP-OCRv6 text, formula sanitizer, recognition-verified
> normalization for camera captures).

## 1. What it is

PRISM (Pipeline for Robust Image-to-Structured Markup) converts a document
page image (photo/screenshot, PNG/JPG) into structured LaTeX (`main.tex` +
compiled PDF) and OmniDocBench-style Markdown. It handles text, display math,
tables (with row/col spans), pictures, captions, multi-column layouts, and
reading order — in English and Chinese.

**Design thesis:** own the *efficiency corner* of the document-parsing
frontier. Every competitor above PRISM on the leaderboard runs 0.9B–241B-param
VLMs or multi-GB GPU pipelines; PRISM runs specialist models totalling
**~283 MB on CPU**, ~3.0 s/page, ≤2.6 GB RAM. Every design decision trades a
little accuracy ceiling for a lot of footprint.

**Platform:** Windows 11 tested, Python 3.12, onnxruntime CPU; no torch in any
inference process.

## 2. Architecture (stage by stage)

```
image → [normalization] → [layout detection] → [detection postprocess]
      → [formula_v2 pass] → [content extraction: OCR / math / tables]
      → [reading order + assembly] → main.tex → (pdflatex/xelatex) PDF
                                   → tex_to_md → OmniDocBench markdown
```

Shared core: `pipeline/page_core.build_document()` is used by BOTH the product
CLI (`pipeline/orchestrate.py`) and the benchmark runner
(`benchmarks/run_omnidocbench.py`) — routing/column/assembly fixes land once.
(Decision: the two paths drifted early on and produced divergent bugs; a
shared core ended that.)

### 2.1 Normalization (`normalization/`)

Always: **deskew** (projection-profile angle search ±15°) + **modality
detection** (256-bin grayscale histogram entropy; <0.55 → screenshot).
Then a three-way gate decides whether the corrective stack runs:

1. **Screenshot** → skip all corrections (cap longest side 1800 px).
2. **Phone-photo but pure-white background present** (`white_frac ≥ 0.02`,
   LAB L>250) → clean digital doc mislabeled by entropy → skip corrections.
3. **Genuine camera capture** (no pure white — real sensors never produce it)
   → pre-cap to 1800px shorter side (heavy steps must not run on 9MP), then:
   gray-world white balance → perspective rectification (morph gradient /
   Hough / contour strategies, each candidate quad validated for convexity,
   corner angles 50–130°, and containment of the bright document region —
   background quads previously warped wood grain over the page) → glare
   inpainting (LAB L>230, Telea; skipped when the mask exceeds 20% of the
   frame — that's paper, not glare) → shadow removal (divide-by-blur, ratio
   scaled ×255 so paper stays white — a ×128 bug had been collapsing every
   capture to flat gray) → CLAHE only if contrast is still low (std<45) →
   mild ≤1.5× upscale floor for tiny captures. Moiré FFT notch is opt-in
   (`PRISM_MOIRE=1`): on paper photos it erased text-line frequencies.
   cv2 applies JPEG EXIF orientation itself — no explicit handling needed.

**Why the gate:** the ablation series (see paper.md) showed skipping Stage 1
on digital pages wins on EVERY metric (+11.6pp formula EN alone) — the
corrections destroy clean renders. An earlier moiré/glare-metric gate fired on
~100% of pages (white paper trips the same statistics); the white-fraction
test achieved 100% defect recall at ~10% FP on a labeled sweep.

**Recognition-verified mode (2026-07-05, product default)**: pixel statistics
cannot separate lighting defects from page design (PPT gradients out-score
real shadows on every metric tried), so in product mode each camera-branch
correction is a PROPOSAL gated by a DBNet probe (`normalization/verified.py`,
PP-OCRv6 det at 640px, ~120 ms): kept only if the confident-text-surface
score improves ≥2%. Corrections are guilty until proven innocent — clean
pages reject everything by construction. A `_paper_shadow_dim` reroute also
rescues shadowed captures that the white-frac shortcut mislabels clean
(dim>0.05 of local paper white). The benchmark runner pins
`PRISM_NORM_STRICT=1` = the exact open-loop routing above (verified on the
169-page benchmark shortcut population — byte-identical). Shadow removal now
divides by a TEXT-FREE background (morph-close + median, gain cap 3×) and
glare inpainting keeps only compact blobs ≤1.5% of page each.

**Fidelity image:** a copy taken before the destructive steps; all Picture and
formula crops are cut from it so recognizers see original pixels.

### 2.2 Layout detection (`pipeline/ppdoclayout_onnx.py`)

**PP-DocLayout_plus-L** (RT-DETR, 20 classes, 800×800 stretch-resize,
NMS-free, 124 MB ONNX) run through raw onnxruntime — validated box-for-box
against the Paddle reference. Labels map to the PRISM vocabulary
(`PPDL2PRISM`); `formula_number` and `seal` are dropped.

Thresholds (decision — measured, see paper.md): **0.50 for everything except
Formula at 0.30**. Coverage recall of GT formulas is 92.4% at 0.30 vs 87% at
0.50; 0.15 was also measured and rejected (no CDM gain, text cost).
PP-StructureV3 ships the same model at formula-threshold 0.30.

History: replaced a two-detector combo (YOLOv11n-DocLayNet + DocLayout-YOLO);
a dedicated MFD formula detector was evaluated and rejected (paper.md).

### 2.3 Detection postprocess (`pipeline/detection_postprocess.py`)

conf filter (0.3 global; Formula exempt down to its own gate) → giant-Formula
drop (>50% of page = background FP) → class-aware NMS (IoU 0.4) →
chips-beat-blocks pre-pass (an outer Formula containing ≥2 higher-conf
Formulas is a spurious merged block → dropped) → cross-class containment
resolution with protections:

- Section-header/Page-header/Title/Caption never consumed (bold headers get
  double-detected inside Text boxes).
- **Formula inside Text never consumed** (it's a display equation inside a
  paragraph region; the formula_v2 pass masks it out of the text crop
  instead). This rule alone was silently deleting recovered formulas.
- **A Picture covering >70% of the page never consumes anything** — PPT
  slides/colorful textbooks are detected as one background Picture that used
  to swallow every real det (56 pages emitted nothing).
- Partial cross-class overlaps: lower-confidence box clipped, EXCEPT
  Formula-vs-Text (left intact; masking handles it — clipping desyncs the
  bbox from its already-cut crop).

Then same-class vertical merge (gap <8 px, x-overlap >50%; List-item and
headers exempt), 2 px pad, clamp.

### 2.4 Formula pass (`pipeline/formula_v2.py`) — the +21 CDM subsystem

Runs at the top of `build_document`. Root cause it fixes: the detector boxes
multi-equation stacks as ONE region while GT annotates per line (32% of GT
formulas), it fires on inline math chips at high conf, and merged crops
truncate Texo's 256-token decode.

1. **Giant-FP drop** (belt-and-braces with postprocess).
2. **Inline guard (ink-beside test):** a Formula ≥80% inside a Text det, with
   width <60% of it, is inspected on the text det's binarized crop: if the
   formula's horizontal band has prose ink beside it (>2% outside a 12%
   margin) it's inline math → dropped (left to text OCR). Display equations
   own their band. (Decision: pure geometry guards mis-killed 105 real
   equations; ink decides cleanly.)
3. **Text dedup/masking:** a Text det ≥85% covered by Formula dets duplicates
   them → dropped; partial overlaps get the formula region whited out of the
   text CROP (MinerU-style masking, no bbox surgery).
4. **Block split:** each Formula crop is Otsu-binarized and split into
   equation lines: candidate cuts at runs of ≥6 near-empty rows (emptiness
   relative to the densest row — survives tinted scans); a cut is REJECTED if
   a vertical ink run bridges it (matrix bracket / big paren: ink in ≥70% of
   gap rows in some column touching both sides); a thin segment WIDER than
   both neighbours is a fraction bar → numerator/bar/denominator fused; the
   gap-merge threshold (0.25 × median line height) uses heights measured
   BEFORE tiny-fragment merging (descender islets otherwise inflate it and
   cascade the whole block into one band). Each band is then column-split at
   horizontal gaps ≥ max(28 px, 1.4×band height) — GT annotates "equation,
   qualifier" pairs separately — and a small (<max(50 px, 8% width))
   first/last chunk across such a gap is an equation number → trimmed (GT
   excludes eq numbers; their tokens were penalizing every matched formula).
   Implementation notes: cv2 Otsu returns thr=0.0 on pure-b/w images (ink is
   `<= thr`, not `<`); ≤12 sub-dets cap (more = not an equation stack).

### 2.5 Content extraction (subprocess workers)

All heavy inference runs in persistent subprocess workers
(`multiprocessing.Pipe`, spawn), started in a background thread that overlaps
model load with normalization+detection. Benchmark uses dual OCR/math workers.

- **Text** (`text_worker.py`): RapidOCR with **PP-OCRv6-small det+rec**
  (unified EN/CJK charset read from ONNX metadata; `PRISM_OCR_V6=0` restores
  the old v4 stack). Block-level A/B vs the pipeline's v4 config: EN
  0.101→0.059, ZH 0.101→0.086, mixed 0.190→0.155, handwriting 0.200→0.132,
  ~2× faster rec. The EN/CJK engine split and the dedicated en-v4 rec are
  gone (one model pair serves both; language routing preserved for engine
  params only). Crop prep unchanged: screenshots → Sauvola only when
  background non-white; photos → autocontrast → unsharp → binarize;
  quiet-zone padding.
- **Math** (`math_worker_onnx.py`): Texo-distill (20M distilled
  UniMERNet/PP-FormulaNet-S family), ONNX encoder + merged decoder. Crop prep:
  Otsu binarize → ink-margin crop → 384 longest side → 384×384 pad. Decode:
  256-token cap (bounds hallucination cost), repetition guards, quality gate
  (tilde-spam, repeated-pattern, over-generation), row/col split retries on
  gate failure. Recognition measured NOT to be the formula bottleneck (ties
  PP-FormulaNet-S at 1/3 size on clean crops).
- **Tables** (`rtable_worker.py`/`rtable_child.py`, primary): **RapidTable
  SLANet-plus** (7.4 MB) in a `.venv_rtable` stdio child process; predicts
  structure AND reads cells in one pass over the crop with its own
  PP-OCRv6-small OCR — no token-to-grid assignment. Emits final `<table>`
  HTML which passes through `latex_builder`/`tex_to_md` under quarantine.
  Chosen after per-table diagnosis: TATR shattered dense newspaper tables
  (GT 15 rows → 53 predicted) and token assignment cost 10pp content-vs-
  structure. Full-benchmark TEDS 71.4→**78.8**; catastrophic tables
  −0.01→0.57 mean. `PRISM_RTABLE=0` kill-switch.
- **Tables fallback** (`tatr_worker_onnx.py` + `table_tokens` OCR task):
  TATR v1.1-all INT8 (30 MB) with spanning cells → `\multicolumn`/
  `\multirow` → HTML colspan/rowspan; token splitting by character
  interpolation. Used when RapidTable emits no cells; the coordinate
  heuristic remains the last resort.
  (Decision trail: spans were being discarded while 39% of GT tables have
  them — spanned-table TEDS 0.603→0.647 after the fix. Table detection gate
  0.30 rejected twice, incl. WITH RapidTable: 78.63 vs 78.83.)

### 2.6 Reading order & assembly (`layout_utils.py`, `page_core.py`)

- Column count via gutter analysis (histogram of zero-coverage vertical
  slices, validated 3–8; ±tolerance 2-column heuristic for photographed
  academic papers).
- 1–2 columns: semantic DAG (geometric top-bottom/left-right baseline +
  caption→figure pairing + footnote sinking), paracol assembly for 2-col.
- **Complex pages: recursive XY-cut** (`xycut_order`) — applied when the
  gutter detector reports 3+ columns OR a busy "1-column" page (≥8 dets;
  half the newspapers land there because touching boxes hide the gutters;
  safe because XY-cut degenerates to top-down on a true single column).
  Newspapers are vertical REGIONS (masthead, article blocks separated by
  banners) each with its own columns; the old flat "full-width first, then
  columns" model scrambled them. Measured: newspaper RO 0.595→0.390 and
  newspaper text 0.153→0.139; regression on 60 book/academic pages: RO
  0.279→0.264 (no damage). XY-cut prefers horizontal cuts (top band first),
  then vertical, falls back geometric; near-page boxes excluded from cuts.
- Empty-output rescue: if the final markdown has <30 content chars, the whole
  page is OCR'd directly (`PRISM_PAGE_OCR_FALLBACK=1` default).

### 2.7 Output

- `latex_builder.py`: class→environment mapping (Formula → `\[...\]`, tables
  as booktabs tabular with spans, `xeCJK` preamble when Chinese detected).
- `tex_to_md.py`: LaTeX → OmniDocBench Markdown. **Display-math blocks are
  quarantined via placeholders through all text-mode conversions** — the
  text-mode `\\→newline` rule was destroying array/matrix row separators in
  every multi-row formula prediction (a long-standing score-suppressing bug).

## 3. Configuration surface

All validated behavior is DEFAULT-ON. Env vars remain as kill-switches:

| Var | Default | Meaning |
|---|---|---|
| `PRISM_PPDL_CONF` | 0.30 | Formula-class detection threshold |
| `PRISM_FML_V2` | 1 | formula pass + postprocess protections |
| `PRISM_TBL_V2` | 1 | table spans + token splitting + converter |
| `PRISM_RO_V2` | 1 | XY-cut ordering for 3+ column pages |
| `PRISM_PAGE_OCR_FALLBACK` | 1 | whole-page OCR when output empty |
| `PRISM_USE_PPDL_LAYOUT` | 1 | (PP-DocLayout is the sole detector) |

## 4. Ablations & key measurements (details in paper.md)

- Stage-1 normalization: skip-on-digital wins everywhere (formula EN +11.6pp).
- Formula "detection ceiling" was a strict-IoU measurement artifact; coverage
  recall 87→96% across thresholds; fixes above took CDM 56.90→78.11.
- Formula conf: 0.30 optimal (0.15 = same CDM, worse text/RO).
- MFD detector: +3.3 CDM for +167 MB/+2 s — rejected.
- Table spans: TEDS 69.96→71.46 (spanned 0.603→0.647).
- Empty-page rescue: text 0.749→0.308 on the 56 affected pages.
- traditional_chinese routing: text 0.913→0.379 on those pages.
- Latency cost of ALL v10 features vs v9: +0.17 s median (4.54→4.71).

## 5. Known limits (ranked by Overall impact)

1. **Formula ZH ~62** — Texo (20M) can't render CJK-inside-formula; the fix
   is a distilled/quantized PP-FormulaNet_plus-M-class recognizer (617 MB,
   1.3 s/formula on GPU as-is — a training project, not wiring).
2. **Tables (residual after RapidTable)**: 40 GT tables never detected (WSJ
   agate stock listings invisible to PPDL at any conf/tiling); EN TEDS 71.6
   vs ZH 82.5 — the EN gap IS the newspaper mode; 139/665 tables where TATR
   beat SLANet-plus (no cheap selector found).
3. **Reading order 0.238** vs MinerU 0.153 — theirs is a learned order model;
   XY-cut targets the newspaper mode; the residual is ZH pages and complex
   wraps (O-shaped 0.83).
4. **Handwriting / historical documents** (~45 pages, text 0.6–0.99) —
   recognizer capacity, out of scope for the current model set.
5. 19 stylized textbook covers where even raw OCR reads zero lines.

## 6. Repo map

```
pipeline/
  orchestrate.py         product CLI (per-image), worker lifecycle
  page_core.py           shared extraction + assembly core
  ppdoclayout_onnx.py    PP-DocLayout_plus-L RT-DETR (raw onnxruntime)
  detection_postprocess.py  conf/NMS/containment + all protections
  formula_v2.py          formula pass (guard/mask/split/trim)
  text_worker.py         RapidOCR subprocess (EN/CJK/mixed, table tokens)
  math_worker_onnx.py    Texo ONNX subprocess (quality gate, split retries)
  tatr_worker_onnx.py    TATR INT8 subprocess (spans)
  layout_utils.py        reading order (DAG, XY-cut), columns, crops
  latex_builder.py       LaTeX assembly
  tex_to_md.py           LaTeX → OmniDocBench markdown (math quarantined)
  models_interface.py    model singletons + table heuristic
  onnx_config.py         thread governance
normalization/           stage-1 gate + corrections + modality
benchmarks/run_omnidocbench.py  benchmark runner (GT lang hints, perf.json)
omnidocbench_eval/       official eval harness fork (utf-8 fixes only)
models/ weights/ Texo/   ONNX weights (see paperresults.md §4 breakdown)
paper.md                 experiment log (every dead end, with numbers)
docs/paperresults.md     results vs SOTA (v1.6 + v1.5 + efficiency)
docs/formula_fix_v2.md   deep-dive: the formula recall investigation
docs/pipeline_audit_2026-07-04.md  deep-dive: tables/empty-pages/routing
```

## 7. Reproducing the headline number

```powershell
# full 1651-page run (defaults = validated config), writes preds + perf.json
python benchmarks/run_omnidocbench.py `
  --gt-json data/omnidocbench_full/OmniDocBench.json `
  --images-dir data/omnidocbench_full/images `
  --pred-dir preds/odb_full_v10 --skip-eval
# official eval incl. CDM (WSL: TeX Live 2026 + ImageMagick shim required)
wsl -u root -e bash -lc 'export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH \
  CDM_PDFLATEX=/usr/local/texlive/2026/bin/x86_64-linux/pdflatex CDM_SAVE_VIS=0 \
  PYTHONPATH=/mnt/c/PROJECTS/s2l2/testprism/omnidocbench_eval; \
  cd /mnt/c/PROJECTS/s2l2/testprism/omnidocbench_eval && \
  python3 pdf_validation.py --config /mnt/c/PROJECTS/s2l2/testprism/data/omnidocbench_full/eval_cdm_v10full.yaml'
```
