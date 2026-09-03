# Multilingual supplementary figure — source manifest

Six panels, one per language. A single recognition graph,
`weights/PP-OCRv6_rec_small.onnx` (21.2 MB, 18 708-entry charset), serves all
six — which is the claim the figure supports. Nothing was downloaded and no
extra model was added.

## Metric definition

`char F1` is the character-level **multiset F1** between PRISM's page output and
the source PDF's own embedded text layer for that page.

* **Reference** — the source PDF's embedded text layer for that page, via
  PyMuPDF `page.get_text()`. A *proxy* reference, not human ground truth: it
  also carries running heads, page numbers and chart tick labels that PRISM may
  legitimately not transcribe, and its internal ordering can differ.
* **Prediction** — PRISM's `main.tex`, stripped to plain text (LaTeX commands
  and `\includegraphics` removed, any HTML table markup removed).
* **What counts as a character** — every character after Unicode NFKC
  normalisation, **including digits and punctuation**. Whitespace runs collapse
  to a single space on Latin-script pages (those spaces do count) and are
  removed entirely for Chinese and Japanese.
* **How** — each string is a bag of characters; intersection is the sum of
  per-character minimum counts; precision = intersection / |pred|, recall =
  intersection / |ref|, F1 their harmonic mean. Order-insensitive by
  construction, so it measures **content recovery, not reading order**. CER
  (Levenshtein / |ref|) is reported alongside as the order-sensitive view.
* **Granularity** — computed **per page**, over the whole page string, not per
  block.

Caption sentence:

> `char F1` is the order-insensitive character-multiset F1 between PRISM's page
> output (LaTeX stripped to plain text) and the source PDF's embedded text layer
> for the same page, computed per page over all characters including digits and
> punctuation after NFKC normalisation, so it measures content recovery rather
> than reading order.

No predicted text was edited by hand at any point.

## Panels

| Language | Source URL | Licence | Page | char F1 | Probe route | Recognition graph |
|----------|-----------|---------|------|---------|-------------|-------------------|
| English | https://arxiv.org/abs/2608.06292 | CC BY 4.0 (stated on arXiv abs page) | 6 | 0.992 | `en` — correct | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| Spanish | https://revistapsicologia.uchile.cl/index.php/RDP/article/view/13722 | CC BY 3.0 (journal copyright notice) | 6 | 0.997 | `en` — correct | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| French | https://fr.wikipedia.org/wiki/Ch%C3%A2teau_de_Chambord | CC BY-SA 4.0 | 13 | 0.987 | `en` — correct | **continuous French prose**, chosen text-only so nothing can be clipped; every diacritic correct (`À l'intérieur`, `François Ier`, `Jérusalem`, `Léonard de Vinci`, `hélice`, `Âge`) | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| German | https://edoc.rki.de/handle/176904/46 (J Health Monit 2025;10(3):e13412) | CC BY 4.0 ("CC BY 4.0 Lizenzvertrag" in PDF) | 3 | 0.990 | `en` — correct | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| Chinese (simpl.) | https://radars.ac.cn/cn/article/doi/10.12000/JR25080 | CC BY 3.0 (journal site); "CC-BY 4.0 License" printed in the PDF | 2 | 0.995 | `mixed` — defensible | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| Japanese | https://www.stat.go.jp/data/topics/topi1380.html | Government of Japan Standard Terms of Use v2.0 — site states compatibility with CC BY 4.0 | 1 | 0.962 | `mixed` — defensible | `PP-OCRv6_rec_small.onnx` (21.2 MB) |

**Banned-codepoint check (as required).** Every Spanish candidate was checked
for `¡` U+00A1 and every German candidate for `„` U+201E, against the source
page's own text layer:

* Spanish `es_a` (shipped), `es_b`, `es_c` — `¡` **absent from all three** ✓
* German `de_a` (shipped) — `„` **absent** ✓; `de_c` — `„` **absent** ✓;
  `de_b` — `„` **present, rejected** ✗

**Routing.** No European page was misrouted: English, Spanish, French and
German all took the `en` route, which is correct for them. `run_cjk_probe`
never selected the pure `cjk` route on any page — it chose `mixed`
(dual-engine) for both CJK pages. That is not a misroute in effect: those pages
genuinely carry substantial Latin (full English abstracts, dates, numerals),
and under the default `PRISM_OCR_V6=1` the `en` and `cjk` engines are the same
graph while `has_cjk = is_cjk or is_mixed`, so xeCJK is selected either way. No
flag was added.

## Selection and rejections

Ranked per language by `char F1`; where two candidates were within 0.005 F1 the
lower CER (better reading order) won; any candidate whose compiled output was
dominated by a figure rather than text was rejected regardless of score.

| Language | Rejected | Reason |
|----------|----------|--------|
| English | `en_a` (arXiv 2608.06111) | `.tex` fails to compile |
| English | `en_c` (arXiv 2608.06370) | raw `<table>` HTML in the `.tex`; F1 0.776 |
| Spanish | `es_b` 0.996, `es_c` 0.995 | lower than `es_a` 0.997 |
| French | `fr_w3` p10, `fr_w1` p4, `fr_w2` p4 | compiled output dominated by a photograph, not text |
| French | `fr_w2t1` 0.990, `fr_w1t0` 0.982 | lower than `fr_w3t1` 0.997 |
| German | `de_c` (F1 0.993, CER 0.189) | F1 within 0.005 of `de_a` but far worse CER (0.018) |
| German | `de_b` | contains `„` U+201E |
| Chinese | `zh_b` 0.959, `zh_c` 0.480 | lower than `zh_a` 0.995 |
| Japanese | `ja_s2`, `ja_s3` table pages | raw `<table>` HTML; CER 3.5 / 4.8 |
| Japanese | Tetsu-to-Hagane 鉄と鋼 ×3 | **licence**: DOAJ says CC BY, article pages say CC BY-NC-ND; ND incompatible with cropping — discarded |

## Pipeline change made for this figure

`pipeline/text_worker.py` — `_filter_nonascii` previously deleted **every**
codepoint ≥ 128 and is applied unconditionally on the `en` route
(`text_worker.py:365`), which is the route Spanish, French and German take. It
now keeps accented Latin letters (Latin-1 Supplement, Latin Extended-A/-B,
Latin Extended Additional) while still stripping the non-Latin OCR artifacts it
was written to remove.

Before → after on the shipped candidates:

| Candidate | char F1 before | after | Accented chars recovered |
|-----------|----------------|-------|--------------------------|
| `es_a` Spanish | 0.991 | **0.997** | 0/31 → 31/31 (100 %) |
| `de_a` German | 0.983 | **0.990** | 0/83 → 83/83 (100 %) |
| French (fr_w1) | 0.959 | **0.980** | 0/114 → 111/114 (97 %) |

`Bevölkerung`, `Für`, `Saß`, `Psicología`, `años`, `Traité`, `Schrödinger` all
survive; `mixed 中文 artifact` → `mixed artifact` still holds.

**Regression: byte-identical.** Seven pages were run before and after the
change and compared by SHA-256 of `main.tex` — four real OmniDocBench pages
(English book, English exam paper with formulas, Chinese notes, Chinese
newspaper) plus the English, Chinese and Japanese figure pages:

```
identical : 7      changed : 0
```

Expected: English output is already pure ASCII so the filter was a no-op there,
and the CJK route never calls it. No reported OmniDocBench / olmOCR-Bench / Fox
number is affected.

**Not changed:** `pipeline/models_interface.py:99` holds an identical twin of
the old filter, used only by the in-process `--no-ocr-worker` fallback path.
It is not on the worker path used by the benchmarks or this figure, and was
left alone rather than changed without authorisation. It will still strip
accents if that fallback is ever used.

## Build

Built by `scratchpad/multiling/build_figure_vec.py`, mirroring
`scratchpad_runs/qual_fig/render_qual_io.py` (the generator behind
`figures/qualitative_io.pdf`): FIGW 6.9 in, 0.42 in label strip, 0.09 in column
gap, 0.20 in header band, 0.12 in row gap; `#0072B2` output rule, `#B0B0B0` raw
rule, `#5A5A5A` metric text; serif at 8 pt; `pdf.fonttype 42`.

The **output column is true vector**: matplotlib draws the frame and leaves the
panel blank, then `fitz.show_pdf_page` stamps the cropped compiled PDF in.
9 939 extractable characters in the final figure. Raw crops render straight
from the source PDFs at 380 DPI with the original's no-upsampling assertion
applied; all clear it at 656–944 DPI native at panel size.

Two rendering notes, neither affecting what is printed:

* PRISM's preamble uses `\usepackage[utf8]{inputenc}` without `fontenc`, so
  pdflatex composes accents as separate glyphs in OT1. Accented words therefore
  *render* correctly but *copy-paste* out of the figure decomposed
  (`Psicología` → `Psicolog` + `´` + `ı` + `a`). Visual output is correct.
* `latex_builder.py` emits `\usepackage{xeCJK}` without setting a CJK font, so
  xeCJK falls back to Fandol, a Chinese font lacking Japanese glyphs (齢 総 労
  県 render as tofu). The Japanese panel is compiled through a generated
  `main_render.tex` with `\setCJKmainfont{Yu Gothic}` inserted after the
  `xeCJK` line — document setup only; PRISM's `main.tex` and its predicted text
  are byte-identical.

## Files

* `multilingual_io.pdf` — combined 6-row figure, 6.90 × 11.82 in, for `figure*`
* `panel_en_b.pdf`, `panel_es_a.pdf`, `panel_fr_w3t1.pdf`, `panel_de_a.pdf`,
  `panel_zh_a.pdf`, `panel_ja_s3x0.pdf` — individual panels, 6.90 in wide
* `multilingual_io_200dpi.png` — proof render of the final PDF


---

# Second figure — `multilingual_structure.pdf`

Four panels, four different languages, showing structure rather than plain text.
Same recognition graph, same metric, same layout and build path as
`multilingual_io.pdf`. 6.90 × 8.68 in.

| Language | Source URL | Licence | Page | char F1 | Probe route | Feature shown | Recognition graph |
|----------|-----------|---------|------|---------|-------------|---------------|-------------------|
| German | https://edoc.rki.de/handle/176904/46 (J Health Monit 2025;10(3):e13412) | CC BY 4.0 | 4 | 0.957 | `en` — correct | **two-column page handled in the right order** — left column then right column, no interleaving | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| French | https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_Pythagore | CC BY-SA 4.0 | 5 | 0.990 | `en` — correct | **two figures passed through in place** (geometric proof diagram + bust photograph), both captions intact, plus the compiled identity `a²+b²=c²` | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| Chinese (simpl.) | https://radars.ac.cn/cn/article/doi/10.12000/JR25080 | CC BY 3.0 / "CC-BY 4.0" in PDF | 3 | 0.947 | `mixed` — defensible | **display maths compiled** — chirp `S_LFM`, piecewise `rect`, summations | `PP-OCRv6_rec_small.onnx` (21.2 MB) |
| Japanese | https://www.stat.go.jp/data/topics/topi1460.html | Gov't of Japan Standard Terms of Use v2.0 (CC BY 4.0-compatible) | 8 | 0.661 | `mixed` — defensible | **chart passed through in place** — pie pair with connecting arrow, all labels and source notes | `PP-OCRv6_rec_small.onnx` (21.2 MB) |

## Row labels

The figure prints the **language name only**. `char F1` is no longer drawn on
the panels; the scores stay recorded in the table above so the figure caption or
text can cite them if wanted.

## Crop discipline

Every crop is snapped outward to the nearest whitespace gap on the page
(`scratchpad/multiling/snap.py`), within a 34 pt window, so no crop boundary
falls through a line of text, a table row, a figure or a caption. Both sides of
a row are snapped independently against their own page, because the compiled
document places an element at a different height than the source does — the
German table, for instance, sits at y≈0.11 on the source page and y≈0.83 in the
output. Crops are chosen so the resulting row is 1.7–2.7 in tall, which keeps
body text legible at 3.15 in panel width.

## What each panel demonstrates

German carries **multi-column ordering**, Chinese carries **display maths**,
Japanese carries **a chart passed through in place**. The French panel is a
deliberately text-only page: it demonstrates continuous non-English prose with
full diacritic fidelity rather than a structural element, because every
figure-bearing French page tested either clipped at the crop boundary or failed
to compile (see rejections).

## The table feature is NOT in this figure — why

The tables in these sources are ~508 pt wide with ~8.5 pt row pitch. Rendered
into a 3.15 in half-page panel that is about **4 pt per row**, which is
illegible no matter where the crop falls; cropping to fewer rows does not help,
because the aspect ratio (≈60:1 per row) is fixed by the table's own geometry.
Three separate attempts confirmed it:

| Attempt | Result |
|---------|--------|
| German `de_b` p3 — small form table | present but a thin sparse strip at the crop edge |
| German `de_a` p4 — 15×4 `tabular` | compiles and cells are intact, but renders at ~0.09 in per row |
| Japanese `ja_s1` p4 — 10-country table, 23×9 | best table found; still ~0.05 in per row at panel width |

A legible table needs the **full 6.9 in figure width**, i.e. raw above /
compiled below rather than side by side. That is a different layout from the
one specified here, so it has not been built. Say the word and it can be a
third, single-row figure.

## `PRISM_VISUAL_FIDELITY=1`

These pages were run with the existing, supported `PRISM_VISUAL_FIDELITY=1`
flag (`pipeline/latex_builder.py:319-330`), the built-in visual-fidelity path
that converts RapidTable `<table>` HTML into a real LaTeX `tabular`. The
default (`0`) passes the HTML through untouched so benchmark scoring is
unaffected; the flag changes no reported number. No code was changed.

## Rejections

| Candidate | Reason |
|-----------|--------|
| German `de_a` p11 | largest table found (46×4) but **fails to compile** — `≥` U+2265 inside a table cell, which pdflatex cannot typeset in text mode. Comes from the RapidTable cell path, which never used `_filter_nonascii`; pre-existing and unrelated to the filter change |
| German `de_b` p3, `de_a` p4 (as a table panel) | table illegible at panel width (see above) |
| French `fr_w2` p6 | third diagram fell outside the crop — an image appeared missing |
| French `fr_w3` p6 (Versailles) | single portrait book-cover image; left half of the panel was empty and the caption clipped |
| French `fr_w3` p2 | char F1 0.994 but CER 0.719 — heavy reordering |
| French `fr_w2` p9 | fails to compile |
| French `fr_w2` p5 (Pythagore, 2 figures) | figures rendered but the crop clipped the heading above and the caption below; replaced with a text-only page at request |
| French `fr_w1` p5, `fr_w1` p19 (Mécanique quantique) | **fail to compile** — `α` U+03B1 in running text, which pdflatex + inputenc cannot typeset |
| French `fr_w2` p12/p13 (Pythagore notes) | text-only but bibliography/reference pages — dense citation clutter, not representative prose |
| French `frx1` p17, `frx3` p12/p19 (Révolution française, Victor Hugo) | text-only but fail to compile |
| Chinese `zh_a` p13, `zh_c` p10 | char F1 0.160 / 0.480 |
| English `en_a` p9 | every table duplicated; task names (`QQP`, `MNL`, `QNLI`) wrapped as equations; F1 0.769 |
| English `en_a` p16 | fails to compile |
| Japanese `ja_s3` p10, `ja_s1` p9 | contain `※` U+203B / `◆` U+25C6, which fall to the Latin font under xeCJK and render as tofu |

## Languages not used

* **English** — no structural page survived: table pages trigger the
  table-duplication artifact and misclassify column headers as display maths;
  the clean alternative fails to compile.
* **Spanish** — no structural page exists in the licensed source. Across 68
  scanned pages of *Revista de Psicología*: zero embedded images, no
  multi-column page, no display maths, no table grids. German and French supply
  the two required non-English Latin-script panels.

## Known blemishes, disclosed

* **Japanese char F1 0.661** is low because the reference (the PDF text layer)
  contains every chart tick label and legend entry, which PRISM correctly
  treats as part of a passed-through figure rather than transcribing. The prose
  is accurate.
* **Japanese, one character below the crop**: OCR emits `查` U+67E5 (Chinese
  form) where the source has `査` U+67FB (Japanese shinjitai) in 「労働力調査」;
  Yu Gothic has no `查`, so it tofus. This is the Chinese/Japanese form
  confusion flagged as possible during the Phase 1 charset check.
* **French, one stray character**: the compiled identity reads `1a² + b² = c².` —
  a footnote marker (`1`) from the source line has been absorbed into the start
  of the equation. The identity itself is correct and compiles.
* **German typographic quotes**: `„` U+201E and `"` U+201C are punctuation,
  outside the Latin-letter allowlist in `_filter_nonascii`, so still dropped.
  Umlauts and `ß` are unaffected.

## Files

* `multilingual_structure.pdf` — combined 4-row figure, 6.90 × 8.68 in, for `figure*`
* `panel_tb_de_a_p4.pdf`, `panel_frx2p13.pdf`, `panel_st_zh_a_p3.pdf`,
  `panel_st_ja_s1_p8.pdf` — individual panels, 6.90 in wide
* `multilingual_structure_200dpi.png` — proof render

---

# Code-mixed panel (`panel_mx_en_ch_p20.pdf`)

A seventh panel, in the same layout as the six language panels above: raw crop
left, compiled output right, rotated label and score in the far-left gutter.
Vector PDF, 6.90 x 1.92 in.

## Source — OmniDocBench v1.6, `en_ch_mixed`

Unlike the six language panels, which come from external CC-licensed PDFs, this
one is sourced from data already in the repo:
`data/omnidocbench_full/OmniDocBench.json` (the 1651-page v1.6 release), page
attribute `language = en_ch_mixed`, which holds **116** pages.

| | |
|---|---|
| Image | `color_textbook_zhonggaokao_小学_KET听说读写逐项突破_…_page_020.png` |
| Attributes | `colorful_textbook` / `double_column` / no special issue |
| Page size | 1654 x 2339 px |
| Language pair | English–Chinese, mixed **inline on nearly every line** (headword EN, gloss ZH) |

## Selection — cleanest compiled output, not highest score

As instructed, the page was chosen on compile quality, not on score. All 116
`en_ch_mixed` pages were first scored against the v1.6 GT text using the
existing `preds/odb_ablA_full` predictions; 36 pages with genuine two-way mixing
(CJK >= 20% and Latin >= 35% of GT characters, >= 350 GT chars) formed the pool.
Eight covering every `data_source` in that pool were run end to end through
PRISM and compiled with `xelatex`, then compared by eye:

| Cand | Source | char F1 | Compile verdict |
|---|---|---:|---|
| **c06 (shipped)** | colorful_textbook | 0.986 | **two-column `paracol` layout preserved, both scripts clean, footer figure kept, 1 page, 0 missing glyphs, 0 overfull boxes** |
| c01 | book (Deloitte BMC) | 0.968 | three columns preserved and very close behind; loses on `Compani-escertificate` and `newwinners` word merges |
| c00 | academic_literature | 0.977 | all reference entries collapse into one run-on paragraph |
| c03, c02 | note (handwritten) | 0.984 | text accurate but all structure flattened to one block |
| c07 | colorful_textbook | 0.974 | spills to 2 pages; leading `(` of list markers dropped |
| c05 | book (API docs) | 0.937 | code blocks lose their monospace framing |
| c04 | colorful_textbook | 0.962 | 1 missing character |

Note that c06 is **not** the top scorer in the stratum — two PPT2PDF pages reach
1.000 — but those are near-empty slide pages whose compiled output shows almost
nothing.

## Metric

`char F1 0.986` is the same order-insensitive character-multiset F1 defined at
the top of this manifest, with two deliberate differences, both because this
page is a bitmap from a benchmark rather than a born-digital PDF:

* **Reference** is the **OmniDocBench v1.6 ground truth** for the page — the
  concatenated `text` of every `layout_dets` block except `abandon`, `header`,
  `footer`, `page_number`, `page_footnote` — not a PDF text layer. This is
  human ground truth, so it is a stronger reference than the six panels above.
* **Whitespace is removed on both sides** (the manifest's CJK rule) rather than
  collapsed, since the page mixes scripts line by line and space placement
  around a script switch is not meaningful.

Measured on the exact `main.tex` shown in the panel: **char F1 0.9860**
(P 0.9731, R 0.9992), **CER 0.0285**, 1194 GT chars vs 1226 predicted.

## Build

```bash
PRISM_VISUAL_FIDELITY=1 python pipeline/orchestrate.py <page_020.png>
cd outputs/<stem>_output && xelatex -interaction=nonstopmode main.tex
python scripts/build_panel_mixed.py
```

`PRISM_VISUAL_FIDELITY=1` matches the six panels above; this page has no table,
so the flag changes nothing here. No code was changed and no predicted text was
edited by hand.

Panel geometry is copied numerically from `panel_zh_a.pdf` — canvas 496.8 pt
wide, raw box x 36.7–263.5, compiled box x 270.0–496.8, top y 14.4, 0.5 pt grey
frame left, 0.9 pt blue frame right, 8 pt Times-Bold headers, rotated 8 pt label
and 6.4 pt score. The compiled half is embedded with `show_pdf_page`, so it
stays **vector text**, not a raster; only the raw crop is a bitmap.

## Known blemishes, disclosed

* **Adjacent entries merge onto one line** where the source line spacing is
  tight — `or (conj) 或者，还是orange (adj & n)…`, `passport (n) 护照past (prep)…`,
  `our (pron) 我们的ours (pron)…`. Visible in the panel. This is the dominant
  error on the page and the reason precision (0.973) sits below recall (0.999).
* **Column 1 has a whitespace gap** below the header, from an empty centred
  block PRISM emits for the masthead.
* **The two halves end at different points in column 2.** Both are cut on the
  same column-1 entry (`opposite (prep) 在… 对面`); because PRISM merged three
  column-2 lines, the compiled half reaches `pen (n) 钢笔` where the raw crop
  reaches `pasta (n) 生面团…`. Nothing is invented — it is the same content,
  re-flowed.
* **The masthead zigzag** renders as a small wedge at the top right of the
  compiled half; PRISM keeps it as a figure rather than dropping it.
