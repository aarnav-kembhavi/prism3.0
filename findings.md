# Findings — S2 aggregation, CDM weighting, Figure 1

**2026-08-04** · No benchmarks run. Every number below is read from the stored annotation file or the stored harness output; nothing was re-scored, adjusted or reconstructed to make figures agree.

---

## TASK 1 — the two missing pages

### Answer

**The missing value is `other`, and it is 2 pages.**

| Page ID                                         | language | data_source | layout       |
| ----------------------------------------------- | -------- | ----------- | ------------ |
| `page-0b772bee-71ec-4d4a-a406-d7691b1a6e86.png` | `other`  | magazine    | other_layout |
| `page-19300c18-7995-4099-a94c-b4c61d7622cb.png` | `other`  | magazine    | other_layout |

### (a) Every distinct `language` value in the annotation JSON

Read directly from `data/omnidocbench_full/OmniDocBench.json` (1651 page records, `page_info.page_attribute.language`):

| language | pages |
|---|---:|WE HAVE WE A
| `simplified_chinese` | 765 |
| `english` | 755 |
| `en_ch_mixed` | 116 |
| `traditional_chinese` | 13 |
| **`other`** | **2** |
| **TOTAL** | **1651** |

The attribute is present on all 1651 pages — there are no null or absent values. There are **five** distinct values, not four.

### (b) What S2 is missing

S2's four rows sum to 755 + 116 + 765 + 13 = **1649**. The `other` row (2 pages) is the entire deficit. S1 is unaffected because it groups by `data_source`, which has ten values that sum to 1651 — and both of these pages are `magazine`, a type S1 already lists.

### (c) Where the row was lost — **not where you expected**

I looked for all three of the causes you named. None of them exists:

- **No hard-coded four-language list anywhere in the repo.** The only hard-coded language list is `benchmarks/compare/collect_metrics.py:27`, which lists _three_ (`english`, `simplified_chinese`, `en_ch_mixed`) and feeds a different comparison table, not S2.
- **No groupby that drops unmatched rows.** `benchmarks/make_report.py:150` `get_group()` tries `data[el]['group'][key]` first, then falls back to `get_page_prefix()`, which iterates _every_ key with the `language: ` prefix and filters only on `isinstance(v, (int, float))`. Nothing language-specific. I checked the harness output: `data[el]['group']` contains only `sample_count`, so the fallback always runs and would return all five.
- **No filter without an else branch** on the S2 path.

**The harness itself is correct and did emit the row.** From `odb_full_v20_quick_match_metric_result.json`:

```
text_block.page.Edit_dist    : ALL, language: en_ch_mixed, english, other,
                               simplified_chinese, traditional_chinese
reading_order.page.Edit_dist : ALL, language: en_ch_mixed, english, other,
                               simplified_chinese, traditional_chinese
```

`display_formula` and `table` legitimately have no `other` key, because neither page carries a GT display formula or a GT table.

The four-row form appears for the first time in **`docs/section_scores_odb_full_v20.md:22–29`**, and `paper/sec/X_suppl.tex:43–46` reproduces those four rows verbatim. No script writes `docs/section_scores_*.md` — I searched every `.py`/`.ps1`/`.sh`/`.ipynb` in the tree and the git history, including a deleted `generate_final_report.py` recovered from the object store (it has no language handling at all).

**Conclusion: this is a hand-transcription omission at the point `section_scores_odb_full_v20.md` was written, not a code defect.** The complete data was sitting in the harness output the whole time. I am not going to name a culprit script that does not exist.

### (d) Per-page metrics for the two pages

|              |          `page-0b772bee…` |          `page-19300c18…` |
| ------------ | ------------------------: | ------------------------: |
| Text edit ↓  |                   0.13139 |                   0.03622 |
| Order edit ↓ |                   0.23529 |                   0.00000 |
| CDM ↑        | — (no GT display formula) | — (no GT display formula) |
| TEDS ↑       |           — (no GT table) |           — (no GT table) |

Mean of the two: text 0.08380, order 0.11765 — which reproduces the harness's `language: other` group values exactly (0.083804, 0.117647).

### (e) Corrected Table S2

| Language            |        N |    Text ↓ | CDM ↑ | TEDS ↑ |   Order ↓ |
| ------------------- | -------: | --------: | ----: | -----: | --------: |
| english             |      755 |     0.070 |  90.9 |   75.1 | **0.126** |
| en_ch_mixed         |      116 |     0.080 |  93.3 |   81.5 |     0.152 |
| simplified_chinese  |      765 |     0.098 |  77.0 |   83.0 |     0.201 |
| traditional_chinese |       13 |     0.212 |     — |   71.3 |     0.324 |
| **other**           |    **2** | **0.084** | **—** |  **—** | **0.118** |
| **TOTAL**           | **1651** |           |       |        |           |

LaTeX row to insert into `X_suppl.tex` after `traditional\_chinese` (not applied — no `.tex` was modified):

```latex
other                & 2   & 0.084 & --   & --   & 0.118 \\
```

**One further discrepancy while I was in there:** S2 currently prints english Order as **0.124**; the harness value is **0.12624**, which rounds to **0.126**. The table above uses 0.126. Every other published cell matches the harness to the printed precision.

---

## TASK 2 — CDM aggregation

### Answer: yes. The harness aggregates CDM over **formula instances**, not pages.

Verified against the raw dump, not inferred. For `odb_full_v20`:

| quantity                                                         |                value |
| ---------------------------------------------------------------- | -------------------: |
| formula instances scored (`display_formula_per_sample_CDM.json`) |             **2352** |
| pooled mean over those 2352 samples, computed here               | **0.881154 → 88.12** |
| harness `display_formula.all.CDM.all` (the headline)             | **0.881154 → 88.12** |
| — identical to 9 decimal places                                  |                    ✓ |
| formula-bearing pages                                            |              **313** |
| page-average (mean of per-page means), computed here             | **0.869730 → 86.97** |
| harness `display_formula.page.CDM.ALL`                           | **0.869730 → 86.97** |
| — identical                                                      |                    ✓ |

So the two keys are exactly what their names say: `all.CDM.all` is a **pooled per-formula mean**, `page.CDM.ALL` is a **page-average**. `scratchpad_runs/ablation_gates/overall.py:16` — the script that produced every Overall in the paper — reads `all.CDM.all`, i.e. the pooled figure. S1's CDM column is the page-average family.

### Where your 1214 and 85.89 come from

Six of S1's ten document types carry a CDM value. Their **total** page counts are 253 + 215 + 276 + 159 + 118 + 193 = **1214**. Weighting S1's published CDM values by those totals gives **85.66** (I get 85.66 rather than your 85.89; the gap is sensitive to the rounded inputs, and I have not adjusted anything to close it).

That weighting mixes two different denominators: it applies **total** page counts as weights to values that are means over **formula-bearing** pages only. Of those 1214 pages, only **313** carry a matched display formula. Reweighting the same published values by formula-bearing pages gives **86.97** — which lands exactly on the harness page-average, as it should.

### S1's CDM column under the headline (pooled) weighting

Recomputed from `odb_full_v20_quick_match_display_formula_per_sample_CDM.json`:

| data_source         | formula-bearing pages | formula instances | **POOLED (headline weighting)** | page-average (as published in S1) |
| ------------------- | --------------------: | ----------------: | ------------------------------: | --------------------------------: |
| PPT2PDF             |                    40 |                99 |                       **87.57** |                             88.57 |
| academic_literature |                    56 |               597 |                       **89.59** |                             89.11 |
| book                |                   113 |               779 |                       **84.39** |                             83.74 |
| colorful_textbook   |                    16 |                91 |                       **86.30** |                             81.99 |
| exam_paper          |                    82 |               758 |                       **91.72** |                             91.15 |
| note                |                     6 |                28 |                       **70.56** |                             73.50 |
| **ALL**             |               **313** |          **2352** |                       **88.12** |                         **86.97** |

The pooled column sums to the headline exactly. The largest per-type divergence is `colorful_textbook` (+4.3), where 91 formulas are concentrated on 16 pages.

### Are S1 and the headline from different runs? No.

Table 1 (`paper/sec/4_experiments.tex:27`) prints PRISM CDM as **88.12**, which is `odb_full_v20`. S1 and S2 are captioned v20. **They are the same run.**

The **88.14** in your question is the `odb_ablA_full` (v21) number, which I reported in `ABLATION_PROVENANCE.md` — a different run, not what Table 1 prints. So the apparent 88.14-vs-85.89 gap is two separate effects stacked: a v21 number compared against v20 tables (0.02), and pooled-vs-page weighting applied with the wrong denominator (the rest).

**The real, legitimate discrepancy is 88.12 vs 86.97 — 1.15 points — and it is purely the weighting.** Both are correct numbers for what they measure. If you want S1's CDM column to be reconcilable with the headline, use the POOLED column above; if you want it to describe typical per-page quality, keep the current column and state the weighting in the caption. I have changed nothing.

---

## TASK 3 — Figure 1

### Both premises are wrong, and I did not want to silently "fix" a figure that does not say what you think

**(i) `docs/normalise.png` is not Figure 1, and is not in the paper at all.** There is no `\includegraphics` of it anywhere, and no `.tex` file in `paper/` or `paper_overleaf/` contains the string `normalise`. The paper's figures are:

|          | label                                 | source                   | kind                                     |
| -------- | ------------------------------------- | ------------------------ | ---------------------------------------- |
| Figure 1 | `fig:frontier` (`1_intro.tex:8`)      | `paper/fig/frontier.tex` | TikZ/pgfplots — efficiency frontier plot |
| Figure 2 | `fig:arch` (`3_method.tex:8`)         | `paper/fig/arch.tex`     | TikZ — architecture overview             |
| Figure 3 | `fig:splice` (`4_experiments.tex:48`) | `paper/fig/splice.tex`   | TikZ                                     |

`normalise.png` and its near-twin `normalisation_pipeline.png` are docs-only assets.

**(ii) Neither PNG contains the string you quoted.** I rendered and read both. There is **no verification-gate box and no mention of Tesseract** in either. Both diagrams depict an _earlier_ architecture — deskew, modality detection, white balance, rectification, shadow/glare/moiré, CLAHE, DPI resize, plus a Stage 1.5 per-region pass — with no probe and no gate anywhere. That is why they contradict Sec 3: they predate the verified-normalization work entirely, not because a label is wrong.

I could not find `Tesseract v5 Fast` anywhere in the repo. The only Tesseract references are in the DocUNet rebuttal scripts (`scripts/rebuttal/phase5_docunet.py:9`), where it is correctly named as the _canonical external protocol_ that was blocked, not as PRISM's probe.

**(iii) There is no source file to hand back to you.** No `.drawio`, `.svg`, `.pptx`, `.vsdx`, `.excalidraw`, `.ai` or `.fig` exists anywhere in the tree. Both PNGs are flat 1536×1024 RGB rasters with empty PNG metadata — no generator, no editable original. So neither of your two paths (code, or a drawing file you edit) applies.

### What I built instead

A **new vector figure from scratch**, in code, showing the pipeline the paper actually describes — including the verification gate that the old raster lacks entirely.

|               |                                                                                                                               |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Source        | **`figures/normalisation_verified.tex`** (TikZ, editable, commented with the `verified.py` line numbers each value came from) |
| Vector output | **`figures/normalisation_verified.pdf`** (6.59 × 6.17 in, 105 KB)                                                             |
| Raster output | **`figures/normalisation_verified_300dpi.png`** (300 DPI, 308 KB)                                                             |

Content is read out of the source, not from the old diagram:

- probe model `weights/PP-OCRv6_det_small.onnx` (`verified.py:32`), 640 px longer side (`:33`)
- accept rule `score_after >= 1.02 * score_before` (`:38`, `:151`)
- both probe-trust guards: shorter side < 450 px (`:48`, `:123`) and `score_before == 0` (`:130`)
- the forced correction order white balance → rectify → glare → shadow → CLAHE-if-std<45
- the pre-destructive fidelity copy that recognizer crops read from

**Your change (a)** is satisfied by construction: the gate box reads **"Run PP-OCRv6 detection probe (9.9 MB, in-inventory)"**. (9.9 MB decimal = the 9.47 MiB on disk — consistent.)

**Your change (b), legibility:** the figure is drawn at 6.59 in wide, so at `\textwidth` in a two-column layout it scales **up** slightly (1.06×) and its 10 pt body type renders at **10.6 pt**. For comparison, `fig:arch` is currently `\resizebox{\linewidth}` inside a single-column `figure` — 7 pt type squeezed into ~3.25 in — and the old raster is a fixed 1536 px, i.e. 219 DPI at `\textwidth`, where its smallest labels are only a few points tall. This is well past the ~40 % you asked for; I dropped the decorative per-step thumbnails and the legend to buy the room, as you authorised.

### One thing worth your attention

`fig:arch` — the architecture figure that _is_ in the paper — has a plain "Normalize" box and **no verification gate at all**, while `3_method.tex:29` says _"Verification-gating is not a preprocessing trick but the system's organizing principle."_ The paper's main diagram omits its headline contribution. `figures/normalisation_verified.pdf` can be dropped in as a second method figure, or its gate block folded into `arch.tex`. I have not modified `paper/fig/arch.tex` or any other `.tex`.
