# Deep weakness analysis — v14/v15 per-page results (2026-07-06 overnight)

Basis: `odb_full_v14` per-page/per-sample result JSONs (v15 = v14 + uncovered-text
rescue; table/formula analysis carries over unchanged). Upside = Overall points
recoverable if the bucket is fixed to the stated target.

## Quantified lever map (sorted by upside)

| # | Bucket | n | Current | Target | Upside (Overall pts) | Root cause |
|---|---|---|---|---|---|---|
| 1 | English tables | 116 | TEDS 0.460 | 0.90 | **+2.56** | Newspaper agate tables: detection misses + structure collapse. Struct-only is 0.566 → structure, not cell content |
| 2 | — of which newspaper tables | 55 | TEDS 0.239 | 0.90 | +1.82 | ~35 near-zero tables on 4 pages (BostonGlobe p25 ≈24 tables, WSJ p18, Chicago p19, Times UK p31) — dense stock-listing pages |
| 3 | Book formulas | 621 | CDM 0.763 | 0.88 | **+1.03** | Texo truncation at 256 tokens on matrices/determinants + CJK hallucination |
| 4 | ZH formulas | 216 | CDM 0.648 | 0.88 | +0.71 | 100 formulas have CJK text in GT; Texo emits unrelated Greek (`未检测` → `\Gamma_i^k`). 55 "empty preds" are matcher misses caused by same garbage |
| 5 | ZH text pages | 710 | edit 0.1265 | 0.08 | +0.71 | diffuse (OCR accuracy) |
| 6 | PPT + textbook + exam text | 591 | 0.139–0.172 | 0.08 | +0.88 combined | diffuse; colorful backgrounds, handwriting |
| 7 | academic tables | 44 | TEDS 0.628 | 0.90 | +0.60 | structure on dense academic tables |
| 8 | trad-ZH text | 12 | edit 0.3015 | — | +0.06 | vertical text (V3 has vertical_text class) |
| 9 | historical_document | 5 | edit 0.768 | — | +0.07 | tiny population, degraded scans |

Reading order (RO 0.2383) is **not** part of the Overall formula — but the same
detection-geometry failures that scramble RO also scramble text matching on
newspapers (partial 580 / extra 348 in MinerU-gap forensics).

## Fix plan (tonight)

1. **PP-DocLayoutV3 swap** (`PRISM_PPDL_V3=1`, net-zero size: 124MB replaces
   124MB plus-L). Measured on the 4 worst newspaper-table pages, tables
   detected @conf>0.5: BostonGlobe **78 vs 13**, WSJ **4 vs 0**, Chicago
   **20 vs 9**, Times UK above-gate vs below-gate. Also brings: built-in
   reading order (`read_order` output column, wired via `PRISM_RO_MODEL`),
   `vertical_text` class (trad-ZH), `inline_formula` class (future),
   column-accurate newspaper boxes (kills colsplit heuristic need).
2. **CJK formula hybrid** (`PRISM_FML_CJK`, default on): on CJK pages, probe
   each formula crop with line OCR; ≥2 CJK chars → emit OCR-derived LaTeX
   (`\text{}`-wrapped) instead of Texo hallucination.
3. **Texo decode cap 256→512** (`PRISM_FML_MAXTOK`): 256 truncated real
   matrix formulas; only formulas that legitimately pass 256 pay extra time.
4. Rejected on size: SLANeXt wired/wireless (350MB each), OCRv6-medium
   (+112MB, A/B'd noise earlier).

## A/B results (2026-07-06, all on subsets below)

**V3 layout swap** (211-page split subset, v15 baseline → arm):

| arm | text↓ | TEDS(page)↑ | struct↑ | RO↓ |
|---|---|---|---|---|
| v15 baseline (plus-L + geometric RO) | 0.1102 | 65.46 | 77.26 | 0.3526 |
| V3 + geometric RO | 0.1124 | 73.99 | 84.32 | 0.3985 |
| **V3 + model RO (v16 config)** | **0.0970** | **73.99** | **84.32** | **0.1707** |

- V3 detection alone: **+8.5 TEDS** (newspaper agate tables recovered), text flat,
  RO *worse* — V3's finer boxes fragment geometric ordering.
- V3 + its own read_order: RO **halved** (newspaper 0.391→0.158), text −0.013
  (correct order improves matcher adjacency merging). Table win unchanged.
- Bug found on the way: the benchmark runner's `_layout_from_cache` dropped
  `read_order`, so the first "model RO" arm silently tested geometric RO —
  detected because two supposedly-different arms scored byte-identical.

**Formula fixes** (73-page formula subset, edit-dist proxy; CDM at v16 full run):

| arm | fml page↓ | fml ZH↓ | fml EN↓ | text↓ |
|---|---|---|---|---|
| v15 baseline | 0.4802 | 0.5448 | 0.3561 | 0.1980 |
| CJK hybrid + maxtok 512 | **0.4182** | **0.4961** | **0.2687** | 0.1971 |

- EN −0.087 = the 256→512 token-cap fix (long matrices no longer truncate).
- ZH −0.049 = OCR-hybrid `\text{}` replacement (39 pages fired).
- Text untouched → no collateral damage.

**v16 config**: `PRISM_PPDL_V3=1` + defaults (model RO, FML_CJK, MAXTOK 512,
text rescue). Weights unchanged at ~283MB (V3 replaces plus-L 1:1).

## Validation subsets

- `_split_pages.json` (211: 151 newspapers + 60 book/academic controls) —
  arm `preds/v3_split` = pure V3 (FML changes pinned off for attribution).
- `_fml_cjk_pages.json` (73: 43 pages with CJK-GT formulas + 30 EN
  formula-heavy controls incl. long-GT for the token-cap test) — arm
  `preds/fml_cjk_v1` vs v15 baseline (`eval_fmlcjk_base.yaml`).
