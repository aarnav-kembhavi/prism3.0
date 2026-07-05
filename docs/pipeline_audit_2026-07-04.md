# Pipeline audit — tables, empty pages, routing (2026-07-04)

Follow-up to `docs/formula_fix_v2.md` (Formula CDM 56.90 → 77.83). This pass
audited everything EXCEPT formulas: mined the full-run (odb_full_v9) results
for the weakest buckets, root-caused each, fixed, and A/B-measured. All fixes
are code-only: no new models, no latency budget spent.

## Where the weakness was (full 1651-page run, by impact)

| bucket | metric | value | pages/tables | root cause found |
|---|---|---|---|---|
| tables overall | TEDS | 70.0 | 665 tables | spans never emitted; tokens straddling columns; tokens >80% width deleted; empty rows dropped |
| tables with col/rowspan (39% of GT!) | TEDS | 60.3 | 258 | TATR detects `table spanning cell` — the worker threw them away |
| empty pages | text edit | ~1.0 | 56 | whole-page Picture/Formula FP boxes containment-consume every real det |
| traditional_chinese | text edit | 0.913 | 13 | benchmark routed only `simplified_chinese` to the CJK engine; trad pages got ENGLISH OCR |
| newspaper tables | TEDS | 52.4 | 69 | (unfixed — dense rule-less tables, TATR limit) |
| handwriting/historical | text | 0.65–0.99 | ~45 | (unfixed — model capacity) |

## Fixes (env-gated `PRISM_TBL_V2=1`; empty-page fixes ride `PRISM_FML_V2`)

1. **Spanning cells** (`tatr_worker_onnx.py`): spanning-cell detections are
   snapped to the row/col grid; covered cells pool their tokens into the
   anchor; emitted as `\multicolumn`/`\multirow` in the LaTeX tabular (so the
   product PDF renders them too). `tex_to_md._table_content_to_html` parses
   them back to `colspan`/`rowspan` with positional rowspan bookkeeping.
2. **Token-to-cell assignment**: OCR det boxes are line-level, so one token
   often straddles columns — now split at column boundaries with
   character-position interpolation (`_split_token_by_cols`).
3. **`_tokens_full` width filter** (`text_worker.py`): tokens wider than 80%
   of the crop were dropped wholesale (killed full-width header cells);
   disabled under TBL_V2 (splitting handles them).
4. **Empty rows kept** (GT has `<tr></tr>`), single-column tables no longer
   nuked by the 0.85-width column filter, booktabs rules stripped in the
   converter (they were gluing onto the first cell).
5. **Empty-page rescue** (`detection_postprocess.py` + `run_omnidocbench.py`):
   - a Picture covering >70% of the page is background — it no longer
     containment-consumes the Text/Table/Formula dets inside it (33 PPT
     slides were losing ALL content this way);
   - a Formula covering >50% of the page is dropped before overlap
     resolution (same failure, PPT variant), and an outer Formula containing
     ≥2 higher-conf Formulas loses to the chips;
   - `PRISM_PAGE_OCR_FALLBACK=1` (default): if the final markdown has <30
     content chars, OCR the whole page and use that.
6. **traditional_chinese → CJK engine** (`run_omnidocbench.py`).

## Measured results

- **Tables (458 pages, 665 tables)**: TEDS page-ALL **69.96 → 71.46**;
  spanned-GT tables 0.603 → 0.647; unspanned flat (0.723 → 0.716); text on
  those pages 0.1101 → 0.1048; reading order 0.415 → 0.407. 240/633 pred
  tables now carry spans. Eval: `result/tbl_v2_quick_match_*` vs
  `result/tbl_base_v9_quick_match_*`.
- **Formerly-empty pages (56)**: text edit **0.749 → 0.308** (37/56 now
  produce content; the rest are stylized textbook covers that even raw
  RapidOCR reads as zero lines). Reading order on them 0.182. Full-set text
  impact ≈ −1.5pp. Eval: `result/empty_v2_quick_match_*`.
## CONFIRMED — full official run (v10, 1651 pages, all flags on)

| | v9 | **v10** | peers |
|---|---|---|---|
| **Overall** | 70.37 | **78.46** | Marker 78.44, MinerU-Pipeline 85.75 |
| Formula CDM | 56.90 | **78.11** (EN 84.7 / ZH 61.4) | Marker 85.2, MinerU 83.1 |
| Table TEDS | 69.96 | **71.46** | Marker 65.8, MinerU 80.4 |
| Text edit ↓ | 0.1575 | **0.1419** (tradZH 0.913→0.379) | Marker 0.157, MinerU 0.063 |
| Reading order ↓ | 0.3234 | **0.2864** | Marker 0.243, MinerU 0.154 |
| v1.5-subset Overall | 73.87 | **80.23** | PP-StructureV3 86.73 |

**PRISM 78.46 > Marker 78.44** on the current official set — at 245 MB,
CPU-only, median 4.71 s/page (v9: 4.54), peak RAM 2.24 GB, wall 2.58 h.
Run: `preds/odb_full_v10` (PRISM_FML_V2=1 PRISM_PPDL_CONF=0.30
PRISM_TBL_V2=1). Eval artifacts: `result/odb_full_v10_quick_match_*`.

## Still-open weaknesses (ranked)

1. **40 empty-pred tables** — never detected (layout) or suppressed; next
   lever is the table conf gate + a TATR-detect fallback pass.
2. **Newspaper tables TEDS 52** — dense borderless layouts; TATR capacity.
3. **Table content quality** — structure-only 0.79 vs content 0.69: cell OCR
   remains the gap after alignment (CJK cells especially).
4. **Handwriting / historical_document** — recognizer capacity, ~45 pages.
5. **Formula ZH 60.7** — Texo CJK ceiling (see formula_fix_v2.md).
