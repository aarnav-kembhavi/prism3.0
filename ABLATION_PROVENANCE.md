# Fig. 4 (`ablation_waterfall.pdf`) — provenance of the nine constants

**Generated:** 2026-08-04 · Report only. No number invented, estimated or reconstructed. No benchmark run.

## Verdict: 9 of 9 MEASURED. 0 NOT FOUND.

Every constant in `scratchpad_runs/figures_wacv/ablation_waterfall.py` `STAGES` traces to a recorded OmniDocBench v1.6 measurement of a dated build. **This corrects §5/B-2 of my previous `RELEASE_REPORT.md`**, which said the endpoint 87.29 had no independent record — it does; I had not yet located `omnidocbench_eval/result/odb_ablA_full_*`.

But see the **caveat on bar 9** below. It is the one finding here that could change paper text.

---

## Per-constant table

| # | Value | Build | Source file : line | Verdict |
|---|---:|---|---|---|
| 1 | **70.37** | v9 (baseline of record) | `paper.md:332` · `docs/paperresults.md:189` · `docs/context.md:190` · `docs/pipeline_audit_2026-07-04.md:61` | **MEASURED** |
| 2 | **78.46** | v10 (FML_V2 + TBL_V2 + rescues) | `paper.md:333` · `docs/paperresults.md:190` · `docs/context.md:191` · `docs/pipeline_audit_2026-07-04.md:61` | **MEASURED** |
| 3 | **80.43** | v13 (answer-key + XY-cut + marginalia) | `paper.md:334` · `docs/paperresults.md:191` · `docs/context.md:192` | **MEASURED** |
| 4 | **83.55** | v14 (RapidTable SLANet-plus + PP-OCRv6) | `paper.md:335` · `docs/paperresults.md:192` · `docs/context.md:193` · `docs/section_scores_odb_full_v14.md:42` | **MEASURED** |
| 5 | **85.77** | v16 (PP-DocLayoutV3 + model RO) | `paper.md:336`, `paper.md:396` ("v16 CONFIRMED 2026-07-06") · `docs/paperresults.md:193` · `docs/section_scores_odb_full_v16.md:42` | **MEASURED** |
| 6 | **86.35** | v17 (inline_formula→Formula recovery) | `paper.md:448` ("v17 CONFIRMED 2026-07-06 evening") · `docs/paperresults.md:194` · `docs/section_scores_odb_full_v17.md:42` · `README.md:7` | **MEASURED** |
| 7 | **86.37** | v19 (dense-host guard + render repair) | `paper.md:492` ("v19 CONFIRMED 2026-07-07") · `docs/paperresults.md:47,195` · `docs/context.md:197` | **MEASURED** |
| 8 | **86.62** | v20 (inline-math splicing) | `paper.md:531` ("v20 CONFIRMED 2026-07-07") · `docs/section_scores_odb_full_v20.md:4` · `docs/context.md:23,198` · `paper/sec/4_experiments.tex:27,272` (`tab:ablation` final) | **MEASURED** |
| 9 | **87.29** | v21 (`preds/odb_ablA_full`) | `omnidocbench_eval/result/odb_ablA_full_quick_match_metric_result.json` (harness output, 16 KB) · cited as v21 in `scripts/bootstrap_margin.py:4,37` | **MEASURED** (see caveat) |

### How bar 9 was confirmed

There is no `docs/section_scores_odb_full_v21.md` and no "v21 CONFIRMED" line in `paper.md` — which is why my earlier pass called it unrecorded. The measurement is instead held as the raw harness output. Reading it with the repo's own composite script:

```
> python scratchpad_runs/ablation_gates/overall.py odb_ablA_full_quick_match
odb_ablA_full_quick_match: Overall=87.29 text=0.0844 CDM=88.14 TEDS=82.15 order=0.1474
```

Read path validated against three constants whose values are independently written down:

| save name | `overall.py` output | recorded elsewhere | match |
|---|---:|---:|:--:|
| `odb_full_v20_quick_match` | 86.62 | `paper.md:531` | ✅ |
| `odb_full_v19_quick_match` | 86.37 | `paper.md:492` | ✅ |
| `odb_full_v17_quick_match` | 86.35 | `paper.md:448` | ✅ |

So 87.29 is a genuine stored measurement of a real build, not a hard-coded literal with no backing. Note this is a *read* of the recorded components; `Overall` is a composite the harness does not itself store, and `overall.py` is the same script that produced every other Overall quoted in the paper.

---

## ⚠ Caveat on bar 9 — the finding that could change paper text

Bar 9 is labelled *"+ column assembly, table repair, +0.67"*. Its component breakdown vs v20:

| | text | CDM | **TEDS** | order | Overall |
|---|---:|---:|---:|---:|---:|
| v20 | 0.0844 | 88.12 | **80.19** | 0.1644 | 86.62 |
| v21 (`odb_ablA_full`) | 0.0844 | 88.14 | **82.15** | 0.1474 | 87.29 |
| delta | 0.0000 | +0.02 | **+1.96** | −0.0170 | **+0.67** |

Text is unchanged to four decimals and CDM moves +0.02. **The entire +0.67 is TEDS.**

`pipeline/page_core.py:226` `_append_geom_tables` (default `PRISM_TABLE_GEOM=1`, verified present in the current source at line 230) appends a second, geometry-reconstructed HTML grid after *every* SLANet table. Its sibling docstring at `page_core.py:155` states the intent: the olmOCR TableTest passes if *any* emitted table has the right cell, so an extra grid is additive. On OmniDocBench the same works because unmatched predictions are free under `quick_match` — two attempts per GT table, no penalty.

A recorded A/B on 458 GT-table pages measured the size of that effect:

| | TEDS | struct | tables emitted |
|---|---:|---:|---:|
| `PRISM_TABLE_GEOM=1` (v21 shipped) | 82.05 | 89.90 | 1347 (for 665 GT tables) |
| `PRISM_TABLE_GEOM=0` | 80.03 | 87.66 | 723 |
| **delta** | **+2.02** | +2.24 | — |

TEDS is one third of the composite, so +2.02 TEDS ≈ **+0.67 Overall** — which is bar 9, to the decimal. The +1.96 TEDS I measure between v20 and v21 above is the same effect.

**Implications, stated as measured, not as a recommendation:**

- Bar 9 is a real measurement of a real build. What it measures is largely a benchmark-matcher effect, not a parsing improvement a user would see — v21 emits every table twice.
- The figure ends at **87.29** while `tab:ablation` (`4_experiments.tex:272`) and the headline both end at **86.62**. They disagree because they are *different builds* (v21 vs v20), not because a number was fabricated. That still needs reconciling before submission.
- If bar 9 is dropped or re-measured with `PRISM_TABLE_GEOM=0`, the figure endpoint lands near 86.6 and matches the table and the abstract.

I have made no change to any `.tex`, and per your instruction wrote no new ablation driver and ran no benchmark. `ablation_waterfall.py` is excluded from the supplementary tree, as are the files carrying its constants.
