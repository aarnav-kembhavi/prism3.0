# Per-section scores — odb_full_v20 (MinerU crossed)

text / reading-order = edit distance (lower better); CDM / TEDS ×100 (higher better).
**Overall 86.62** (text 0.0844, CDM 88.12, TEDS 80.19, RO 0.1644) — beats
MinerU-Pipeline 86.47. Config = v19 + inline-math splicing (PRISM_INLINE_SPLICE).

## By document type

| group | TextEdit↓ | CDM↑ | TEDS↑ | OrderEdit↓ |
|---|---|---|---|---|
| PPT2PDF | 0.0657 | 88.57 | 83.98 | — |
| academic_literature | 0.0804 | 89.11 | 71.98 | — |
| book | 0.0833 | 83.74 | 87.98 | — |
| colorful_textbook | 0.0970 | 81.99 | 88.98 | — |
| exam_paper | 0.1270 | 91.15 | 89.42 | — |
| historical_document | 0.5602 | — | — | — |
| magazine | 0.0559 | — | 71.54 | — |
| newspaper | 0.1028 | — | 67.87 | — |
| note | 0.1013 | 73.50 | 57.74 | — |
| research_report | 0.0142 | — | 82.30 | — |

## By language

| group | TextEdit↓ | CDM↑ | TEDS↑ |
|---|---|---|---|
| en_ch_mixed | 0.0801 | 93.30 | 81.46 |
| english | 0.0699 | 90.93 | 75.07 |
| simplified_chinese | 0.0976 | 76.98 | 83.00 |
| traditional_chinese | 0.2122 | — | 71.29 |

## By layout

| group | TextEdit↓ | CDM↑ | TEDS↑ |
|---|---|---|---|
| 1andmore_column | 0.0635 | 93.82 | 87.26 |
| double_column | — | 90.09 | 84.96 |
| other_layout | — | 74.26 | 78.40 |
| single_column | — | 86.68 | 78.80 |
| three_column | — | 84.92 | 57.11 |

## Deltas vs v19 (86.37)

- Overall +0.25 (86.37 → 86.62), above the ±0.1 run-to-run noise floor.
- CDM +0.56 (87.56 → 88.12) — primary driver; spliced inline `$latex$`
  spawns free equation_inline items rescuing GT display formulas.
- Text edit −0.0021 (0.0865 → 0.0844) — inline math no longer read as unicode.
- TEDS −0.02, RO +0.0008 — both within noise (splice does not touch tables/order).
- Biggest section gains: note text 0.1061→0.1013 & CDM 72.3→73.5, book text
  0.0918→0.0833, exam CDM stays high, EN text 0.0723→0.0699.
