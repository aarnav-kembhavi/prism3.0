# PRISM vs PDF2LaTeX Benchmark Report

## Executive Summary
This report evaluates the PRISM Screen-to-LaTeX system against the PDF2LaTeX (Wang & Liu, 2020) dataset and baseline.

## Dataset Statistics
- **Number of pages:** 102
- **Total Ground Truth Characters:** 232557

## Comparison Table

| Metric | PDF2LaTeX (Paper) | PRISM (Ours) | Improvement |
| ------ | ----------------- | ------------ | ----------- |
| Overall EDR | 81.1% | 62.9% | -22.5% |
| BLEU-4 | 92.1* | 43.5 | - |
| Avg. Latency | - | 10.08s | - |
| Peak RAM | - | 1588.7 MB | - |

*Note: PDF2LaTeX reported 92.1 BLEU for formula recognition specifically.*

## Performance Breakdown
- **Average Total Latency:** 10.08s
- **Average Peak RAM:** 1480.0 MB
- **Peak RAM (max page):** 1588.7 MB
- **Average OCR Latency:** 3.87s
- **Average Layout Latency:** 1.50s
- **Average Math Latency:** 1.07s
- **Average Table Latency:** 0.00s

## Failure Analysis
TBD

