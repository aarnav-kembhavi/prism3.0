# PRISM vs PDF2LaTeX Benchmark Report

## Executive Summary
This report evaluates the PRISM Screen-to-LaTeX system against the PDF2LaTeX (Wang & Liu, 2020) dataset and baseline.

## Dataset Statistics
- **Number of pages:** 101
- **Total Ground Truth Characters:** 230099

## Comparison Table

| Metric | PDF2LaTeX (Paper) | PRISM (Ours) | Improvement |
| ------ | ----------------- | ------------ | ----------- |
| Overall EDR | 81.1% | 63.7% | -21.5% |
| BLEU-4 | 92.1* | 39.2 | - |
| Avg. Latency | - | 8.60s | - |
| Peak RAM | - | 1446.0 MB | - |

*Note: PDF2LaTeX reported 92.1 BLEU for formula recognition specifically.*

## Performance Breakdown
- **Average OCR Latency:** 0.00s
- **Average Layout Latency:** 0.41s

## Failure Analysis
TBD

