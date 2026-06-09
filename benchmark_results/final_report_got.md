# PRISM + GOT-OCR2 Benchmark Report

## Dataset Statistics
- **Number of pages:** 102

## Accuracy vs Texo baseline

| Metric | Texo (prev) | GOT-OCR2 | Change |
|--------|------------|----------|--------|
| Overall EDR | 62.9% | 73.5% | +10.6pp |
| Math EDR | 22.2% | 29.5% | +7.3pp |
| Text EDR | 69.8% | 53.7% | -16.1pp |
| BLEU-4 | 43.5 | 60.3 | +16.8 |
| ROUGE-L | 72.5% | 81.5% | +9.0pp |
| CER | 37.1% | 26.5% | -10.6pp |

## Performance

| Metric | Value |
|--------|-------|
| Avg total latency | 26.05s |
| Avg math latency | 19.14s |
| Avg OCR latency | 3.02s |
| Avg peak RAM | 1684 MB |
| Peak RAM (max) | 1787 MB |
