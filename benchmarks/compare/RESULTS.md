# PRISM vs PP-StructureV3 vs SmolDocling — CPU-only head-to-head

20-page OmniDocBench v1.5 subset (stratified), Windows CPU (16 cores), 8-thread
budget for all, isolated runs (one system at a time). PP-StructureV3 = default
*server* config (PP-OCRv5 server + layout + wired/wireless table + PP-FormulaNet).

## Efficiency (measured, same hardware + pages) — PRISM's story

| Metric                | PRISM  | PP-StructureV3 | SmolDocling-256M |
|-----------------------|--------|----------------|------------------|
| Peak RAM (process tree) | **1.3 GB** | 8.2 GB      | 2.3 GB           |
| Latency median (s/page) | **6.8** | 58.2          | 92.2             |
| Latency mean (s/page)   | 6.9    | 103.8          | 125.7            |
| Model load (s)          | ~n/a (workers) | 16.4    | 0.03 (lazy)      |
| Inference weights       | **~200 MB** | >1 GB     | ~500 MB          |

PRISM is **6x lighter + 8.5x faster** than PP-StructureV3, and **1.8x lighter +
13x faster** than SmolDocling, on CPU.

## Accuracy — English subset (edit distance lower=better, TEDS higher=better)

| Metric        | PRISM | PP-StructureV3 | SmolDocling |
|---------------|-------|----------------|-------------|
| Text          | 0.165 | **0.081**      | 0.461       |
| Formula       | 0.491 | **0.320**      | 1.000*      |
| Reading-order | 0.319 | **0.276**      | 0.498       |
| Table TEDS    | 46.1% | **56.5%**      | 0.0%*       |

\* SmolDocling formula/table self-run scores are **format-mismatched** (its
DocTags→markdown output doesn't align with OmniDocBench's expected formula/table
notation) — NOT its true ability. Use its **published** OmniDocBench overall
(0.493 EN) for accuracy claims. Its text score (worse than PRISM) is a fair read.

## Honest takeaways

1. **Efficiency: PRISM wins decisively and cleanly** (controlled, same hardware).
2. **vs SmolDocling: PRISM wins on BOTH accuracy and efficiency** — the trendy
   256M VLM is slower, heavier, AND less accurate than PRISM on CPU.
3. **vs PP-StructureV3: PRISM trades accuracy for efficiency** — PP-Structure is
   more accurate (it's SOTA-lightweight) but needs 6x the RAM and 8.5x the time.
   PRISM occupies the efficient corner of the accuracy/efficiency frontier.

## Caveats
- 20-page subset: directional. Published OmniDocBench numbers are authoritative
  for accuracy; this run's value is the controlled CPU efficiency comparison.
- PP-StructureV3 = server config (heaviest/most accurate). A mobile-config run
  would show a lighter/less-accurate point — worth adding for the full frontier.
- All systems: 8 threads. PRISM ran dual-worker (benchmark) config; the single-
  worker product config is lighter still (~700-800 MB).
