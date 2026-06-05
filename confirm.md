# Benchmark Results Audit: Screen2LaTeX/PRISM

## 1. Executive Summary
A rigorous audit of `benchmark_results/benchmark_results.csv` and the supporting evaluation code (`evaluation/eval.py`, `run_comprehensive_benchmark.py`) has identified **critical systemic flaws** in the metric definitions and implementation. The current results are mathematically invalid for several key NLP metrics (WER, BLEU, ROUGE) and suggest a total failure or bypass of the math recognition component.

---

## 2. Issue Log & Audit Findings

| Issue Detected | Severity | Evidence | Recommended Fix |
| :--- | :--- | :--- | :--- |
| **WER is Hardcoded to 100** | **Critical** | `wer` is exactly 100.0 for every sample in the CSV. | Stop removing spaces (`Rule 4`) before tokenization, or use a character-level WER (not standard). |
| **Math EDR is Exactly 0** | **Critical** | `math_edr` is 0.0 for every sample. `math_latency` is a fixed 40% heuristic. | Investigate if `FormulaNet` is falling back to `center`+`includegraphics` every time. Fix `split_math_and_text` to handle `$$`. |
| **Invalid BLEU/ROUGE Scores** | **Major** | BLEU scores are in scientific notation (e.g., `4.39E-20`). | Metrics must be computed **before** `Rule 4` (space removal) to allow proper word-level tokenization. |
| **Negative Text EDR Values** | **Major** | `text_edr` reaches values as low as `-43.52`. | Implement a length-normalized penalty or "Bounded EDR" to avoid misleading negative percentages. |
| **Heuristic Latency Logging** | **Minor** | `math_latency` is always exactly `0.4 * ocr_latency`. | Add explicit `time.perf_counter()` around the `run_math_recognition_batched` call in `models_interface.py`. |

---

## 3. Detailed Technical Analysis

### A. The "WER = 100" Bug
The implementation of `compute_cer_wer` in `run_comprehensive_benchmark.py` calls `.split()` on the strings `pred_norm` and `gt_norm`. However, these strings have already passed through `normalize_latex`, where **Rule 4** (`re.sub(r'\s+', '', latex_str)`) removes all whitespace.
*   **Result:** `pred_norm.split()` returns a list with exactly one element (the entire string).
*   **Metric Failure:** Unless the prediction is a 100% character-perfect match, the edit distance between a 1-element list and another 1-element list will always be 1. The formula `(1 / 1) * 100` results in a constant 100% error rate.

### B. The "Math EDR = 0" Mystery
The metric `math_edr` being 0 suggests that `pred_math` is consistently empty.
*   **Heuristic Evidence:** The CSV reports `math_latency` as a derived heuristic, which strongly suggests the math stage is either being skipped or is failing and returning a non-math fallback.
*   **Regex Evidence:** The `split_math_and_text` regex `(\$[^$]*\$)` fails to capture display math if it is wrapped in `$$` (Rule 2 in `normalizer.py` does not normalize `$$` to `$`).

### C. Interpretation of Negative `text_edr`
The definition used is `1.0 - (ed / len(gt))`. 
*   **Valid?** Yes, mathematically possible.
*   **Interpretation:** A value of `-43.5` implies the edit distance is **44.5 times larger** than the actual ground truth length. This signifies "hallucination loops" where the OCR repeats characters or produces massive amounts of noise, which the current `levenshtein_distance` implementation treats as expensive insertions.

---

## 4. Reviewer Criticisms (Top 5)

If this CSV were submitted for peer review (e.g., at DocEng or ICDAR), the following criticisms would likely lead to an immediate rejection:

1.  **"Methodological Flaw in Tokenization":** The authors have computed word-level metrics (WER, BLEU, ROUGE) on strings where whitespace has been removed. This invalidates the core assumption of these metrics (word-boundary overlap), rendering the reported scores scientifically meaningless.
2.  **"Suspicious Uniformity in Error Rates":** A WER of exactly 100% across 102 heterogeneous document samples is a statistical impossibility in a functioning system. This points to an evaluation pipeline bug that undermines the credibility of the entire results table.
3.  **"Deceptive Latency Reporting":** The use of a fixed 40% heuristic to "calculate" math latency instead of empirical measurement is unacceptable for a systems paper. It hides the actual performance characteristics of the `Texo` model.
4.  **"Failed Math Recognition Analysis":** An EDR of 0 for math suggests the system is failing to produce any LaTeX-compliant math environments. The authors must explain why their math specialist model (FormulaNet) contributes nothing to the final accuracy.
5.  **"Lack of Error Analysis for Hallucinations":** The presence of extreme negative EDR values (indicative of massive over-generation) is left unaddressed. A rigorous evaluation requires a qualitative analysis of why the system produces 40x more characters than the ground truth in certain cases.
