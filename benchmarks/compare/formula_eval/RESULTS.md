# Texo vs PP-FormulaNet-S — formula recognition head-to-head

120 English isolated-formula crops from OmniDocBench (GT-cropped, i.e. perfect
boxes), scored by normalized edit distance vs GT LaTeX (lower=better).

| Model                    | Size   | Speed/crop | Mean edit-dist | Exact |
|--------------------------|--------|------------|----------------|-------|
| **Texo** (current)       | 79 MB  | 103 ms     | **0.262**      | 25/120|
| PP-FormulaNet-S          | 224 MB | 523 ms     | 0.268          | 27/120|

## Key finding: the recognizer is NOT the bottleneck
Both strong formula models land ~0.26 on CLEAN crops — statistically tied.
PP-FormulaNet-S (2.8x the size, 5x slower) gives ZERO accuracy benefit over
Texo. Do NOT swap the model.

But PRISM's END-TO-END formula edit distance is ~0.43 (981) / ~0.49 (20-pg).
Clean-crop recognition is 0.26. So ~40% of the formula error comes from the
DETECTION/CROP pipeline, not the recognizer:
  - loose / merged formula bounding boxes (YOLO+DocLayout)
  - multi-equation regions sent as one 384x384 crop (Texo can't split)
  - crop preprocessing (Otsu binarize / pad / resize)

## Recommendation (data-driven, redirected)
1. KEEP Texo — it matches Baidu's efficient formula model at 1/3 the size and
   5x the speed. It is already a strong, efficient choice (a paper point).
2. Fix formula DETECTION/CROP — that is where ~0.17 of the ~0.43 end-to-end
   error lives, and it costs 0 extra MB. Tighter/split formula boxes is the win.
