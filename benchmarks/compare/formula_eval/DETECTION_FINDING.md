# The real formula bottleneck: DETECTION recall, not recognition

20 English math pages, 224 GT display formulas.

## Recall of formula detection (IoU>0.5)
| Detector                          | Recall | Notes |
|-----------------------------------|--------|-------|
| Current (DocLayNet + DocLayout)   | 29%    | lowering conf to 0.05 doesn't help (blind) |
| MFD yolo_v8_ft @ 640              | 34%    | its default res too low for dense pages |
| **MFD yolo_v8_ft_dyn @ 1280**     | **78%**| the fix — high-res dedicated detector |
| MFD @ 1600                        | 78%    | no gain, 1.6x slower |

When a formula IS detected, boxes are tight (IoU 0.86-0.93) and merging is rare
(5.5%). So the ENTIRE formula weakness is recall: PRISM was missing ~70% of
formulas, and the edit-distance metric hid it (only scores detected formulas).

## Root cause
1. PRISM never used its own dedicated MFD model (models/MFD/YOLO/, 167MB onnx) —
   it relied on general layout detectors (DocLayNet 'Formula', DocLayout
   'isolate_formula') that are weak on formulas.
2. Detection ran at 640px; dense-math pages need 1280px or small formulas vanish.

## Fix
Wire in MFD (yolo_v8_ft_640_dyn.onnx, class 'isolated') at imgsz=1280 for formula
regions. +167MB (total ~367MB, still << competitors), +~2s/page. Recognition
(Texo) stays — it was never the problem.

## Update: MFD isn't a clean win (measured end-to-end)

The 78% "recall" was an artifact: it counts BOTH MFD classes (embedding=inline,
isolated=display), 534 boxes/20pg (~27/page — mostly genuine inline math).
- MFD 'isolated' only (clean): 31% recall — barely above the current 29%.
- MFD both classes: 78% but floods inline math as display blocks (unusable).
- A width/size filter on 'embedding' does NOT separate mislabeled-display from
  true-inline (both same size): iso+emb@15%pw = 34% recall only.

End-to-end A/B (MFD 'isolated' on vs off, 20 formula pages, 224 GT formulas):
  Formula edit-dist  0.433 -> 0.399  (-0.035, modest)
  Reading-order      0.311 -> 0.294  (-0.017)
  Text               0.150 -> 0.150  (flat, no regression)
  Formula blocks out  79   ->  93    (+14)

CONCLUSION: formula weakness = detection recall (confirmed), but it is genuinely
HARD to close cheaply. The unused MFD model only cleanly adds ~+14 formulas
(-0.035 edit) at +167MB / +2s/page — marginal cost/benefit for an efficiency-
first system. Set to OPT-IN (PRISM_USE_MFD=1). Real fixes require a better
display-formula detector or fine-tuning MFD's display class — actual ML work,
not a config change. Honest paper framing: PRISM trades some formula recall for
efficiency; dense-math display-formula detection is the frontier.
