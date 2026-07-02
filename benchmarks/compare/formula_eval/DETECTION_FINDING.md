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
