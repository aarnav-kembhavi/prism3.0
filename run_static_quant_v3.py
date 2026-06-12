"""
run_static_quant_v3.py
----------------------
Third attempt at static INT8 YOLO quantization.
Strategy:
  - QOperator format (native INT8 ops, avoids QDQ boundary artifacts)
  - Backbone-only quantization: only model.0 through model.9 (119 nodes)
  - Exclude model.10 onwards (neck + head, 201 nodes) to avoid feature corruption
  - CalibrationMethod.Entropy (best for preserving detection feature distributions)
  - per_channel=False, reduce_range=False
"""
import gc, os
import numpy as np
import cv2
import psutil
import onnxruntime as ort
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, CalibrationMethod,
    QuantFormat, QuantType,
)
PREP_MODEL  = "yolov11n-doclaynet-prep.onnx"
SRC_MODEL   = "yolov11n-doclaynet.onnx"
OUT_MODEL   = "yolov11n-doclaynet-static-v3.onnx"

CALIB_DIR   = Path("benchmark_results/quant_comparison/float32_yolo")
IMG_SIZE    = 640

# ── Collect calibration images ────────────────────────────────────────────────
calib_images = sorted(CALIB_DIR.glob("*/assets/normalized.png"))[:26]
print(f"[*] Found {len(calib_images)} calibration images")


def _preprocess(path: str) -> np.ndarray:
    """Letterbox + normalize exactly as YOLO expects."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    dy, dx = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw] = r
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0


class YOLOCalibReader(CalibrationDataReader):
    def __init__(self, images):
        self.images = [_preprocess(str(p)) for p in images]
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.images):
            return None
        inp = {"images": self.images[self._idx]}
        self._idx += 1
        return inp


# ── Build exclude list (model.10 onwards) ────────────────────────────────────
import onnx
m = onnx.load(PREP_MODEL)
exclude_nodes = [
    n.name for n in m.graph.node
    if any(n.name.startswith(f"/model.{i}") for i in range(10, 30))
]
print(f"[*] Excluding {len(exclude_nodes)} nodes (model.10+, neck+head)")
print(f"    Quantizing {len(m.graph.node) - len(exclude_nodes)} nodes (model.0-9, backbone only)")

# ── Quantize ─────────────────────────────────────────────────────────────────
print(f"[*] Running static INT8 quantization (QOperator, Entropy, backbone-only)...")
calib_reader = YOLOCalibReader(calib_images)

quantize_static(
    PREP_MODEL,
    OUT_MODEL,
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QOperator,     # native INT8 ops, no QDQ boundary artifacts
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,                      # simpler, more stable
    reduce_range=False,
    calibrate_method=CalibrationMethod.Entropy,
    nodes_to_exclude=exclude_nodes,
)
print(f"[*] Saved: {OUT_MODEL}")
print(f"    Size: {Path(OUT_MODEL).stat().st_size/1024**2:.1f} MB")
print(f"    (vs float32: {Path(SRC_MODEL).stat().st_size/1024**2:.1f} MB)")

# ── Quick validation ──────────────────────────────────────────────────────────
print(f"\n[*] Validating {OUT_MODEL}...")

process = psutil.Process(os.getpid())
baseline_mb = process.memory_info().rss / 1024**2
print(f"    Baseline RAM: {baseline_mb:.0f} MB")

sess = ort.InferenceSession(OUT_MODEL)
after_load_mb = process.memory_info().rss / 1024**2
print(f"    After load: {after_load_mb:.0f} MB  (+{after_load_mb - baseline_mb:.0f} MB)")

# Compare against float32 baseline
sess_f32 = ort.InferenceSession(SRC_MODEL)
after_f32_mb = process.memory_info().rss / 1024**2
print(f"    After float32 load: {after_f32_mb:.0f} MB")

test_inp = calib_images[0]
inp_arr = _preprocess(str(test_inp))

out_v3  = sess.run(None, {"images": inp_arr})[0]
out_f32 = sess_f32.run(None, {"images": inp_arr})[0]

scores_v3  = out_v3[0, 4:, :].max(axis=0)
scores_f32 = out_f32[0, 4:, :].max(axis=0)

print(f"\n    float32  - max conf: {scores_f32.max():.4f}  anchors >0.1: {(scores_f32 > 0.1).sum()}")
print(f"    static-v3 - max conf: {scores_v3.max():.4f}  anchors >0.1: {(scores_v3 > 0.1).sum()}")
print(f"    static-v3 - output range: [{out_v3.min():.4f}, {out_v3.max():.4f}]")

if scores_v3.max() > 0.1:
    print("\n  [PASS] Static INT8 v3 produces valid detections!")
else:
    print("\n  [FAIL] Static INT8 v3 still produces no detections.")
    print("         Max confidence:", scores_v3.max())
