"""
Export TATR (Table Transformer) to ONNX + INT8 quantized ONNX.

Usage:
    python scripts/export_tatr_onnx.py

Outputs:
    models/tatr_structure.onnx          (~115 MB FP32)
    models/tatr_structure_int8.onnx     (~30 MB INT8)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"
OUT_DIR  = Path(__file__).parent.parent / "models"
OUT_DIR.mkdir(exist_ok=True)
FP32_PATH = OUT_DIR / "tatr_structure.onnx"
INT8_PATH = OUT_DIR / "tatr_structure_int8.onnx"

# ── 1. Load PyTorch model ─────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")
from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")

# ── 2. Export to ONNX ─────────────────────────────────────────────────────────
# TATR input: [1, 3, H, W]  (ImageNet-normalised, longest-edge ≤ 800)
# TATR output: logits [1, 100, 7], pred_boxes [1, 100, 4]
dummy = torch.zeros(1, 3, 600, 800)

print(f"Exporting to {FP32_PATH} ...")
torch.onnx.export(
    model,
    {"pixel_values": dummy},
    str(FP32_PATH),
    opset_version=17,
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
    dynamic_axes={
        "pixel_values": {2: "height", 3: "width"},
        "logits":       {0: "batch"},
        "pred_boxes":   {0: "batch"},
    },
    do_constant_folding=True,
)
size_mb = FP32_PATH.stat().st_size / 1e6
print(f"FP32 ONNX saved: {FP32_PATH} ({size_mb:.0f} MB)")

# ── 3. INT8 dynamic quantization ──────────────────────────────────────────────
print(f"Quantizing to INT8 -> {INT8_PATH} ...")
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    str(FP32_PATH),
    str(INT8_PATH),
    weight_type=QuantType.QUInt8,
)
size_int8 = INT8_PATH.stat().st_size / 1e6
print(f"INT8 ONNX saved: {INT8_PATH} ({size_int8:.0f} MB)")
print(f"Size reduction: {size_mb:.0f} MB → {size_int8:.0f} MB ({size_mb/size_int8:.1f}x)")

# ── 4. Quick smoke test ───────────────────────────────────────────────────────
print("\nSmoke-testing INT8 model ...")
import onnxruntime as ort
sess = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])
dummy_np = np.zeros((1, 3, 600, 800), dtype=np.float32)
logits, boxes = sess.run(None, {"pixel_values": dummy_np})
print(f"Output shapes: logits={logits.shape}, pred_boxes={boxes.shape}")
print("Export complete.")
