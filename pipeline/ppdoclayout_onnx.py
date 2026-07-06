"""
ppdoclayout_onnx.py
-------------------
Raw onnxruntime wrapper for PP-DocLayout_plus-L (RT-DETR) — no paddle, no torch.

PP-DocLayout_plus-L replaces the two former layout detectors (YOLOv11n-DocLayNet
+ DocLayout-YOLO) with a single RT-DETR model. Validated to beat the two-detector
combo on every OmniDocBench category (formula/text/reading-order/table) while
netting only +42 MB.

RT-DETR is NMS-free (300 object queries). Preprocessing (from inference.yml):
  * Resize to 800x800, keep_ratio=False, bilinear
  * NormalizeImage norm_type=none, mean=0, std=1  (scale to [0,1], no mean/std)
Inputs : im_shape[B,2], image[B,3,800,800], scale_factor[B,2]
Outputs: [300*B, 6] = (class_id, score, x1, y1, x2, y2) in ORIGINAL pixels.

The onnxruntime output is validated box-for-box against the paddle model
(see scripts/validate_ppdoclayout_onnx.py).
"""

import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from pipeline.onnx_config import apply_session_threads, ort_providers

# PP-DocLayout_plus-L label list (order = class_id), from inference.yml
PPDL_LABELS = [
    'paragraph_title', 'image', 'text', 'number', 'abstract', 'content',
    'figure_title', 'formula', 'table', 'reference', 'doc_title', 'footnote',
    'header', 'algorithm', 'footer', 'seal', 'chart', 'formula_number',
    'aside_text', 'reference_content',
]

# PP-DocLayout label -> PRISM class vocab (dropped keys are ignored)
PPDL2PRISM = {
    'text': 'Text', 'abstract': 'Text', 'content': 'Text', 'reference': 'Text',
    'reference_content': 'Text', 'aside_text': 'Text', 'algorithm': 'Text',
    'footnote': 'Footnote',
    'paragraph_title': 'Section-header', 'doc_title': 'Title', 'figure_title': 'Caption',
    'header': 'Page-header', 'footer': 'Page-footer', 'number': 'Page-header',
    'formula': 'Formula',
    'table': 'Table',
    'image': 'Picture', 'chart': 'Picture',
    # dropped: formula_number (eq numbers, GT ignores), seal (stamps)
}

# PP-DocLayoutV3 label list (order = class_id), from the exported config.json.
# V3 output rows are [class_id, score, x1, y1, x2, y2, read_order].
PPDL_V3_LABELS = [
    'abstract', 'algorithm', 'aside_text', 'chart', 'content',
    'display_formula', 'doc_title', 'figure_title', 'footer', 'footer_image',
    'footnote', 'formula_number', 'header', 'header_image', 'image',
    'inline_formula', 'number', 'paragraph_title', 'reference',
    'reference_content', 'seal', 'table', 'text', 'vertical_text',
    'vision_footnote',
]

PPDL_V3_2PRISM = {
    'text': 'Text', 'abstract': 'Text', 'content': 'Text', 'reference': 'Text',
    'reference_content': 'Text', 'aside_text': 'Text', 'algorithm': 'Text',
    'vertical_text': 'Text', 'vision_footnote': 'Text',
    'footnote': 'Footnote',
    'paragraph_title': 'Section-header', 'doc_title': 'Title', 'figure_title': 'Caption',
    'header': 'Page-header', 'footer': 'Page-footer', 'number': 'Page-header',
    'display_formula': 'Formula',
    'table': 'Table',
    'image': 'Picture', 'chart': 'Picture',
    # dropped: inline_formula (sub-line spans; handled separately if enabled),
    # formula_number, seal, header_image, footer_image (decorative marginalia)
}

_IMGSZ = 800


class PPDocLayoutOnnxDetector:
    """Sole layout detector: raw-onnxruntime RT-DETR, mapped to PRISM classes."""

    def __init__(self, model_path: str, imgsz: int = _IMGSZ, map_to_prism: bool = True,
                 keep_inline_formula: bool = False):
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        apply_session_threads(so)
        self.sess = ort.InferenceSession(model_path, so,
                                         providers=ort_providers())
        self.imgsz = imgsz
        self.map_to_prism = map_to_prism
        self.keep_inline_formula = keep_inline_formula
        self._in = {i.name for i in self.sess.get_inputs()}
        # V3 emits [cls, score, x1, y1, x2, y2, read_order] (7 cols) plus a
        # mask tensor we ignore; plus-L emits 6 cols.
        out0 = self.sess.get_outputs()[0].shape
        self.is_v3 = (len(out0) == 2 and out0[1] == 7)
        self._labels = PPDL_V3_LABELS if self.is_v3 else PPDL_LABELS
        self._map = PPDL_V3_2PRISM if self.is_v3 else PPDL2PRISM

    def _preprocess(self, img_rgb: np.ndarray):
        import cv2
        h, w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0           # norm_type=none => scale only
        x = x.transpose(2, 0, 1)[None]                   # [1,3,800,800]
        im_shape = np.array([[self.imgsz, self.imgsz]], dtype=np.float32)
        # scale_factor = resized / original  (RT-DETR maps boxes back to original)
        scale_factor = np.array([[self.imgsz / h, self.imgsz / w]], dtype=np.float32)
        return np.ascontiguousarray(x), im_shape, scale_factor

    def detect(self, image, conf: float = 0.5, **_) -> list:
        """Return [{bbox:[x1,y1,x2,y2], class_name, confidence}] in original pixels."""
        if isinstance(image, str):
            pil = ImageOps.exif_transpose(Image.open(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = ImageOps.exif_transpose(image).convert("RGB")
        else:
            pil = Image.fromarray(image).convert("RGB")
        img_rgb = np.array(pil)
        H, W = img_rgb.shape[:2]

        x, im_shape, scale_factor = self._preprocess(img_rgb)
        feed = {'image': x}
        if 'im_shape' in self._in:
            feed['im_shape'] = im_shape
        if 'scale_factor' in self._in:
            feed['scale_factor'] = scale_factor
        out = self.sess.run(None, feed)[0]               # [N,6|7] = cls,score,box(,ro)

        results = []
        for row in out:
            cls_id, score, x1, y1, x2, y2 = row[:6]
            read_order = float(row[6]) if self.is_v3 else None
            if score < conf:
                continue
            label = self._labels[int(cls_id)] if 0 <= int(cls_id) < len(self._labels) else None
            if self.map_to_prism:
                name = self._map.get(label)
                if name is None and label == 'inline_formula':
                    # V3 labels handwritten/standalone formulas 'inline' where
                    # plus-L called them display (notes pages lost every formula
                    # to the dropped class). Route them as Formula and let
                    # formula_v2's inline-FP guard drop the true in-text chips.
                    if os.environ.get('PRISM_INLINE_FML_DISPLAY', '1') != '0':
                        name = 'Formula'
                    elif self.keep_inline_formula:
                        name = 'InlineFormula'
                if name is None:
                    continue
            else:
                name = label
            x1 = float(np.clip(x1, 0, W)); x2 = float(np.clip(x2, 0, W))
            y1 = float(np.clip(y1, 0, H)); y2 = float(np.clip(y2, 0, H))
            if x2 <= x1 or y2 <= y1:
                continue
            det = {'bbox': [x1, y1, x2, y2], 'class_name': name,
                   'confidence': float(score)}
            if read_order is not None:
                det['read_order'] = read_order
            results.append(det)
        return results
