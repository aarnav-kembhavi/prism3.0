"""
yolo_onnx.py
------------
Raw onnxruntime YOLO detectors — no ultralytics, no torch.

The pipeline previously loaded both layout models through `ultralytics.YOLO`,
which imports torch into the *main* process (~400 MB baseline). That defeats
the whole point of the ONNX-subprocess-worker architecture, whose stated goal
is to keep torch out of the main process. This module runs both models with
onnxruntime directly and reproduces ultralytics' pre/post-processing:

  * YOLOv11n-DocLayNet — raw head `[1, 4+nc, 8400]`, needs conf-filter + NMS.
  * DocLayout (YOLOv10m) — end-to-end head `[1, 300, 6]` = xyxy+conf+cls,
    already NMS-free; just filter by confidence.

Both use the canonical letterbox (aspect-preserving resize + 114-pad to a
stride multiple) so predicted boxes map back to original-image pixels exactly
as ultralytics does. Class names are read from the ONNX metadata.

Validated to reproduce ultralytics detections box-for-box (see
scripts/validate_yolo_onnx.py).
"""

import ast
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from pipeline.onnx_config import apply_session_threads


def letterbox(img_rgb: np.ndarray, new_shape, stride: int = 32,
              color=(114, 114, 114), scaleup: bool = True):
    """Resize preserving aspect ratio and pad to a stride multiple.

    Returns (padded_img, ratio, (pad_left, pad_top)) so boxes can be mapped
    back: x_orig = (x_padded - pad_left) / ratio.
    """
    import cv2
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = img_rgb.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        img_rgb = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_rgb = cv2.copyMakeBorder(img_rgb, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return img_rgb, r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list:
    """Greedy NMS on xyxy boxes; returns kept indices (score-desc)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


class YoloOnnxDetector:
    """Drop-in raw-onnxruntime replacement for the two ultralytics detectors."""

    def __init__(self, model_path: str, imgsz: int = 640):
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        apply_session_threads(so)
        self.sess = ort.InferenceSession(model_path, so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta['names']) if 'names' in meta else {}

    def _preprocess(self, img_rgb: np.ndarray):
        padded, r, (dx, dy) = letterbox(img_rgb, self.imgsz)
        x = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x), r, (dx, dy)

    def detect(self, image, conf: float = 0.25, iou: float = 0.7,
               max_det: int = 300) -> list:
        """Return detections as list of {bbox:[x1,y1,x2,y2], class_name, confidence}."""
        if isinstance(image, str):
            pil = Image.open(image)
            pil = ImageOps.exif_transpose(pil).convert("RGB")  # honor phone-photo orientation
        elif isinstance(image, Image.Image):
            pil = ImageOps.exif_transpose(image).convert("RGB")
        else:  # numpy RGB
            pil = Image.fromarray(image).convert("RGB")
        img_rgb = np.array(pil)
        H, W = img_rgb.shape[:2]

        x, r, (dx, dy) = self._preprocess(img_rgb)
        out = self.sess.run(None, {self.input_name: x})[0]

        # Decide head type from the concrete output shape (ONNX metadata shapes
        # can be symbolic): end-to-end YOLOv10 emits [1, N, 6] (xyxy+conf+cls);
        # a raw head emits [1, 4+nc, anchors] where the last dim is thousands.
        if out.ndim == 3 and out.shape[2] == 6:
            dets = self._post_end2end(out, conf)
        else:
            dets = self._post_raw(out, conf, iou, max_det)

        # Map boxes from letterboxed space back to original pixels.
        results = []
        for x1, y1, x2, y2, sc, cls in dets:
            ox1 = (x1 - dx) / r
            oy1 = (y1 - dy) / r
            ox2 = (x2 - dx) / r
            oy2 = (y2 - dy) / r
            ox1 = float(np.clip(ox1, 0, W)); ox2 = float(np.clip(ox2, 0, W))
            oy1 = float(np.clip(oy1, 0, H)); oy2 = float(np.clip(oy2, 0, H))
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            results.append({
                'bbox': [ox1, oy1, ox2, oy2],
                'class_name': self.names.get(int(cls), str(int(cls))),
                'confidence': float(sc),
            })
        return results

    def _post_raw(self, out, conf, iou, max_det):
        """YOLOv11 head [1, 4+nc, A] → conf filter + per-class NMS."""
        p = out[0].T                       # [A, 4+nc]
        boxes_cxcywh = p[:, :4]
        scores_all = p[:, 4:]
        cls = scores_all.argmax(1)
        sc = scores_all.max(1)
        m = sc >= conf
        if not m.any():
            return []
        boxes_cxcywh, sc, cls = boxes_cxcywh[m], sc[m], cls[m]
        # cxcywh → xyxy
        cx, cy, w, h = boxes_cxcywh.T
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        # per-class NMS (agnostic=False, matches ultralytics default)
        keep_all = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            k = _nms(xyxy[idx], sc[idx], iou)
            keep_all.extend(idx[k].tolist())
        keep_all = sorted(keep_all, key=lambda i: -sc[i])[:max_det]
        return [(xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], sc[i], cls[i])
                for i in keep_all]

    def _post_end2end(self, out, conf):
        """YOLOv10 head [1, N, 6] = xyxy+conf+cls, already NMS-free."""
        p = out[0]                          # [N, 6]
        m = p[:, 4] >= conf
        p = p[m]
        return [(row[0], row[1], row[2], row[3], row[4], row[5]) for row in p]
