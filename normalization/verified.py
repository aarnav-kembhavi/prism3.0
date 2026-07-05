# -*- coding: utf-8 -*-
"""
Recognition-verified normalization.

Every camera-branch correction (glare inpaint, shadow flatten, CLAHE,
rectification) is treated as a PROPOSAL. A proposal is accepted only if a
cheap recognition probe — the text-detection model scored on a downscaled
copy — does not get worse. Rationale: lighting defects and page design are
statistically indistinguishable at the pixel level (OmniDocBench PPT
gradients score higher on shadow metrics than real shadows), so any open-loop
heuristic either misses defects or damages clean pages. Closing the loop with
the recognizer decides each case by the only criterion that matters: does the
page read better afterwards?

The probe is PP-OCRv6 DBNet (10 MB ONNX, ~120 ms at 640 px). Probe score =
sum over detected text boxes of (box area x mean probability inside the box),
normalized by image area — i.e. "how much confident text surface does the
detector see". Corrections that erase strokes (over-aggressive shadow divide,
glare inpaint on paper) lower it; corrections that lift text out of shadow
raise it.

Product-path only: the benchmark runner pins PRISM_NORM_STRICT=1 which keeps
the validated open-loop routing byte-identical. Disable with PRISM_NORM_VERIFY=0.
"""
import os

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DET_MODEL = os.path.join(ROOT_DIR, "weights", "PP-OCRv6_det_small.onnx")
_PROBE_SIDE = 640          # probe resolution (longer side)
# Accept only on clear improvement: corrections are guilty until proven
# innocent. With a tolerance instead (accept when "not much worse") the
# verifier passed through corrections that cost accuracy on clean pages —
# measured on the 40-page synthetic study, mean block edit rose 0.177->0.185.
_ACCEPT_GAIN = 1.02        # accept if score_after >= score_before * this
_BIN_THRESH = 0.3

_session = None


def available() -> bool:
    return (os.environ.get("PRISM_NORM_VERIFY", "1") != "0"
            and os.path.exists(_DET_MODEL))


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = 4
        _session = ort.InferenceSession(_DET_MODEL, so,
                                        providers=["CPUExecutionProvider"])
    return _session


def probe_score(image_bgr) -> float:
    """Confident-text-surface score of a page image (higher = reads better)."""
    sess = _get_session()
    h, w = image_bgr.shape[:2]
    scale = _PROBE_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    else:
        img = image_bgr
    # DBNet input: multiples of 32
    H = max(32, (img.shape[0] // 32) * 32)
    W = max(32, (img.shape[1] // 32) * 32)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
        np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = x.transpose(2, 0, 1)[None]
    inp = sess.get_inputs()[0].name
    prob = sess.run(None, {inp: np.ascontiguousarray(x)})[0][0, 0]  # HxW map
    mask = (prob > _BIN_THRESH).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0
    # score: probability mass inside confident regions, per image area
    return float(prob[mask.astype(bool)].sum()) / (H * W)


def verified_apply(name, fn, image_bgr, score_before=None):
    """Apply fn(image) only if the probe does not get worse.

    Returns (image_after_or_before, score_of_returned_image, accepted: bool).
    """
    if score_before is None:
        score_before = probe_score(image_bgr)
    try:
        candidate = fn(image_bgr)
    except Exception as exc:
        print(f"  [norm-verify] {name}: correction failed ({exc}); keeping input")
        return image_bgr, score_before, False
    if candidate is image_bgr:
        return image_bgr, score_before, False
    score_after = probe_score(candidate)
    if score_after >= score_before * _ACCEPT_GAIN:
        print(f"  [norm-verify] {name}: accepted "
              f"({score_before:.4f} -> {score_after:.4f})")
        return candidate, score_after, True
    print(f"  [norm-verify] {name}: REJECTED "
          f"({score_before:.4f} -> {score_after:.4f})")
    return image_bgr, score_before, False
