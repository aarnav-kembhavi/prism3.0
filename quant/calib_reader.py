"""
CalibrationDataReader for PP-DocLayoutV3 static QDQ quantization, plus the
head-node selector used by --exclude-heads.

Calibration images are held out: the reader draws from quant/calib_pages.json,
which is built to exclude every page in the frozen 30-page eval set.

Preprocessing mirrors pipeline/ppdoclayout_onnx.py._preprocess exactly
(resize to imgsz square, /255, CHW, plus im_shape and scale_factor inputs) --
calibrating on a different distribution than inference sees is the usual way
static quantization silently goes wrong.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
_IMGSZ = 800


def _preprocess(path: Path, imgsz: int = _IMGSZ):
    pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = np.asarray(pil)
    h, w = img.shape[:2]
    x = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    im_shape = np.array([[imgsz, imgsz]], dtype=np.float32)
    scale_factor = np.array([[imgsz / h, imgsz / w]], dtype=np.float32)
    return np.ascontiguousarray(x), im_shape, scale_factor


class LayoutCalibrationReader(CalibrationDataReader):
    def __init__(self, model_path: str, calib_dir: str, n: int, imgsz: int = _IMGSZ):
        import onnxruntime as ort
        self.inputs = {i.name for i in ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]).get_inputs()}
        spec = json.loads((ROOT / "quant" / "calib_pages.json").read_text(encoding="utf-8"))
        img_dir = ROOT / spec["image_dir"]
        self.paths = [img_dir / p for p in spec["pages"][:n]]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} calibration images missing, e.g. {missing[0]}")
        self.imgsz = imgsz
        self._it = None
        print(f"  calibration: {len(self.paths)} held-out pages from {spec['image_dir']}")

    def _gen(self):
        for i, p in enumerate(self.paths):
            if i and i % 20 == 0:
                print(f"    calibrated on {i}/{len(self.paths)}")
            x, im_shape, scale_factor = _preprocess(p, self.imgsz)
            feed = {"image": x}
            if "im_shape" in self.inputs:
                feed["im_shape"] = im_shape
            if "scale_factor" in self.inputs:
                feed["scale_factor"] = scale_factor
            yield feed

    def get_next(self):
        if self._it is None:
            self._it = self._gen()
        return next(self._it, None)

    def rewind(self):
        self._it = None


QUANTIZABLE = {"Conv", "MatMul", "Gemm"}


def head_nodes(model_path: str, max_depth: int = 33) -> list[str]:
    """
    Names of the box-regression and classification head layers.

    Found structurally rather than by name: BFS backwards from the graph
    outputs, keeping only ops the quantizer would actually touch
    (Conv/MatMul/Gemm). Reshape/Cast/Concat plumbing sits between the outputs
    and the real layers -- excluding those would protect nothing, since they
    are never quantized in the first place.

    On PP-DocLayoutV3 the first quantizable op is 16 hops back and the head
    cluster runs to ~33; past that the walk fans into the shared decoder and
    the HGNetv2 backbone, which we DO want quantized (that is where the 130 MB
    lives). max_depth is the cutoff between the two.
    """
    model = onnx.load(model_path, load_external_data=False)
    g = model.graph
    producer = {out: n for n in g.node for out in n.output}

    frontier = {o.name for o in g.output}
    seen, out = set(), []
    for _ in range(max_depth):
        nxt = set()
        for t in frontier:
            n = producer.get(t)
            if n is None or id(n) in seen:
                continue
            seen.add(id(n))
            if n.op_type in QUANTIZABLE and n.name:
                out.append(n.name)
            nxt.update(n.input)
        frontier = nxt
        if not frontier:
            break
    return out
