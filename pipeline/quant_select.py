"""
Quantized-graph selection.

Every ONNX graph in the stack is registered here with its fp32 path. A
quantized sibling lives alongside it with an `_int8` suffix; the fp32 file is
never overwritten. Which variant each stage loads is chosen at run time by
environment variable, so any combination is runnable without rebuilding.

    PRISM_QUANT=texo_encoder,texo_decoder   # only those two load INT8
    PRISM_QUANT=all                         # every graph with an INT8 sibling
    PRISM_QUANT=                            # (unset/empty) pure fp32

A per-graph override wins over the list, and can also point at an explicit
file -- used for the PP-DocLayoutV3 heads-excluded vs fully-quantized variants:

    PRISM_QUANT_PPDOCLAYOUT_V3=0                      # force fp32
    PRISM_QUANT_PPDOCLAYOUT_V3=1                      # default _int8 sibling
    PRISM_QUANT_PPDOCLAYOUT_V3=/abs/path/to/x.onnx    # explicit graph

If a selected INT8 graph is missing the loader falls back to fp32 and says so,
rather than dying -- a half-built sweep stays runnable.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# key -> fp32 path, relative to repo root
GRAPHS = {
    "texo_encoder":    "Texo/model/onnx/encoder_model.onnx",
    "texo_decoder":    "Texo/model/onnx/decoder_model_merged.onnx",
    "ppocr_rec":       "weights/PP-OCRv6_rec_small.onnx",
    "ppocr_det":       "weights/PP-OCRv6_det_small.onnx",
    "ppocr_rec_en":    "weights/en_PP-OCRv4_rec.onnx",
    "ppdoclayout_v3":  "models/ppdoclayoutv3/PP-DocLayoutV3.onnx",
    # RapidTable ships slanet-plus inside the child venv's site-packages.
    "slanet_plus":     ".venv_rtable/Lib/site-packages/rapid_table/models/slanet-plus.onnx",
}

_announced: set = set()


def int8_path(key: str) -> Path:
    """The `_int8` sibling path for a graph (whether or not it exists yet)."""
    p = ROOT / GRAPHS[key]
    return p.with_name(p.stem + "_int8" + p.suffix)


def _selected(key: str) -> bool:
    override = os.environ.get(f"PRISM_QUANT_{key.upper()}")
    if override is not None:
        return override not in ("", "0")
    wanted = os.environ.get("PRISM_QUANT", "")
    if not wanted:
        return False
    if wanted.strip().lower() == "all":
        return True
    return key in {t.strip() for t in wanted.split(",") if t.strip()}


def graph_path(key: str, fp32: str | os.PathLike | None = None) -> str:
    """
    Resolve the graph to load for `key`.

    `fp32` lets a caller pass the path it already computed (keeps existing
    call sites authoritative about their own fp32 location).
    """
    base = Path(fp32) if fp32 is not None else ROOT / GRAPHS[key]

    override = os.environ.get(f"PRISM_QUANT_{key.upper()}", "")
    if override not in ("", "0", "1"):
        cand = Path(override)              # explicit graph file
    elif _selected(key):
        cand = base.with_name(base.stem + "_int8" + base.suffix)
    else:
        return str(base)

    if cand.exists():
        if key not in _announced:
            _announced.add(key)
            print(f"[quant] {key}: {cand.name}")
        return str(cand)

    if key not in _announced:
        _announced.add(key)
        print(f"[quant] {key}: INT8 graph missing ({cand.name}), using fp32")
    return str(base)


def active() -> dict:
    """{key: resolved path} for every registered graph -- for run manifests."""
    return {k: graph_path(k) for k in GRAPHS}
