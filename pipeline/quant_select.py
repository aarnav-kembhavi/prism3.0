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

fp16 lives in its own namespace, so an int8 and an fp16 sweep can never
silently shadow one another:

    PRISM_FP16=texo_encoder,ppocr_rec   # those load their _fp16 sibling
    PRISM_FP16=all                      # every graph with an _fp16 sibling
    PRISM_FP16_TEXO_ENCODER=0           # per-graph override, same rules

    PRISM_FP16_RUNTIME=fp32   # VARIANT A: back-convert to fp32 in memory at
                              # load time (default). Disk halves; RAM, kernels
                              # and latency unchanged; cost is weight rounding.
    PRISM_FP16_RUNTIME=fp16   # VARIANT B: run the fp16 graph natively.

Naming a graph in both PRISM_QUANT and PRISM_FP16 is a configuration error and
raises, rather than quietly picking one -- an earlier round of this sweep spent
real time measuring an fp32 stack it believed was quantized.

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


def _selected(key: str, var: str = "PRISM_QUANT") -> bool:
    override = os.environ.get(f"{var}_{key.upper()}")
    if override is not None:
        return override not in ("", "0")
    wanted = os.environ.get(var, "")
    if not wanted:
        return False
    if wanted.strip().lower() == "all":
        return True
    return key in {t.strip() for t in wanted.split(",") if t.strip()}


def fp16_path(key: str) -> Path:
    """The `_fp16` sibling path for a graph (whether or not it exists yet)."""
    p = ROOT / GRAPHS[key]
    return p.with_name(p.stem + "_fp16" + p.suffix)


def graph_path(key: str, fp32: str | os.PathLike | None = None) -> str:
    """
    Resolve the graph to load for `key`.

    `fp32` lets a caller pass the path it already computed (keeps existing
    call sites authoritative about their own fp32 location).
    """
    base = Path(fp32) if fp32 is not None else ROOT / GRAPHS[key]

    want_int8 = _selected(key, "PRISM_QUANT")
    want_fp16 = _selected(key, "PRISM_FP16")
    if want_int8 and want_fp16:
        raise RuntimeError(
            f"{key} is selected in both PRISM_QUANT and PRISM_FP16; refusing to "
            f"guess which precision was meant")

    q_override = os.environ.get(f"PRISM_QUANT_{key.upper()}", "")
    f_override = os.environ.get(f"PRISM_FP16_{key.upper()}", "")
    if q_override not in ("", "0", "1"):
        cand, prec = Path(q_override), "int8"      # explicit graph file
    elif f_override not in ("", "0", "1"):
        cand, prec = Path(f_override), "fp16"
    elif want_int8:
        cand, prec = base.with_name(base.stem + "_int8" + base.suffix), "int8"
    elif want_fp16:
        cand, prec = base.with_name(base.stem + "_fp16" + base.suffix), "fp16"
    else:
        return str(base)

    if cand.exists():
        if prec == "fp16":
            # Catch consumers that from-imported InferenceSession after this
            # module was imported (RapidOCR, RapidTable). Cheap and idempotent.
            try:
                from pipeline.fp16_runtime import sweep_modules, _record_sweep
                _record_sweep(sweep_modules())
            except Exception:
                pass
        _announce(key, cand, True, precision=prec)
        return str(cand)

    _announce(key, base, False, missing=cand.name, precision=prec)
    return str(base)


def _announce(key, path, is_reduced, missing=None, precision="int8"):
    """
    Record the resolution once per process.

    Model loads happen in worker SUBPROCESSES whose stdout is not captured by
    the benchmark runner, so a print alone cannot prove which graph a stage
    actually opened. PRISM_QUANT_MANIFEST=<jsonl> makes every process append
    what it resolved, which is what the sweep audits -- otherwise a silent
    fp32 fallback would look exactly like "quantization cost us nothing".
    """
    if key not in _announced:
        _announced.add(key)
        if missing:
            print(f"[quant] {key}: {precision} graph missing ({missing}), using fp32")
        elif is_reduced:
            print(f"[quant] {key}: {Path(path).name} ({precision})")

    man = os.environ.get("PRISM_QUANT_MANIFEST", "")
    if not man:
        return
    import json
    import inspect
    # Record the calling module: several stages build their own model paths
    # (text_worker.py replicates models_interface), so knowing that *some*
    # process resolved a graph is not enough -- we need to know the stage that
    # actually runs page OCR resolved it too.
    caller = "?"
    try:
        for fr in inspect.stack()[1:6]:
            mod = Path(fr.filename).name
            if mod != "quant_select.py":
                caller = mod
                break
    except Exception:
        pass
    try:
        with open(man, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "key": key, "caller": caller,
                                "file": Path(path).name,
                                "int8": bool(is_reduced) and precision == "int8",
                                "fp16": bool(is_reduced) and precision == "fp16",
                                "precision": precision if is_reduced else "fp32",
                                "runtime": os.environ.get("PRISM_FP16_RUNTIME", "fp32"),
                                "missing": missing}) + "\n")
    except OSError:
        pass


def active() -> dict:
    """{key: resolved path} for every registered graph -- for run manifests."""
    return {k: graph_path(k) for k in GRAPHS}


# Variant A needs ort.InferenceSession patched before ANY session is built.
# Python binds the callable before evaluating arguments, so installing this
# lazily from graph_path() would be too late for call sites of the shape
# `ort.InferenceSession(graph_path(...), ...)`. Import time is early enough:
# every call site imports this module on a line above its session creation.
# No-ops unless PRISM_FP16_RUNTIME is fp32 and something selects an fp16 graph.
if os.environ.get("PRISM_FP16") or any(
        k.startswith("PRISM_FP16_") and k != "PRISM_FP16_RUNTIME" for k in os.environ):
    try:
        from pipeline.fp16_runtime import install as _install_fp16
        _install_fp16()
    except Exception as _e:      # never break a normal run over the sweep
        print(f"[quant] fp16 runtime patch unavailable: {_e}")
