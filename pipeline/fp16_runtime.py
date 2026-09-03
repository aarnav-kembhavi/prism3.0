"""
fp16 graphs on disk, fp32 at run time (the Variant A half of the fp16 pass).

A graph converted with onnxconverter_common.float16.convert_float_to_float16
is half the size on disk. Variant A pays only the fp16 *weight rounding* cost:
at load time the graph is converted back to fp32 in memory and the
InferenceSession is built from those bytes, so kernels, RAM and latency are
whatever they were before. Variant B skips the back-conversion and lets ORT
run the fp16 graph natively.

Why a monkeypatch rather than a path swap
-----------------------------------------
Only three stages build their own InferenceSession. The rest hand a *path* to
RapidOCR / RapidTable, which open it themselves, so returning bytes from
graph_path() could not work. Patching ort.InferenceSession catches every
consumer uniformly, and composes with the two patches the codebase already
installs (models_interface.py's no-arena patch, text_worker.py's re-apply in
the subprocess): each captures whatever is currently bound and forwards **kw,
so the wrappers chain in either order.

This module deliberately imports only onnx + numpy -- no onnxconverter_common.
pipeline/rtable_child.py runs in .venv_rtable, and the reverse conversion has
to work there too. Only the offline forward conversion needs the converter.
"""
import os
from pathlib import Path

FLOAT = 1
FLOAT16 = 10


def _iter_graphs(graph):
    """The graph and every subgraph nested in a node attribute, recursively."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from _iter_graphs(attr.g)
            for g in attr.graphs:
                yield from _iter_graphs(g)


def _fix_type(tp):
    """Retype FLOAT16 -> FLOAT in a TypeProto, through sequence/optional/map."""
    which = tp.WhichOneof("value")
    if which == "tensor_type":
        if tp.tensor_type.elem_type == FLOAT16:
            tp.tensor_type.elem_type = FLOAT
    elif which == "sequence_type":
        _fix_type(tp.sequence_type.elem_type)
    elif which == "optional_type":
        _fix_type(tp.optional_type.elem_type)
    elif which == "map_type":
        _fix_type(tp.map_type.value_type)


def _fix_tensor(t):
    """Retype a FLOAT16 TensorProto to FLOAT in place. Returns True if changed."""
    if t.data_type != FLOAT16:
        return False
    import numpy as np
    from onnx import numpy_helper
    arr = numpy_helper.to_array(t).astype(np.float32)
    new = numpy_helper.from_array(arr, t.name)
    t.CopyFrom(new)
    return True


def fp16_model_to_fp32(model):
    """
    Invert convert_float_to_float16 in memory.

    Three edits, applied to the graph and every subgraph:
      1. FLOAT16 initializers and node-attribute tensors -> FLOAT
      2. FLOAT16 tensor types on inputs/outputs/value_info -> FLOAT
      3. Cast(to=FLOAT16) -> Cast(to=FLOAT)

    Step 3 is safe to do blindly *here*: quant/fp16_convert.py asserts that the
    source fp32 graph contains no Cast-to-float16 of its own (verified: zero
    across all seven graphs), so every such node was inserted by the converter.
    Rewriting them to Cast(to=FLOAT) rather than splicing them out keeps the
    graph topology identical -- they become identity casts, which ORT's
    optimizer drops.

    The result is the original fp32 graph with every weight rounded through
    fp16, which is exactly the cost Variant A is meant to measure.
    """
    n_init = n_cast = n_attr = 0
    for g in _iter_graphs(model.graph):
        for init in g.initializer:
            n_init += _fix_tensor(init)
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            _fix_type(vi.type)
        for node in g.node:
            for attr in node.attribute:
                if attr.HasField("t"):
                    n_attr += _fix_tensor(attr.t)
                for t in attr.tensors:
                    n_attr += _fix_tensor(t)
                if attr.HasField("tp"):
                    _fix_type(attr.tp)
                for tp in attr.type_protos:
                    _fix_type(tp)
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == FLOAT16:
                        attr.i = FLOAT
                        n_cast += 1
    return model, {"initializers": n_init, "attr_tensors": n_attr, "casts": n_cast}


def fp32_bytes_from_fp16(path):
    """Serialized fp32 model reconstructed from an fp16 graph on disk."""
    import onnx
    model = onnx.load(str(path))
    model, stats = fp16_model_to_fp32(model)
    return model.SerializeToString(), stats


_installed = False


def install():
    """
    Patch ort.InferenceSession so a *_fp16.onnx path is loaded as fp32 bytes.

    Idempotent, and a no-op unless Variant A is active. Installed at
    pipeline.quant_select import time: Python binds the callable before it
    evaluates arguments, so `ort.InferenceSession(graph_path(...))` in
    math_worker_onnx.py would already hold the unpatched function if this were
    installed lazily from graph_path().
    """
    global _installed
    if _installed or os.environ.get("PRISM_FP16_RUNTIME", "fp32") != "fp32":
        return
    try:
        import onnxruntime as ort
    except Exception:
        return
    _installed = True
    _orig = ort.InferenceSession

    def _fp16_aware_session(path_or_bytes, sess_options=None, providers=None, **kw):
        p = path_or_bytes
        if isinstance(p, (str, Path)) and str(p).endswith("_fp16.onnx"):
            try:
                blob, stats = fp32_bytes_from_fp16(p)
                _record(str(p), stats)
                path_or_bytes = blob
            except Exception as e:
                # Never silently fall through to the fp16 graph: that would
                # measure Variant B while reporting Variant A.
                raise RuntimeError(
                    "fp16->fp32 back-conversion failed for %s: %s" % (p, e)) from e
        return _orig(path_or_bytes, sess_options=sess_options, providers=providers, **kw)

    ort.InferenceSession = _fp16_aware_session
    global _orig_session, _wrapper
    _orig_session, _wrapper = _orig, _fp16_aware_session
    sweep_modules()


_orig_session = None
_wrapper = None


def sweep_modules():
    """
    Rebind `InferenceSession` in modules that already from-imported it.

    rapidocr_onnxruntime/utils/infer_engine.py does
    `from onnxruntime import InferenceSession` at import time, which copies the
    name into its own namespace -- rebinding onnxruntime.InferenceSession
    afterwards does nothing for it. The codebase's own patches only survive
    this by luck of ordering (text_worker.py patches before it imports
    RapidOCR); this module is imported later, so it has to fix up the copies.

    Idempotent, and only ever replaces the exact function object we wrapped.
    """
    if _wrapper is None:
        return []
    import sys
    hit = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            if getattr(mod, "InferenceSession", None) is _orig_session:
                mod.InferenceSession = _wrapper
                hit.append(name)
        except Exception:
            continue
    return hit


def _record_sweep(mods):
    man = os.environ.get("PRISM_QUANT_MANIFEST", "")
    if not man or not mods:
        return
    import json
    try:
        with open(man, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "event": "fp16_module_sweep",
                                "modules": mods}) + chr(10))
    except OSError:
        pass


def _record(path, stats):
    """Append the back-conversion to the run manifest (worker stdout is lost)."""
    man = os.environ.get("PRISM_QUANT_MANIFEST", "")
    if not man:
        return
    import json
    try:
        with open(man, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "event": "fp16_to_fp32",
                                "file": Path(path).name, **stats}) + "\n")
    except OSError:
        pass
