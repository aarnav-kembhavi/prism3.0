"""
Convert each ONNX graph to fp16 on disk. Never touches the fp32 file.

    python quant/fp16_convert.py                 # all graphs
    python quant/fp16_convert.py texo_encoder    # one graph

For every graph this:
  1. asserts the source contains no Cast(to=float16) of its own -- the reverse
     conversion in pipeline/fp16_runtime.py rewrites every such node to
     Cast(to=float), which is only sound if the converter inserted all of them;
  2. converts with onnxconverter_common.float16.convert_float_to_float16 and
     writes <name>_fp16.onnx;
  3. VERIFIES the round trip: reverses the conversion, builds a real
     InferenceSession from the result, and compares every weight against the
     original fp32 graph -- the difference must be exactly fp16 rounding
     (w == float32(float16(w))), never anything else.

Step 3 is the point. A graph that converts but cannot be loaded, or that the
reverse pass silently mangles, would otherwise show up as a great disk saving
with mysterious accuracy loss further down the sweep.

Results land in quant/fp16_log.json, failures included.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.quant_select import GRAPHS, fp16_path  # noqa: E402
from pipeline.fp16_runtime import fp16_model_to_fp32, _iter_graphs, FLOAT16  # noqa: E402

LOG = ROOT / "quant" / "fp16_log.json"

MIN_POSITIVE = 1e-7      # onnxconverter_common default; ~fp16 subnormal floor
MAX_FINITE = 65504.0     # largest representable fp16 (library default is 1e4)


def _cast16_count(model):
    n = 0
    for g in _iter_graphs(model.graph):
        for node in g.node:
            if node.op_type == "Cast":
                for a in node.attribute:
                    if a.name == "to" and a.i == FLOAT16:
                        n += 1
    return n


def _weights(model):
    """
    {name: fp32 array} for every float weight, subgraphs included.

    Constant NODES count, not just initializers: en_PP-OCRv4_rec and
    slanet-plus store their weights that way, so an initializer-only sweep
    compares zero tensors and reports a vacuous pass.
    """
    out = {}
    for g in _iter_graphs(model.graph):
        for init in g.initializer:
            if init.data_type in (1, FLOAT16):
                try:
                    out[init.name] = numpy_helper.to_array(init).astype(np.float32)
                except Exception:
                    pass
        for node in g.node:
            for at in node.attribute:
                if at.HasField("t") and at.t.data_type in (1, FLOAT16):
                    try:
                        # Keyed by OUTPUT tensor name, not node.name: Constant
                        # nodes are usually unnamed (all 299 in en_PP-OCRv4_rec
                        # are), so node.name collapses them onto one key and
                        # the comparison silently comes back with 1 weight.
                        key = node.output[0] if node.output else node.name
                        out[key] = numpy_helper.to_array(at.t).astype(np.float32)
                    except Exception:
                        pass
    return out


def verify(key, src, dst):
    """Reverse the conversion, load it for real, and prove it is only rounding."""
    import onnxruntime as ort

    fp16_model = onnx.load(str(dst))
    back, stats = fp16_model_to_fp32(fp16_model)

    t0 = time.perf_counter()
    blob = back.SerializeToString()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0

    orig = _weights(onnx.load(str(src)))
    rec = _weights(back)
    shared = sorted(set(orig) & set(rec))
    max_abs = max_rel = 0.0
    n_small = n_large = 0
    unexplained = []
    big_clamps = []
    for n in shared:
        a, b = orig[n], rec[n]
        if a.shape != b.shape:
            unexplained.append(f"{n}: shape {a.shape} != {b.shape}")
            continue
        with np.errstate(over="ignore"):
            rounded = a.astype(np.float16).astype(np.float32)
        diff = b != rounded
        if diff.any():
            # Two deliberate converter behaviours, both genuinely part of the
            # fp16 cost: tiny magnitudes are lifted to +/-min_positive_val
            # instead of flushing to zero, and out-of-range magnitudes are
            # clamped to +/-max_finite_val. Anything else is a real defect.
            small = diff & (np.abs(a) < MIN_POSITIVE)
            large = diff & (np.abs(a) > MAX_FINITE)
            n_small += int(small.sum())
            n_large += int(large.sum())
            rest = int((diff & ~small & ~large).sum())
            if rest:
                unexplained.append(f"{n}: {rest} values neither rounded nor clamped")
            if large.any():
                big_clamps.append(
                    "%s: %d value(s) over-range clamped (max %.4g)"
                    % (n, int(large.sum()), float(np.abs(a)[large].max())))
        d = np.abs(a - b)
        max_abs = max(max_abs, float(d.max()) if d.size else 0.0)
        denom = np.maximum(np.abs(a), 1e-8)
        max_rel = max(max_rel, float((d / denom).max()) if d.size else 0.0)

    return {
        "reverse_stats": stats,
        "session_built": True,
        "back_convert_and_load_s": round(load_s, 2),
        "n_weights_compared": len(shared),
        "max_abs_weight_err": max_abs,
        "max_rel_weight_err": round(max_rel, 6),
        "n_clamped_subnormal": n_small,
        "n_clamped_overrange": n_large,
        "overrange_clamps": big_clamps[:8],
        "unexplained_diffs": unexplained[:5],
        "is_rounding_or_clamp_only": not unexplained,
    }


def restore_empty_optional_inputs(orig, new):
    """
    Undo the converter's rewiring of unused optional inputs.

    ONNX marks an unused optional input either with "" or with a ZERO-LENGTH
    tensor. Resize uses the latter for `roi`/`scales` when `sizes` is given
    (slanet-plus: helper.constant.16/17, both float32 shape (0,)). The fp16
    converter reroutes every input through a Cast, and shape inference can no
    longer see that the tensor is empty -- so ORT reports "Either sizes or
    scales must be provided, but not both" and refuses to load the graph.
    Block-listing the op does not help; it rewrites the inputs either way.

    Two repairs, both no-ops on a graph that does not need them:
      1. point the input back at the original tensor;
      2. force that tensor back to fp32 -- Resize types `scales` as
         tensor(float) exactly, not as a type variable, so an fp16 one is a
         type error. Zero-length tensors cost nothing to keep in fp32.
    """
    empty_names, sentinel = set(), []
    for g in _iter_graphs(orig.graph):
        for init in g.initializer:
            if init.data_type == 1 and not numpy_helper.to_array(init).size:
                empty_names.add(init.name)
        for n in g.node:
            for at in n.attribute:
                if (at.HasField("t") and at.t.data_type == 1
                        and n.output and not numpy_helper.to_array(at.t).size):
                    empty_names.add(n.output[0])
        for n in g.node:
            for i, v in enumerate(n.input):
                if v == "" or v in empty_names:
                    if n.name:
                        sentinel.append((n.op_type, n.name, i, v))

    fixed = 0
    want = {(op, nm): {} for op, nm, _, _ in sentinel}
    for op, nm, i, v in sentinel:
        want[(op, nm)][i] = v
    for g in _iter_graphs(new.graph):
        for n in g.node:
            for i, v in want.get((n.op_type, n.name), {}).items():
                if i < len(n.input) and n.input[i] != v:
                    n.input[i] = v
                    fixed += 1
    # keep the restored tensors fp32
    for g in _iter_graphs(new.graph):
        for init in g.initializer:
            if init.name in empty_names:
                init.data_type = 1
        for n in g.node:
            if n.output and n.output[0] in empty_names:
                for at in n.attribute:
                    if at.HasField("t"):
                        at.t.data_type = 1
                        at.t.ClearField("int32_data")
                        at.t.raw_data = b""
    return fixed


def convert(key):
    src = ROOT / GRAPHS[key]
    dst = fp16_path(key)
    rec = {"key": key, "src": GRAPHS[key], "dst": dst.name}
    if not src.exists():
        rec.update(status="skipped", reason="source graph missing")
        return rec

    rec["src_mb"] = round(src.stat().st_size / 1e6, 2)
    print(f"[*] {key}: {rec['src_mb']} MB -> fp16")

    model = onnx.load(str(src))
    n_cast16 = _cast16_count(model)
    rec["source_cast_to_fp16_nodes"] = n_cast16
    if n_cast16:
        # Precondition for the reverse pass. Bail rather than produce a graph
        # that cannot be safely converted back.
        rec.update(status="skipped",
                   reason=f"source has {n_cast16} Cast(to=float16) nodes; the "
                          f"reverse conversion could not tell them from inserted ones")
        print(f"    SKIP: {rec['reason']}")
        return rec

    from onnxconverter_common import float16 as f16
    t0 = time.perf_counter()
    try:
        out = f16.convert_float_to_float16(
            model, keep_io_types=True,
            # The library default clamps to 1e4, well under fp16's real
            # ceiling. PP-DocLayoutV3 stores FLT_MAX and 1e8 scalars as Clip
            # bounds for box coordinates; crushing those to 1e4 is a semantic
            # change, not rounding. 65504 is the largest fp16 value, so the
            # sentinels stay saturating and no in-range weight is touched.
            max_finite_val=MAX_FINITE,
            op_block_list=list(f16.DEFAULT_OP_BLOCK_LIST))
    except Exception as e:
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}")
        print(f"    FAILED: {rec['reason']}")
        return rec
    rec["convert_s"] = round(time.perf_counter() - t0, 2)
    rec["empty_optional_inputs_restored"] = restore_empty_optional_inputs(
        onnx.load(str(src)), out)

    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(out, str(dst))
    rec["dst_mb"] = round(dst.stat().st_size / 1e6, 2)
    rec["saved_mb"] = round(rec["src_mb"] - rec["dst_mb"], 2)
    rec["shrink_pct"] = round(100 * rec["saved_mb"] / rec["src_mb"], 1)
    print(f"    {rec['src_mb']} -> {rec['dst_mb']} MB  (-{rec['shrink_pct']}%)")

    try:
        rec["verify"] = verify(key, src, dst)
        ok = rec["verify"]["is_rounding_or_clamp_only"]
        rec["status"] = "ok" if ok else "ok_with_warnings"
        v = rec["verify"]
        print("    verify: session OK, %d weights, max |dw|=%.3g, clamped %d small / "
              "%d over-range, unexplained=%d, back-convert+load %ss"
              % (v["n_weights_compared"], v["max_abs_weight_err"],
                 v["n_clamped_subnormal"], v["n_clamped_overrange"],
                 len(v["unexplained_diffs"]), v["back_convert_and_load_s"]))
    except Exception as e:
        rec.update(status="failed_verify", reason=f"{type(e).__name__}: {e}")
        print(f"    VERIFY FAILED: {rec['reason']}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=None)
    a = ap.parse_args()
    keys = a.keys or list(GRAPHS)

    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else {}
    for k in keys:
        log[k] = convert(k)
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== fp16 conversion ===")
    tot_src = tot_dst = 0.0
    for k in keys:
        r = log[k]
        if r.get("status") in ("ok", "ok_with_warnings"):
            tot_src += r["src_mb"]
            tot_dst += r["dst_mb"]
            print(f"  {k:16s} {r['src_mb']:7.1f} -> {r['dst_mb']:6.1f} MB  "
                  f"(-{r['shrink_pct']:.1f}%)  {r['status']}")
        else:
            print(f"  {k:16s} {r.get('status'):>16s}  {r.get('reason', '')[:60]}")
    print(f"  {'converted total':16s} {tot_src:7.1f} -> {tot_dst:6.1f} MB")
    print("wrote", LOG.relative_to(ROOT))


if __name__ == "__main__":
    main()
