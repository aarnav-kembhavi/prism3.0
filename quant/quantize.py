"""
Quantize one PRISM ONNX graph. Never overwrites fp32 -- writes an `_int8`
sibling (or an explicit --out).

Every graph goes through onnxruntime.quantization.shape_inference.quant_pre_process
first. If preprocessing fails, the graph is logged and SKIPPED, not forced.

    python quant/quantize.py texo_decoder
    python quant/quantize.py ppdoclayout_v3 --mode static --exclude-heads
"""
import argparse, json, os, shutil, sys, tempfile, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.quant_select import GRAPHS, int8_path

LOG = ROOT / "quant" / "quantize_log.json"


def _log(entry):
    log = json.loads(LOG.read_text()) if LOG.exists() else []
    log = [e for e in log if not (e["graph"] == entry["graph"] and e["variant"] == entry["variant"])]
    log.append(entry)
    LOG.write_text(json.dumps(log, indent=2))


def preprocess(src: Path, dst: Path) -> tuple[bool, str]:
    """quant_pre_process -> (ok, message). Never raises."""
    from onnxruntime.quantization.shape_inference import quant_pre_process
    try:
        quant_pre_process(str(src), str(dst), skip_optimization=False,
                          skip_onnx_shape=False, skip_symbolic_shape=False,
                          auto_merge=True, guess_output_rank=True)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run(key, mode, out, exclude_heads, calib_dir, calib_n, calib_method):
    from onnxruntime.quantization import (QuantType, quantize_dynamic,
                                          quantize_static, CalibrationMethod, QuantFormat)
    src = ROOT / GRAPHS[key]
    dst = Path(out) if out else int8_path(key)
    variant = mode + ("_heads_fp32" if exclude_heads else "")
    print(f"=== {key} [{variant}] ===\n  src {src}  ({src.stat().st_size/1e6:.1f} MB)\n  dst {dst}")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"quant_{key}_"))
    pre = tmpdir / "pre.onnx"
    t0 = time.perf_counter()
    ok, msg = preprocess(src, pre)
    print(f"  quant_pre_process: {msg}  ({time.perf_counter()-t0:.1f}s)")
    if not ok:
        _log({"graph": key, "variant": variant, "status": "skipped_preprocess_failed",
              "error": msg, "src_bytes": src.stat().st_size})
        print(f"  SKIPPED -- preprocessing failed, not forcing.")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 2

    excluded = []
    try:
        t0 = time.perf_counter()
        if mode == "dynamic":
            quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8,
                             per_channel=True, extra_options={"MatMulConstBOnly": True})
        else:
            sys.path.insert(0, str(ROOT / "quant"))
            from calib_reader import LayoutCalibrationReader, head_nodes
            if exclude_heads:
                excluded = head_nodes(str(pre))
                print(f"  excluding {len(excluded)} head nodes from quantization")
            reader = LayoutCalibrationReader(str(pre), calib_dir, calib_n)
            quantize_static(
                str(pre), str(dst), reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                calibrate_method=(CalibrationMethod.Percentile
                                  if calib_method == "percentile" else CalibrationMethod.MinMax),
                nodes_to_exclude=excluded,
                extra_options={"CalibMovingAverage": True} if calib_method == "minmax" else None,
            )
        dur = time.perf_counter() - t0
    except Exception as e:
        traceback.print_exc()
        _log({"graph": key, "variant": variant, "status": "failed_quantize",
              "error": f"{type(e).__name__}: {e}", "src_bytes": src.stat().st_size})
        shutil.rmtree(tmpdir, ignore_errors=True)
        return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    sb, db = src.stat().st_size, dst.stat().st_size
    entry = {"graph": key, "variant": variant, "status": "ok",
             "src": GRAPHS[key], "dst": str(dst.relative_to(ROOT)).replace("\\", "/"),
             "src_bytes": sb, "dst_bytes": db,
             "src_mb": round(sb/1e6, 2), "dst_mb": round(db/1e6, 2),
             "shrink_pct": round(100*(1-db/sb), 1),
             "quantize_s": round(dur, 1),
             "n_nodes_excluded": len(excluded),
             "calib": None if mode == "dynamic" else
                      {"method": calib_method, "n_images": calib_n, "dir": calib_dir}}
    _log(entry)
    print(f"  {sb/1e6:.1f} MB -> {db/1e6:.1f} MB  ({entry['shrink_pct']}% smaller)  in {dur:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", choices=sorted(GRAPHS))
    ap.add_argument("--mode", default="dynamic", choices=["dynamic", "static"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-heads", action="store_true")
    ap.add_argument("--calib-dir", default="quant/calib")
    ap.add_argument("--calib-n", type=int, default=100)
    ap.add_argument("--calib-method", default="minmax", choices=["minmax", "percentile"])
    a = ap.parse_args()
    sys.exit(run(a.graph, a.mode, a.out, a.exclude_heads, a.calib_dir, a.calib_n, a.calib_method))
