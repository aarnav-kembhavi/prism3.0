import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import benchmark_clean as bench
from texo.data.processor import EvalMERImageProcessor

logging.getLogger().setLevel(logging.ERROR)
ort.set_default_logger_severity(3)


def load_params(model_dir: Path) -> tuple[int, int, int]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    bos = int(gen.get("bos_token_id", cfg.get("decoder_start_token_id", 0)))
    eos = int(gen.get("eos_token_id", cfg.get("eos_token_id", 2)))
    # For this experiment we want long generations, so prefer generation_config max_length.
    max_length = int(gen.get("max_length", cfg.get("decoder", {}).get("max_length", 20)))
    return bos, eos, max_length


def category_for_len(n: int) -> str:
    if n < 20:
        return "Short"
    if n <= 50:
        return "Medium"
    return "Long"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all images")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    image_dir = base / "TechnoSelection/test_img"
    model_dir = base / "model"
    onnx_dir = model_dir / "onnx"

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if len(images) == 0:
        raise SystemExit(f"No images found in {image_dir}")

    proc = EvalMERImageProcessor(image_size={"width": bench.DEFAULT_IMAGE_SIZE, "height": bench.DEFAULT_IMAGE_SIZE})
    bos_id, eos_id, max_length = load_params(model_dir)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

    lengths: list[int] = []
    latencies_ms: list[float] = []
    by_cat = {"Short": [], "Medium": [], "Long": []}  # list of (len, latency_ms)

    for img_path in images:
        px_np = bench.preprocess_image(img_path, proc).numpy()
        start = time.perf_counter()
        ids = bench.greedy_decode_onnx(enc_sess, dec_sess, px_np, bos_id, eos_id, max_length)
        ms = (time.perf_counter() - start) * 1000.0

        seq_len = int(ids.shape[1])
        lengths.append(seq_len)
        latencies_ms.append(ms)

        cat = category_for_len(seq_len)
        by_cat[cat].append((seq_len, ms))

    def avg(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def avg_pair(pairs, idx: int) -> float:
        return float(np.mean([p[idx] for p in pairs])) if pairs else 0.0

    print(f"{'Category':<10}{'Avg Length':>14}{'Avg Latency(ms)':>18}")
    print("-" * 44)
    for cat in ["Short", "Medium", "Long"]:
        pairs = by_cat[cat]
        print(f"{cat:<10}{avg_pair(pairs,0):>14.2f}{avg_pair(pairs,1):>18.2f}")

    x = np.array(lengths, dtype=np.float64)
    y = np.array(latencies_ms, dtype=np.float64)
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        direction = "positive" if r >= 0 else "negative"
        print(f"Correlation insight: latency vs length Pearson r={r:+.3f} ({direction})")
    else:
        print("Correlation insight: insufficient variance in length/latency to compute correlation.")


if __name__ == "__main__":
    main()

