import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import benchmark_clean as bench
from texo.data.processor import EvalMERImageProcessor, TextProcessor

logging.getLogger().setLevel(logging.ERROR)
ort.set_default_logger_severity(3)


def run_onnx_for_size(images, proc, enc_sess, dec_sess, bos_id: int, eos_id: int, max_length: int) -> float:
    # warmup on first image (not timed)
    warm_px = bench.preprocess_image(images[0], proc).numpy()
    _ = bench.greedy_decode_onnx(enc_sess, dec_sess, warm_px, bos_id, eos_id, max_length)

    total = 0.0
    for p in images[1:]:
        px_np = bench.preprocess_image(p, proc).numpy()
        start = time.perf_counter()
        _ = bench.greedy_decode_onnx(enc_sess, dec_sess, px_np, bos_id, eos_id, max_length)
        total += (time.perf_counter() - start) * 1000.0
    return total / max(1, len(images) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all images, otherwise first N")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    image_dir = base / "TechnoSelection/test_img"
    model_dir = base / "model"
    onnx_dir = model_dir / "onnx"

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if len(images) < 2:
        raise SystemExit("Need at least 2 images in test_img.")

    bos_id, eos_id, max_length = bench.load_generation_params(model_dir)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_sess = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

    tokenizer = TextProcessor(
        config={
            "tokenizer_path": str(base / "data/tokenizer"),
            "tokenizer_config": {"add_special_tokens": True, "max_length": 1024},
        }
    ).tokenizer

    sizes = [256, 384, 512]
    results = {}

    for sz in sizes:
        proc = EvalMERImageProcessor(image_size={"width": sz, "height": sz})
        avg_ms = run_onnx_for_size(images, proc, enc_sess, dec_sess, bos_id, eos_id, max_length)
        results[sz] = avg_ms

        # Optional: print 1 sample output (first image) to verify correctness.
        px_np = bench.preprocess_image(images[0], proc).numpy()
        ids = bench.greedy_decode_onnx(enc_sess, dec_sess, px_np, bos_id, eos_id, max_length)
        out = tokenizer.batch_decode(np.asarray(ids), skip_special_tokens=True)[0]
        out = out.replace("\n", " ").replace("\r", " ")
        out = out[:80] + ("..." if len(out) > 80 else "")
        print(f"Sample({sz}): {out}")

    print("")
    print(f"{'Image Size':<12}{'Avg Latency(ms)':>18}")
    print("-" * 31)
    for sz in sizes:
        print(f"{sz:<12}{results[sz]:>18.2f}")


if __name__ == "__main__":
    main()

