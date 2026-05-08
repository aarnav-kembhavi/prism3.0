import argparse, json, subprocess, sys, time
from pathlib import Path

import onnxruntime as ort
import torch

import benchmark_clean as bench
from texo.data.processor import EvalMERImageProcessor


def run_pytorch(images, proc, model_dir: Path, max_length: int) -> float:
    model = bench.Texo(model_dir, max_length=max_length)
    model.generate(bench.preprocess_image(images[0], proc))  # warmup (not timed)
    total = 0.0
    for p in images[1:]:
        px = bench.preprocess_image(p, proc)
        start = time.perf_counter()
        model.generate(px)
        total += (time.perf_counter() - start) * 1000.0
    return total / max(1, len(images) - 1)


def run_onnx(images, proc, encoder_path: Path, decoder_path: Path, bos_id: int, eos_id: int, max_length: int, threads: int | None) -> float:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads

    enc_sess = ort.InferenceSession(str(encoder_path), sess_options=so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(decoder_path), sess_options=so, providers=["CPUExecutionProvider"])

    warm_px = bench.preprocess_image(images[0], proc).numpy()
    bench.greedy_decode_onnx(enc_sess, dec_sess, warm_px, bos_id, eos_id, max_length)  # warmup (not timed)
    total = 0.0
    for p in images[1:]:
        px_np = bench.preprocess_image(p, proc).numpy()
        start = time.perf_counter()
        bench.greedy_decode_onnx(enc_sess, dec_sess, px_np, bos_id, eos_id, max_length)
        total += (time.perf_counter() - start) * 1000.0
    return total / max(1, len(images) - 1)


def benchmark_config(config: str, image_dir: Path, model_dir: Path, limit: int) -> dict:
    onnx_dir = model_dir / "onnx"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if limit and limit > 0:
        images = images[: limit]
    if len(images) < 2:
        raise SystemExit("Need at least 2 images in the image directory.")

    proc = EvalMERImageProcessor(image_size={"width": bench.DEFAULT_IMAGE_SIZE, "height": bench.DEFAULT_IMAGE_SIZE})
    bos_id, eos_id, max_length = bench.load_generation_params(model_dir)

    if config == "single":
        torch.set_num_threads(1)  # requirement: only set torch threads
        onnx_threads = 1
    else:
        onnx_threads = None

    pt_ms = run_pytorch(images, proc, model_dir, max_length)
    onnx_ms = run_onnx(images, proc, onnx_dir / "encoder_model.onnx", onnx_dir / "decoder_model.onnx", bos_id, eos_id, max_length, threads=onnx_threads)
    return {"torch_ms": pt_ms, "onnx_ms": onnx_ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all images")
    ap.add_argument("--mode", type=str, default="", help="internal: single|multi")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    image_dir = base / "TechnoSelection/test_img"
    model_dir = base / "model"

    if args.mode:
        res = benchmark_config(args.mode, image_dir, model_dir, args.limit)
        print(json.dumps(res))
        return

    # Orchestrate in separate processes so "multi" truly uses defaults.
    cmd = [sys.executable, str(Path(__file__).resolve()), "--limit", str(args.limit), "--mode"]
    def run(mode: str) -> dict:
        p = subprocess.run(cmd + [mode], check=True, capture_output=True, text=True)
        return json.loads(p.stdout.strip().splitlines()[-1])

    single = run("single")
    multi = run("multi")

    single_torch_ms, single_onnx_ms = single["torch_ms"], single["onnx_ms"]
    multi_torch_ms, multi_onnx_ms = multi["torch_ms"], multi["onnx_ms"]

    print(f"{'Config':<12}{'PyTorch(ms)':>12}{'ONNX(ms)':>12}")
    print("-" * 44)
    print(f"{'Single':<12}{single_torch_ms:>12.2f}{single_onnx_ms:>12.2f}")
    print(f"{'Multi':<12}{multi_torch_ms:>12.2f}{multi_onnx_ms:>12.2f}")

    pt_speed = (single_torch_ms - multi_torch_ms) / single_torch_ms * 100.0
    onnx_speed = (single_onnx_ms - multi_onnx_ms) / single_onnx_ms * 100.0
    print(f"Speedup (single -> multi): PyTorch {pt_speed:+.1f}%, ONNX {onnx_speed:+.1f}%")


if __name__ == "__main__":
    main()

