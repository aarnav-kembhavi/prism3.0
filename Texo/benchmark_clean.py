import argparse, json, time, logging, os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from safetensors.torch import load_file

from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet

logging.getLogger().setLevel(logging.ERROR)
ort.set_default_logger_severity(3)

# Disable PyTorch threading to avoid CPU thread contention with ONNXRuntime.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

num_threads = min(6, os.cpu_count())
print(f"Using {num_threads} CPU threads for ONNX")

try:
    from texo import Texo  # type: ignore
except Exception:
    class Texo:  # fallback: load FormulaNet weights without using HF model wrappers
        def __init__(self, model_dir: Path, max_length: int):
            cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            cfg.pop("pretrained", None)
            self.model = FormulaNet(cfg).to("cpu").eval()
            self.model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
            self.max_length = int(max_length)

        @torch.inference_mode()
        def generate(self, pixel_values: torch.Tensor) -> None:
            _ = self.model.generate(pixel_values=pixel_values, num_beams=1, do_sample=False, max_length=self.max_length)


MODEL_DIR = Path("model")
IMAGE_DIR = Path("TechnoSelection/test_img")
DEFAULT_IMAGE_SIZE = 384


def get_model_size_mb(paths: list[Path]) -> float:
    total = sum(p.stat().st_size for p in paths)
    return total / (1024 * 1024)


def load_generation_params(model_dir: Path) -> tuple[int, int, int]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    bos = int(gen.get("bos_token_id", cfg.get("decoder_start_token_id", 0)))
    eos = int(gen.get("eos_token_id", cfg.get("eos_token_id", 2)))
    dec_max = int(cfg.get("decoder", {}).get("max_length", gen.get("max_length", 20)))
    gen_max = int(gen.get("max_length", dec_max))
    return bos, eos, min(dec_max, gen_max)


def preprocess_image(image_path: Path, proc: EvalMERImageProcessor) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return proc(img).unsqueeze(0).contiguous()  # [1,3,H,W]


def greedy_decode_onnx(enc_sess, dec_sess, pixel_values_np, bos_id, eos_id, max_length):
    enc_out = enc_sess.run(None, {"pixel_values": pixel_values_np})[0]
    ids = np.array([[bos_id]], dtype=np.int64)

    for _ in range(1, max_length):
        logits = dec_sess.run(None, {
            "input_ids": ids,
            "encoder_hidden_states": enc_out
        })[0]

        next_id = int(np.argmax(logits[0, -1], axis=-1))
        ids = np.concatenate([ids, np.array([[next_id]], dtype=np.int64)], axis=1)

        if next_id == eos_id:
            break

    return ids   # ← THIS LINE IS MISSING IN YOUR CURRENT SETUP


def run_pytorch(images: list[Path], proc: EvalMERImageProcessor, model_dir: Path, max_length: int) -> float:
    model = Texo(model_dir, max_length)
    model.generate(preprocess_image(images[0], proc))  # warmup (not timed)
    total = 0.0
    for p in images[1:]:
        px = preprocess_image(p, proc)
        start = time.perf_counter()
        model.generate(px)
        total += (time.perf_counter() - start) * 1000.0
    return total / max(1, len(images) - 1)


def run_onnx(images: list[Path], proc: EvalMERImageProcessor, encoder_path: Path, decoder_path: Path, bos_id: int, eos_id: int, max_length: int) -> float:
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_sess = ort.InferenceSession(str(encoder_path), sess_options=so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(decoder_path), sess_options=so, providers=["CPUExecutionProvider"])
    warm_px = preprocess_image(images[0], proc).numpy()
    greedy_decode_onnx(enc_sess, dec_sess, warm_px, bos_id, eos_id, max_length)  # warmup (not timed)
    total = 0.0
    for p in images[1:]:
        px = preprocess_image(p, proc).numpy()
        start = time.perf_counter()
        greedy_decode_onnx(enc_sess, dec_sess, px, bos_id, eos_id, max_length)
        total += (time.perf_counter() - start) * 1000.0
    return total / max(1, len(images) - 1)


def quantize_int8(encoder_src: Path, decoder_src: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_dst, dec_dst = out_dir / "encoder_int8.onnx", out_dir / "decoder_int8.onnx"
    if not enc_dst.exists():
        quantize_dynamic(str(encoder_src), str(enc_dst), weight_type=QuantType.QInt8)
    if not dec_dst.exists():
        quantize_dynamic(str(decoder_src), str(dec_dst), weight_type=QuantType.QInt8)
    return enc_dst, dec_dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, default=str(IMAGE_DIR))
    ap.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    image_dir, model_dir = Path(args.image_dir), Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if len(images) < 2:
        raise SystemExit("Need at least 2 images in the image directory.")

    proc = EvalMERImageProcessor(image_size={"width": DEFAULT_IMAGE_SIZE, "height": DEFAULT_IMAGE_SIZE})
    bos_id, eos_id, max_length = load_generation_params(model_dir)

    size_pt = get_model_size_mb([model_dir / "model.safetensors"])
    size_fp32 = get_model_size_mb([onnx_dir / "encoder_model.onnx", onnx_dir / "decoder_model.onnx"])
    avg_pt = run_pytorch(images, proc, model_dir, max_length)
    avg_fp32 = run_onnx(images, proc, onnx_dir / "encoder_model.onnx", onnx_dir / "decoder_model.onnx", bos_id, eos_id, max_length)

    enc_int8, dec_int8 = quantize_int8(onnx_dir / "encoder_model.onnx", onnx_dir / "decoder_model.onnx", onnx_dir)
    size_int8 = get_model_size_mb([enc_int8, dec_int8])
    avg_int8 = run_onnx(images, proc, enc_int8, dec_int8, bos_id, eos_id, max_length)

    print(f"{'Pipeline':<15}{'Size(MB)':<12}{'Avg Latency(ms)':>18}")
    print("----------------------------------------------")
    print(f"{'PyTorch':<15}{size_pt:>12.2f}{avg_pt:>18.2f}")
    print(f"{'ONNX FP32':<15}{size_fp32:>12.2f}{avg_fp32:>18.2f}")
    print(f"{'ONNX INT8':<15}{size_int8:>12.2f}{avg_int8:>18.2f}")

    for name, size_mb, avg in [("ONNX FP32", size_fp32, avg_fp32), ("ONNX INT8", size_int8, avg_int8)]:
        speedup = (avg_pt - avg) / avg_pt * 100.0
        size_red = (size_pt - size_mb) / size_pt * 100.0
        print(f"{name}: speedup {speedup:+.1f}% | size reduction {size_red:+.1f}%")


if __name__ == "__main__":
    main()

