import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from safetensors.torch import load_file

from texo.data.processor import EvalMERImageProcessor, TextProcessor
from texo.model.formulanet import FormulaNet

import benchmark_clean as bench

logging.getLogger().setLevel(logging.ERROR)
ort.set_default_logger_severity(3)


class Texo:
    """Minimal Texo-like API for CPU-only generation (no HF model wrappers)."""

    def __init__(self, model_dir: Path, max_length: int) -> None:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        cfg.pop("pretrained", None)
        self.model = FormulaNet(cfg).to("cpu").eval()
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
        self.max_length = int(max_length)

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.generate(pixel_values=pixel_values, num_beams=1, do_sample=False, max_length=self.max_length)


def truncate(s: str, n: int = 100) -> str:
    s = s.replace("\n", " ").replace("\r", " ")
    return s[:n] + ("..." if len(s) > n else "")


def console_safe(s: str) -> str:
    # Avoid Windows console encoding errors (cp1252) while keeping comparisons unchanged.
    return s.encode("ascii", errors="backslashreplace").decode("ascii")


def load_onnx_session(path: Path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def ensure_int8_models(onnx_dir: Path) -> tuple[Path, Path]:
    enc_src = onnx_dir / "encoder_model.onnx"
    dec_src = onnx_dir / "decoder_model.onnx"
    enc_dst = onnx_dir / "encoder_int8.onnx"
    dec_dst = onnx_dir / "decoder_int8.onnx"
    if not enc_dst.exists():
        quantize_dynamic(str(enc_src), str(enc_dst), weight_type=QuantType.QInt8)
    if not dec_dst.exists():
        quantize_dynamic(str(dec_src), str(dec_dst), weight_type=QuantType.QInt8)
    return enc_dst, dec_dst


def decode_ids(tokenizer, ids: np.ndarray | torch.Tensor) -> str:
    # batch_decode accepts numpy int arrays / list-of-lists.
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def tokenize_latex(s: str) -> list[str]:
    import re
    # Split LaTeX into tokens (commands + symbols + words)
    tokens = re.findall(r'\\[a-zA-Z]+|[{}_^]|[0-9]+|[a-zA-Z]+|.', s)
    return [t for t in tokens if t.strip()]


def token_overlap(a: str, b: str) -> float:
    ta = tokenize_latex(a)
    tb = tokenize_latex(b)
    if not ta or not tb:
        return 0.0
    set_a, set_b = set(ta), set(tb)
    return len(set_a & set_b) / len(set_a | set_b)


def main() -> None:
    base = Path(__file__).resolve().parent
    image_dir = base / "TechnoSelection/test_img"
    model_dir = base / "model"
    onnx_dir = model_dir / "onnx"

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if len(images) == 0:
        raise SystemExit(f"No images found in {image_dir}")

    proc = EvalMERImageProcessor(image_size={"width": bench.DEFAULT_IMAGE_SIZE, "height": bench.DEFAULT_IMAGE_SIZE})
    bos_id, eos_id, max_length = bench.load_generation_params(model_dir)

    tokenizer = TextProcessor(
        config={"tokenizer_path": str(base / "data/tokenizer"), "tokenizer_config": {"add_special_tokens": True, "max_length": 1024}}
    ).tokenizer

    pt = Texo(model_dir, max_length=max_length)

    enc_fp32 = load_onnx_session(onnx_dir / "encoder_model.onnx")
    dec_fp32 = load_onnx_session(onnx_dir / "decoder_model.onnx")

    enc_int8_path, dec_int8_path = ensure_int8_models(onnx_dir)
    enc_int8 = load_onnx_session(enc_int8_path)
    dec_int8 = load_onnx_session(dec_int8_path)

    fp32_scores = []
    int8_scores = []

    for img_path in images:
        img_name = img_path.name
        px_pt = bench.preprocess_image(img_path, proc)  # torch [1,3,H,W]
        ids_pt = pt.generate(px_pt)
        out_pt = decode_ids(tokenizer, ids_pt)

        px_np = px_pt.numpy()
        ids_fp32 = bench.greedy_decode_onnx(enc_fp32, dec_fp32, px_np, bos_id, eos_id, max_length)
        out_fp32 = decode_ids(tokenizer, ids_fp32)

        ids_int8 = bench.greedy_decode_onnx(enc_int8, dec_int8, px_np, bos_id, eos_id, max_length)
        out_int8 = decode_ids(tokenizer, ids_int8)

        print(console_safe(img_name))
        print(f"PyTorch: {console_safe(truncate(out_pt))}")
        print(f"ONNX FP32: {console_safe(truncate(out_fp32))}")
        print(f"ONNX INT8: {console_safe(truncate(out_int8))}")

        overlap_fp32 = token_overlap(out_pt, out_fp32)
        overlap_int8 = token_overlap(out_pt, out_int8)
        fp32_scores.append(overlap_fp32)
        int8_scores.append(overlap_int8)

        print(f"Overlap FP32: {overlap_fp32:.2f}")
        print(f"Overlap INT8: {overlap_int8:.2f}")
        print("")

    n = len(images)
    print(f"Avg overlap (PT vs ONNX FP32): {sum(fp32_scores) / n:.2f}")
    print(f"Avg overlap (PT vs ONNX INT8): {sum(int8_scores) / n:.2f}")


if __name__ == "__main__":
    main()

