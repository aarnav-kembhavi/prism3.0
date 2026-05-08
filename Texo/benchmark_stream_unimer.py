"""
Stream UniMER-Test from Hugging Face and compare PyTorch vs ONNX FP32 (CPU).

Metrics: average latency (ms), peak working-set memory (MB), BLEU + normalized edit distance.

Run from repo root:
  python benchmark_stream_unimer.py --max-samples 50 --split spe
"""
from __future__ import annotations

import argparse
import ctypes
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from datasets import load_dataset
from PIL import Image
from safetensors.torch import load_file

# Configures torch threads + ONNX thread defaults (import side effects).
import benchmark_clean as bench

from texo.data.processor import EvalMERImageProcessor, TextProcessor
from texo.model.formulanet import FormulaNet
from texo.utils.scores import compute_bleu, compute_edit_distance

logging.getLogger().setLevel(logging.ERROR)
ort.set_default_logger_severity(3)


class TexoPT:
    def __init__(self, model_dir: Path, max_length: int) -> None:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        cfg.pop("pretrained", None)
        self.model = FormulaNet(cfg).to("cpu").eval()
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
        self.max_length = int(max_length)

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.generate(
            pixel_values=pixel_values, num_beams=1, do_sample=False, max_length=self.max_length
        )


def rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(counters), ctypes.sizeof(counters)):
                return counters.WorkingSetSize / (1024**2)
        except Exception:
            pass
    return None


def load_bos_eos_max_length(model_dir: Path, max_length_override: int) -> tuple[int, int, int]:
    if max_length_override > 0:
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        gen = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
        bos = int(gen.get("bos_token_id", cfg.get("decoder_start_token_id", 0)))
        eos = int(gen.get("eos_token_id", cfg.get("eos_token_id", 2)))
        return bos, eos, max_length_override
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    bos = int(gen.get("bos_token_id", cfg.get("decoder_start_token_id", 0)))
    eos = int(gen.get("eos_token_id", cfg.get("eos_token_id", 2)))
    max_len = int(gen.get("max_length", 1024))
    return bos, eos, max_len


def row_to_pil(row: dict) -> Image.Image:
    raw = row.get("image")
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(raw)}")


def preprocess_pil(img: Image.Image, proc: EvalMERImageProcessor) -> torch.Tensor:
    return proc(img).unsqueeze(0).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream UniMER-Test from HF; compare PT vs ONNX FP32.")
    ap.add_argument("--dataset", type=str, default="alephpi/UniMER-Test", help="HF dataset id")
    ap.add_argument("--split", type=str, default="spe", help="Split name (spe, cpe, sce, hwe)")
    ap.add_argument("--model-dir", type=str, default="model")
    ap.add_argument("--max-samples", type=int, default=100, help="Max streamed samples after warmup")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup samples (not counted in latency/accuracy)")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Decoder max tokens (0 = use generation_config.json max_length, e.g. 1024)",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    model_dir = (base / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    onnx_dir = model_dir / "onnx"

    proc = EvalMERImageProcessor(image_size={"width": args.image_size, "height": args.image_size})
    bos_id, eos_id, max_length = load_bos_eos_max_length(model_dir, args.max_length)

    tokenizer = TextProcessor(
        config={
            "tokenizer_path": str(base / "data" / "tokenizer"),
            "tokenizer_config": {"add_special_tokens": True, "max_length": 1024},
        }
    ).tokenizer

    print(f"Loading streaming dataset {args.dataset} split={args.split!r} ...")
    try:
        stream = load_dataset(args.dataset, split=args.split, streaming=True, trust_remote_code=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Try: huggingface-cli login  (if private) or check split name matches the Hub card.")
        raise SystemExit(1) from e

    num_threads = bench.num_threads
    print(f"ONNX intra_op_num_threads={num_threads}, inter_op_num_threads=1 (from benchmark_clean)")

    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_sess = ort.InferenceSession(
        str(onnx_dir / "encoder_model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        str(onnx_dir / "decoder_model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
    )

    pt_model = TexoPT(model_dir, max_length=max_length)

    mem_samples: list[float] = []
    m = rss_mb()
    if m is not None:
        mem_samples.append(m)

    it = iter(stream)
    # Warmup
    for _ in range(args.warmup):
        try:
            row = next(it)
        except StopIteration:
            raise SystemExit("Dataset iterator empty before warmup completed.")
        px = preprocess_pil(row_to_pil(row), proc)
        _ = pt_model.generate(px)
        _ = bench.greedy_decode_onnx(enc_sess, dec_sess, px.numpy(), bos_id, eos_id, max_length)
        m = rss_mb()
        if m is not None:
            mem_samples.append(m)

    lat_pt: list[float] = []
    lat_onnx: list[float] = []
    preds_pt: list[str] = []
    preds_onnx: list[str] = []
    refs: list[str] = []

    n = 0
    while n < args.max_samples:
        try:
            row = next(it)
        except StopIteration:
            break
        ref = row.get("text", "")
        if not isinstance(ref, str):
            ref = str(ref)

        px = preprocess_pil(row_to_pil(row), proc)

        t0 = time.perf_counter()
        ids_pt = pt_model.generate(px)
        lat_pt.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        ids_onnx = bench.greedy_decode_onnx(enc_sess, dec_sess, px.numpy(), bos_id, eos_id, max_length)
        lat_onnx.append((time.perf_counter() - t1) * 1000.0)

        pred_pt = tokenizer.batch_decode(ids_pt.detach().cpu().numpy(), skip_special_tokens=True)[0]
        pred_onnx = tokenizer.batch_decode(np.asarray(ids_onnx), skip_special_tokens=True)[0]

        preds_pt.append(pred_pt)
        preds_onnx.append(pred_onnx)
        refs.append(ref)

        m = rss_mb()
        if m is not None:
            mem_samples.append(m)

        n += 1

    if n == 0:
        raise SystemExit("No timed samples; increase data or lower --warmup.")

    peak_mem = max(mem_samples) if mem_samples else None
    avg_pt = float(np.mean(lat_pt))
    avg_onnx = float(np.mean(lat_onnx))

    bleu_pt = compute_bleu(preds_pt, refs)
    bleu_onnx = compute_bleu(preds_onnx, refs)
    ed_pt = compute_edit_distance(preds_pt, refs)
    ed_onnx = compute_edit_distance(preds_onnx, refs)
    exact_pt = sum(p == r for p, r in zip(preds_pt, refs)) / n * 100.0
    exact_onnx = sum(p == r for p, r in zip(preds_onnx, refs)) / n * 100.0

    print("")
    print(f"Samples (timed): {n}  |  max_length={max_length}  |  image_size={args.image_size}")
    print("")
    print(f"{'Backend':<12}{'Avg latency (ms)':>18}{'BLEU':>10}{'Norm edit dist':>16}{'Exact match %':>14}")
    print("-" * 72)
    print(f"{'PyTorch':<12}{avg_pt:>18.2f}{bleu_pt:>10.4f}{ed_pt:>16.4f}{exact_pt:>13.2f}%")
    print(f"{'ONNX FP32':<12}{avg_onnx:>18.2f}{bleu_onnx:>10.4f}{ed_onnx:>16.4f}{exact_onnx:>13.2f}%")
    print("")
    if peak_mem is not None:
        print(f"Peak observed working set (process): {peak_mem:.1f} MB")
        print("(Same process holds both PT and ONNX; peak is after loading both + streaming.)")
    else:
        print("Memory: could not read RSS (install psutil: pip install psutil for cross-platform RSS).")


if __name__ == "__main__":
    main()
