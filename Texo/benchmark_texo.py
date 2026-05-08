"""
Texo: Normal vs INT8 Quantized benchmark comparison.
Run: python benchmark_texo.py
"""

import time
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from texo.data.processor import EvalMERImageProcessor

MODEL_PATH = "./model"
IMAGE_DIR = "./TechnoSelection/test_img"


def load_model(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def quantize_model(model):
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )


def get_model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)


def inference(model, image_path, tokenizer, device, warmup=False):
    image = Image.open(image_path)
    image_processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    processed_image = image_processor(image).unsqueeze(0).to(device)

    if warmup:
        with torch.no_grad():
            model.generate(pixel_values=processed_image)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(pixel_values=processed_image)
    elapsed = (time.perf_counter() - start) * 1000

    pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return pred_str, elapsed


def run_benchmark(model, tokenizer, images, device, label):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Model size: {get_model_size_mb(model):.1f} MB | Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"{'='*65}")
    print(f"{'Image':<25} {'Latency':>12}  {'Preview'}")
    print(f"{'-'*65}")

    model.to(device)
    model.eval()

    # warmup on first image
    first_img = os.path.join(IMAGE_DIR, images[0])
    inference(model, first_img, tokenizer, device, warmup=True)

    total_ms = 0
    results = []
    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        pred, ms = inference(model, img_path, tokenizer, device)
        total_ms += ms
        results.append((img_name, ms, pred))
        preview = pred[:50] + "..." if len(pred) > 50 else pred
        print(f"{img_name:<25} {ms:>10.1f}ms  {preview}")

    avg = total_ms / len(images)
    print(f"{'-'*65}")
    print(f"{'Average':<25} {avg:>10.1f}ms")
    return avg, get_model_size_mb(model), results


def main():
    device = torch.device('cpu')  # quantization works best on CPU
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer = load_model(MODEL_PATH)

    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]
    images.sort()

    # --- Normal model ---
    avg_normal, size_normal, results_normal = run_benchmark(
        model, tokenizer, images, device, "NORMAL (FP32)"
    )

    # --- Quantized model ---
    print("\nQuantizing model...")
    model_quantized = load_model(MODEL_PATH)[0]  # fresh load
    model_quantized = quantize_model(model_quantized)

    avg_quant, size_quant, results_quant = run_benchmark(
        model_quantized, tokenizer, images, device, "QUANTIZED (INT8)"
    )

    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"{'Metric':<30} {'Normal':>12} {'INT8':>12} {'Speedup':>10}")
    print(f"{'-'*65}")
    print(f"{'Model size (MB)':<30} {size_normal:>12.1f} {size_quant:>12.1f} {size_normal/size_quant:>9.1f}x")
    print(f"{'Avg latency (ms)':<30} {avg_normal:>12.1f} {avg_quant:>12.1f} {avg_normal/avg_quant:>9.1f}x")
    print(f"{'Parameters':<30} {'20.0M':>12} {'20.0M':>12} {'same':>10}")
    print(f"{'='*65}")
    print(f"\nSize reduction: {((size_normal - size_quant) / size_normal * 100):.1f}%")
    print(f"Speed improvement: {((avg_normal - avg_quant) / avg_normal * 100):.1f}%")


if __name__ == "__main__":
    main()