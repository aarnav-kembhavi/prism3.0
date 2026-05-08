"""
Texo inference + timing script.
Run: python run_texo.py
"""

import time
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, PreTrainedTokenizerFast
from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet

MODEL_PATH = "./model"
IMAGE_DIR = "./TechnoSelection/test_img"

def load(path):
    model = VisionEncoderDecoderModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def inference(model, image_path, tokenizer, device):
    model.to(device)
    image = Image.open(image_path)
    image_processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    processed_image = image_processor(image).unsqueeze(0)
    
    start = time.perf_counter()
    outputs = model.generate(pixel_values=processed_image.to(device))
    elapsed = time.perf_counter() - start
    
    pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return pred_str, elapsed * 1000  # ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)
    model.eval()

    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]
    
    print(f"\n{'='*60}")
    print(f"{'Image':<30} {'Latency (ms)':>15}")
    print(f"{'='*60}")
    
    total_time = 0
    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        pred, ms = inference(model, img_path, tokenizer, device)
        total_time += ms
        print(f"{img_name:<30} {ms:>15.1f} ms")
        print(f"  → {pred[:80]}{'...' if len(pred) > 80 else ''}\n")
    
    print(f"{'='*60}")
    print(f"Average latency: {total_time / len(images):.1f} ms")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

if __name__ == "__main__":
    main()