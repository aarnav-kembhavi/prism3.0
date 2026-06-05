import os
import sys
import torch
from PIL import Image

# Setup paths
ROOT_DIR = os.getcwd()
sys.path.append(os.path.join(ROOT_DIR, 'Texo', 'src'))

from texo.data.processor import EvalMERImageProcessor
from transformers import AutoTokenizer, VisionEncoderDecoderModel

MODEL_PATH = os.path.join(ROOT_DIR, "Texo", "model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[*] Testing Texo on {device}...")
print(f"[*] Model path: {MODEL_PATH}")

try:
    import texo.utils.config # Registers 'my_hgnetv2'
    from texo.model.formulanet import FormulaNet
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = FormulaNet.from_pretrained(MODEL_PATH)
    model.eval().to(device)
    processor = EvalMERImageProcessor(image_size={'width': 384, 'height': 384})
    
    # Create a dummy image or load one if exists
    img = Image.new('RGB', (384, 384), color='white')
    
    # Process individually and stack (matching models_interface logic)
    processed_list = [processor(img.convert("RGB"))]
    inputs = torch.stack(processed_list).to(device)
    
    with torch.no_grad():
        outputs = model.generate(pixel_values=inputs)
    
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"[*] Texo Result: '{result}'")

except Exception as e:
    print(f"[!] Texo Failure: {e}")
    import traceback
    traceback.print_exc()
