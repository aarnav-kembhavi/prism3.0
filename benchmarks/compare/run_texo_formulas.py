"""Run Texo on the extracted formula crops, save predictions + latency."""
import os, sys, json, time, glob
sys.path.insert(0, r"C:\PROJECTS\s2l2\testprism")
os.chdir(r"C:\PROJECTS\s2l2\testprism")
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
import pipeline.math_worker_onnx as mw

OUT = "benchmarks/compare/formula_eval"
tok = Tokenizer.from_file("Texo/model/onnx/../tokenizer.json") if os.path.exists("Texo/model/tokenizer.json") else Tokenizer.from_file("Texo/model/tokenizer.json")
so = ort.SessionOptions(); so.enable_cpu_mem_arena = False
enc = ort.InferenceSession("Texo/model/onnx/encoder_model.onnx", so, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession("Texo/model/onnx/decoder_model_merged.onnx", so, providers=["CPUExecutionProvider"])

crops = sorted(glob.glob(os.path.join(OUT, "crops", "*.png")))
preds = {}
lat = []
for cp in crops:
    cid = os.path.splitext(os.path.basename(cp))[0]
    img = Image.open(cp).convert("RGB")
    t = time.perf_counter()
    px = mw._preprocess_to_tensor(img)
    ids = mw._onnx_decode(dec, mw._onnx_encode(enc, px), tok,
                          max_new_tokens=mw._MAX_NEW_TOKENS, rep_penalty=1.15)
    latex = mw._sanitize(tok.decode(ids).strip())
    lat.append(time.perf_counter() - t)
    preds[cid] = latex

import statistics
json.dump(preds, open(os.path.join(OUT, "pred_texo.json"), "w", encoding='utf-8'), ensure_ascii=False)
print(f"Texo: {len(preds)} crops, median {statistics.median(lat)*1000:.0f}ms/crop, mean {statistics.mean(lat)*1000:.0f}ms")
