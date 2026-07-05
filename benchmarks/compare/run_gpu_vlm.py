"""Generic HF VLM runner for OmniDocBench competitor comparison.

Runs a document-parsing VLM on the 20-page subset, saves OmniDocBench-format
markdown predictions, and records RAM / latency / VRAM.

usage: python run_gpu_vlm.py <model_key> <pred_dir> [--device cuda|cpu]
model_key in: got2 | qwen25vl | olmocr | nougat | florence2 | dots

Run in .venv_gpu.  Eval separately via run_omnidocbench.py --eval-only.
"""
import os, sys, json, time, argparse, statistics, traceback
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('PYTHONUTF8', '1')

# 8 GB VRAM: cap vision tokens so doc images don't OOM the 7B-4bit models.
_MAX_PIXELS = 768 * 28 * 28   # ~602k px (~776px square-equivalent)
_MIN_PIXELS = 256 * 28 * 28

ROOT = r"C:\PROJECTS\s2l2\testprism"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "compare"))
os.chdir(ROOT)

from bench_metrics import MetricsTracker

SUBSET = "benchmarks/compare/compare20_subset.json"
IMAGES = "data/omnidocbench/images"


def load_subset():
    d = json.load(open(SUBSET, encoding='utf-8'))
    out = []
    for p in d:
        ip = os.path.join(IMAGES, p['page_info']['image_path'])
        if os.path.exists(ip):
            out.append((os.path.splitext(p['page_info']['image_path'])[0], ip))
    return out


# ─────────────────────────── model adapters ───────────────────────────
# Each returns (generate_fn, model_obj_for_vram) ; generate_fn(image_path)->md

def load_got2(device):
    import torch
    from transformers import AutoProcessor, GotOcr2ForConditionalGeneration
    mid = 'stepfun-ai/GOT-OCR-2.0-hf'
    proc = AutoProcessor.from_pretrained(mid)
    model = GotOcr2ForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device).eval()
    from PIL import Image
    def gen(img):
        image = Image.open(img).convert('RGB')
        inputs = proc(image, return_tensors='pt', format=True).to(device, torch.float16)
        gen_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False,
                                 tokenizer=proc.tokenizer, stop_strings='<|im_end|>')
        return proc.decode(gen_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return gen, model


def load_nougat(device):
    import torch
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    proc = NougatProcessor.from_pretrained('facebook/nougat-base')
    model = VisionEncoderDecoderModel.from_pretrained('facebook/nougat-base').to(device).eval()
    from PIL import Image
    def gen(img):
        image = Image.open(img).convert('RGB')
        px = proc(image, return_tensors='pt').pixel_values.to(device)
        out = model.generate(px, min_length=1, max_new_tokens=1536,
                             bad_words_ids=[[proc.tokenizer.unk_token_id]])
        seq = proc.batch_decode(out, skip_special_tokens=True)[0]
        return proc.post_process_generation(seq, fix_markdown=False)
    return gen, model


def load_qwen25vl(device):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen2.5-VL-7B-Instruct', quantization_config=bnb, device_map='cuda',
        torch_dtype=torch.float16)
    proc = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',
                                         min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
    PROMPT = ("Convert this document page to clean Markdown. Preserve reading order, "
              "headings, lists, and tables (as HTML). Render all mathematical formulas "
              "as LaTeX ($...$ inline, $$...$$ display). Output only the Markdown.")
    def gen(img):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(img)}",
             "min_pixels": _MIN_PIXELS, "max_pixels": _MAX_PIXELS},
            {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors='pt').to('cuda')
        gen_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        return proc.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    return gen, model


def load_olmocr(device):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        'allenai/olmOCR-7B-0225-preview', quantization_config=bnb, device_map='cuda',
        torch_dtype=torch.float16)
    proc = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct',
                                         min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
    from PIL import Image
    PROMPT = ("Below is the image of one page of a document. Just return the plain text "
              "representation of this document as if you were reading it naturally, "
              "including Markdown for headings/lists and LaTeX for equations.")
    def gen(img):
        image = Image.open(img).convert('RGB')
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[image], return_tensors='pt').to('cuda')
        gen_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0]
    return gen, model


def load_florence2(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large',
        trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
    from PIL import Image
    def gen(img):
        image = Image.open(img).convert('RGB')
        task = '<OCR_WITH_REGION>'
        inputs = proc(text=task, images=image, return_tensors='pt').to(device, torch.float16)
        gen_ids = model.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'],
                                 max_new_tokens=1536, num_beams=3, do_sample=False)
        text = proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(text, task=task, image_size=(image.width, image.height))
        # OCR_WITH_REGION returns {'quad_boxes':..,'labels':..}; join labels in order
        v = parsed.get(task, {})
        if isinstance(v, dict) and 'labels' in v:
            return "\n".join(l.strip() for l in v['labels'])
        return str(v)
    return gen, model


def load_dots(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    mid = 'rednote-hilab/dots.ocr'
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map='cuda',
        trust_remote_code=True, attn_implementation='sdpa').eval()
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True,
                                         min_pixels=_MIN_PIXELS, max_pixels=_MAX_PIXELS)
    from PIL import Image
    # dots.ocr OCR prompt: full-page text+layout as Markdown
    PROMPT = ("Please output the text content from the image as clean Markdown, "
              "preserving reading order, tables (HTML) and formulas (LaTeX).")
    def gen(img):
        image = Image.open(img).convert('RGB')
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[image], return_tensors='pt').to('cuda')
        gen_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0]
    return gen, model


LOADERS = {'got2': load_got2, 'nougat': load_nougat, 'qwen25vl': load_qwen25vl,
           'olmocr': load_olmocr, 'florence2': load_florence2, 'dots': load_dots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_key')
    ap.add_argument('pred_dir')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    import torch
    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    os.makedirs(args.pred_dir, exist_ok=True)
    pages = load_subset()
    print(f"[{args.model_key}] {len(pages)} pages, device={device}")

    m = MetricsTracker(); m.start_sampler()
    t0 = m.mark_load_start()
    gen, model_obj = LOADERS[args.model_key](device)
    m.mark_load_end(t0)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for stem, ip in pages:
        try:
            with m.page_timer():
                md = gen(ip)
        except Exception as e:
            print(f"  ERROR {stem}: {e}")
            traceback.print_exc()
            md = ''
        open(os.path.join(args.pred_dir, f"{stem}.md"), 'w', encoding='utf-8').write(md or '')
        print(f"  {stem}: {len(md or '')} chars, {m.page_latencies[-1]:.1f}s")
        if device == 'cuda':
            import torch as _t; _t.cuda.empty_cache()

    m.stop_sampler()
    vram = None
    if device == 'cuda':
        vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    s = m.summary(args.model_key, len(pages))
    s['device'] = device
    s['peak_vram_mb'] = vram
    json.dump(s, open(os.path.join(args.pred_dir, "_efficiency.json"), 'w'), indent=2)
    print(f"[{args.model_key}] DONE peak_rss={s['peak_rss_mb']}MB vram={vram}MB "
          f"lat_median={s['latency_median_s']}s")


if __name__ == '__main__':
    main()
