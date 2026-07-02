"""
Run PP-StructureV3 (paddleocr 3.x) on a list of images, CPU-only, emit
OmniDocBench-style markdown per page, and log RAM/latency via bench_metrics.

Usage:
  .venv_ppocr/Scripts/python run_ppstructure.py <images_dir> <gt_json> <out_pred_dir> [n_threads]
"""
import os, sys, json, time
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_metrics import MetricsTracker

def main():
    images_dir, gt_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    config = sys.argv[5] if len(sys.argv) > 5 else 'server'   # 'server' | 'mobile'
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.makedirs(out_dir, exist_ok=True)

    import paddle
    paddle.set_device('cpu')
    try:
        paddle.set_num_threads(n_threads)
    except Exception:
        pass
    from paddleocr import PPStructureV3

    with open(gt_json, encoding='utf-8') as f:
        gt = json.load(f)
    images = [p['page_info']['image_path'] for p in gt]

    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False)
    if config == 'mobile':
        # Lightweight variants: mobile OCR det/rec + small layout.
        kw.update(
            text_detection_model_name='PP-OCRv5_mobile_det',
            text_recognition_model_name='PP-OCRv5_mobile_rec',
            layout_detection_model_name='PP-DocLayout-S',
        )

    m = MetricsTracker(); m.start_sampler()
    t = m.mark_load_start()
    pipe = PPStructureV3(**kw)
    m.mark_load_end(t)
    print(f"[ppstructure] loaded in {m.load_time_s:.1f}s, cold RSS {m.cold_rss_mb:.0f}MB")

    ok = 0
    for i, img_name in enumerate(images):
        img_path = os.path.join(images_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        if not os.path.exists(img_path):
            print(f"  missing {img_name}"); continue
        try:
            with m.page_timer():
                results = list(pipe.predict(img_path))
            # concatenate markdown across result blocks
            md_parts = []
            for res in results:
                md = getattr(res, 'markdown', None)
                if isinstance(md, dict):
                    md_parts.append(md.get('markdown_texts', '') or md.get('markdown', ''))
                elif isinstance(md, str):
                    md_parts.append(md)
                else:
                    # fallback: try json/dict 'markdown' key
                    d = res if isinstance(res, dict) else getattr(res, 'json', {})
                    md_parts.append(str(d.get('markdown', '')) if isinstance(d, dict) else '')
            md_text = "\n\n".join(p for p in md_parts if p)
            with open(os.path.join(out_dir, f"{stem}.md"), 'w', encoding='utf-8') as f:
                f.write(md_text)
            ok += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            with open(os.path.join(out_dir, f"{stem}.md"), 'w', encoding='utf-8') as f:
                f.write('')
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(images)} done, peak RSS {m.peak_rss_mb:.0f}MB")

    m.stop_sampler()
    m.save(os.path.join(out_dir, '_metrics.json'), f'PP-StructureV3-{config}', ok)

if __name__ == '__main__':
    main()
