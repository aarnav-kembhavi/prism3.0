"""
Run any docling-supported end-to-end VLM (SmolDocling / GraniteDocling /
GOT-OCR2.0) on a list of images, CPU-only, emit markdown, log RAM/latency.

Usage:
  .venv_smol/Scripts/python run_docling_vlm.py <images_dir> <gt_json> <out_dir> <spec> [n_threads]
  spec in: smoldocling | granitedocling | got2
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_metrics import MetricsTracker


def main():
    images_dir, gt_json, out_dir, spec_name = sys.argv[1:5]
    n_threads = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.makedirs(out_dir, exist_ok=True)

    import torch
    torch.set_num_threads(n_threads)

    from docling.document_converter import DocumentConverter, ImageFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        VlmPipelineOptions, smoldocling_vlm_conversion_options,
    )
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel import vlm_model_specs as vms

    SPECS = {
        'smoldocling': smoldocling_vlm_conversion_options,
        'granitedocling': getattr(vms, 'GRANITEDOCLING_TRANSFORMERS', None),
        'got2': getattr(vms, 'GOT2_TRANSFORMERS', None),
    }
    label = {'smoldocling': 'SmolDocling-256M', 'granitedocling': 'GraniteDocling-258M',
             'got2': 'GOT-OCR2.0'}[spec_name]
    spec = SPECS[spec_name]
    if spec is None:
        print(f"spec {spec_name} not available in this docling build"); sys.exit(2)

    with open(gt_json, encoding='utf-8') as f:
        gt = json.load(f)
    images = [p['page_info']['image_path'] for p in gt]

    m = MetricsTracker(); m.start_sampler()
    t = m.mark_load_start()
    opts = VlmPipelineOptions(vlm_options=spec)
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=n_threads)
    converter = DocumentConverter(format_options={
        InputFormat.IMAGE: ImageFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)
    })
    m.mark_load_end(t)
    print(f"[{spec_name}] ready in {m.load_time_s:.1f}s, cold RSS {m.cold_rss_mb:.0f}MB")

    ok = 0
    for i, img_name in enumerate(images):
        img_path = os.path.join(images_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        if not os.path.exists(img_path):
            print(f"  missing {img_name}"); continue
        try:
            with m.page_timer():
                result = converter.convert(img_path)
                md_text = result.document.export_to_markdown()
            with open(os.path.join(out_dir, f"{stem}.md"), 'w', encoding='utf-8') as f:
                f.write(md_text)
            ok += 1
        except Exception:
            import traceback; traceback.print_exc()
            with open(os.path.join(out_dir, f"{stem}.md"), 'w', encoding='utf-8') as f:
                f.write('')
        print(f"  {i+1}/{len(images)} done ({m.page_latencies[-1]:.1f}s), peak RSS {m.peak_rss_mb:.0f}MB")

    m.stop_sampler()
    m.save(os.path.join(out_dir, '_metrics.json'), label, ok)


if __name__ == '__main__':
    main()
