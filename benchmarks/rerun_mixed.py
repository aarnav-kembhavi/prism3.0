"""Re-process only en_ch_mixed pages with CJK engine, then re-run eval."""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from benchmarks.run_omnidocbench import _run_prism_on_images, _write_eval_config, _run_evaluation

GT     = str(ROOT / 'data' / 'omnidocbench' / 'OmniDocBench_available.json')
IMAGES = str(ROOT / 'data' / 'omnidocbench' / 'images')
PRED   = str(ROOT / 'preds' / 'omnidocbench')

with open(GT, encoding='utf-8') as f:
    gt = json.load(f)

images_dir = Path(IMAGES)
image_paths, mixed_pages = [], set()
for page in gt:
    if page['page_info']['page_attribute'].get('language') != 'en_ch_mixed':
        continue
    img_name = page['page_info']['image_path']
    img_path = images_dir / img_name
    if img_path.exists():
        image_paths.append(str(img_path))
        mixed_pages.add(Path(img_name).stem)

print(f'[*] Re-processing {len(image_paths)} en_ch_mixed pages with dual EN+CJK engine...')
_run_prism_on_images(image_paths, PRED, mixed_pages=mixed_pages)

print('\n[*] Running eval...')
config_path = _write_eval_config(GT, PRED, no_cdm=True)
_run_evaluation(config_path)
