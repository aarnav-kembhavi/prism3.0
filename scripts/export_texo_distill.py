"""
Export Texo-distill ONNX model for use in PRISM.

Run once from the repo root after cloning:
    python scripts/export_texo_distill.py

Requires: torch, transformers, optimum, huggingface_hub, tokenizers
(all already present in the PRISM environment)

What this does:
  1. Downloads formulanet_distill_best.pt from alephpi/FormulaNet on HF
  2. Loads it into the FormulaNet architecture with vocab_size=1264
  3. Exports encoder+decoder to ONNX (Texo/model/onnx/)
  4. Installs the matching distill tokenizer (Texo/model/tokenizer.json)
  5. Updates Texo/model/config.json for vocab_size=1264
"""

import sys
import os
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Texo' / 'src'))

import torch
from transformers import VisionEncoderDecoderConfig
from huggingface_hub import hf_hub_download
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import ViTOnnxConfig
from texo.model.formulanet import FormulaNet

MODEL_DIR = ROOT / 'Texo' / 'model'
ONNX_DIR  = MODEL_DIR / 'onnx'
DISTILL_TOK = ROOT / 'Texo' / 'data' / 'unimernet_tokenizer_distill'
TMP_HF = ROOT / 'Texo' / 'model' / '_distill_hf_tmp'

ONNX_DIR.mkdir(parents=True, exist_ok=True)
TMP_HF.mkdir(parents=True, exist_ok=True)


def register_onnx_config():
    register = TasksManager.create_register('onnx')

    @register('my_hgnetv2', *['feature-extraction'])
    class HGNetv2OnnxConfig(ViTOnnxConfig):
        @property
        def inputs(self):
            return {'pixel_values': {0: 'batch_size'}}


def main():
    print('[1/5] Downloading formulanet_distill_best.pt from HF...')
    ckpt_path = hf_hub_download(
        'alephpi/FormulaNet',
        'checkpoints/formulanet_distill_best.pt',
        local_dir=str(TMP_HF / 'hf_download'),
    )
    print(f'      Downloaded to {ckpt_path}')

    print('[2/5] Loading model with vocab_size=1264...')
    cfg = json.loads((MODEL_DIR / 'config.json').read_bytes().decode('utf-8'))
    cfg['decoder']['vocab_size'] = 1264
    cfg['pretrained'] = ''
    config = VisionEncoderDecoderConfig.from_dict(cfg)
    model = FormulaNet(config)
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, f'State dict mismatch: {missing=} {unexpected=}'
    model.eval()

    print('[3/5] Saving as HF model format...')
    model.save_pretrained(str(TMP_HF))
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        shutil.copy(str(DISTILL_TOK / fname), str(TMP_HF / fname))
    shutil.copy(str(MODEL_DIR / 'generation_config.json'), str(TMP_HF))

    print('[4/5] Exporting to ONNX...')
    register_onnx_config()
    main_export(
        str(TMP_HF),
        task='image-to-text-with-past',
        output=ONNX_DIR,
    )

    print('[5/5] Installing tokenizer and config...')
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        shutil.copy(str(DISTILL_TOK / fname), str(MODEL_DIR / fname))
        shutil.copy(str(DISTILL_TOK / fname), str(ONNX_DIR / fname))
    cfg['pretrained'] = ''
    (MODEL_DIR / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    shutil.rmtree(str(TMP_HF), ignore_errors=True)

    # Verify
    from tokenizers import Tokenizer as FastTokenizer
    tok = FastTokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
    assert tok.get_vocab_size() == 1264, f'Unexpected vocab size: {tok.get_vocab_size()}'
    enc_size = (ONNX_DIR / 'encoder_model.onnx').stat().st_size / 1e6
    dec_size = (ONNX_DIR / 'decoder_model_merged.onnx').stat().st_size / 1e6
    print(f'\nDone. encoder={enc_size:.1f} MB  decoder={dec_size:.1f} MB  vocab={tok.get_vocab_size()}')
    print('Texo-distill ONNX is ready.')


if __name__ == '__main__':
    main()
