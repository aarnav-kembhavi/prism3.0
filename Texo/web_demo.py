"""
Local web UI: upload a formula / equation image, get Texo LaTeX for Overleaf.

Usage (from the Texo directory, after `uv sync` or your usual env setup):
  pip install gradio
  python web_demo.py

Open http://127.0.0.1:7860 in your browser. Copy the "LaTeX line" or the full
minimal document into Overleaf (or wrap the line in $...$ / \\[...\\] yourself).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

import texo.utils.config  # noqa: F401 — registers encoder type `my_hgnetv2`
from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet


def _load_texo(model_dir: Path, texo_root: Path) -> tuple[FormulaNet, PreTrainedTokenizerFast, int]:
    """
    Texo checkpoints use FormulaNet + custom HGNet encoder; HF `from_pretrained` cannot load them.
    Weights: prefer `model.safetensors` in model_dir, else `pretrained` .pt path in config.json
    (resolved relative to the Texo project root).
    """
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg_init = dict(cfg)
    st_path = model_dir / "model.safetensors"

    if st_path.is_file():
        cfg_init["pretrained"] = ""
    else:
        pt = (cfg_init.get("pretrained") or "").strip()
        if pt:
            p = Path(pt)
            if not p.is_absolute():
                p = (texo_root / p).resolve()
            else:
                p = p.resolve()
            cfg_init["pretrained"] = str(p) if p.is_file() else ""

    model = FormulaNet(cfg_init)
    if st_path.is_file():
        model.load_state_dict(load_file(str(st_path)), strict=True)

    if not st_path.is_file() and not (cfg_init.get("pretrained") or "").strip():
        raise SystemExit(
            f"No weights found under {model_dir}.\n"
            "  - Add model.safetensors next to config.json, or\n"
            "  - Set config.json 'pretrained' to an existing .pt checkpoint "
            f"(path relative to {texo_root})."
        )

    tok_path = model_dir if (model_dir / "tokenizer.json").is_file() else texo_root / "data" / "tokenizer"
    if not (Path(tok_path) / "tokenizer.json").is_file():
        raise SystemExit(f"Tokenizer not found at {model_dir} or {texo_root / 'data' / 'tokenizer'}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))

    gen_path = model_dir / "generation_config.json"
    max_len = 1024
    if gen_path.is_file():
        max_len = int(json.loads(gen_path.read_text(encoding="utf-8")).get("max_length", max_len))

    return model, tokenizer, max_len


def _pil_rgb(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    import numpy as np

    return Image.fromarray(np.asarray(img)).convert("RGB")


def _overleaf_snippet(latex_line: str) -> str:
    """Minimal compilable document wrapping display math."""
    body = latex_line.strip()
    return (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\begin{document}\n\n"
        "\\[\n"
        f"{body}\n"
        "\\]\n\n"
        "\\end{document}\n"
    )


def build_app(model_dir: Path, texo_root: Path, device: torch.device):
    print(f"Loading model from {model_dir} ...")
    model, tokenizer, max_length = _load_texo(model_dir, texo_root)
    model.to(device)
    model.eval()

    proc = EvalMERImageProcessor(image_size={"width": 384, "height": 384})

    def predict(img):
        pil = _pil_rgb(img)
        if pil is None:
            return "", "", "Upload an image first."
        with torch.inference_mode():
            x = proc(pil).unsqueeze(0).to(device)
            out = model.generate(
                pixel_values=x,
                num_beams=1,
                do_sample=False,
                max_length=max_length,
            )
            line = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        snippet = _overleaf_snippet(line)
        hint = "Paste the snippet into a new Overleaf project, or put the LaTeX line inside $...$ or \\[...\\] in your own preamble."
        return line, snippet, hint

    with gr.Blocks(title="Texo MER") as demo:
        gr.Markdown(
            "## Texo: image → LaTeX\n"
            "Upload a **cropped** formula or equation image (PNG/JPG). "
            "Copy the output into [Overleaf](https://www.overleaf.com/) or any LaTeX editor."
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Image", sources=["upload", "clipboard"])
        btn = gr.Button("Run Texo", variant="primary")
        line_out = gr.Textbox(label="LaTeX line (model output)", lines=4)
        doc_out = gr.Textbox(label="Minimal Overleaf-ready .tex (display math)", lines=16)
        hint_out = gr.Textbox(label="Note", lines=1)
        btn.click(fn=predict, inputs=[inp], outputs=[line_out, doc_out, hint_out])

    return demo


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Texo Gradio web demo")
    ap.add_argument("--model-dir", type=Path, default=here / "model", help="Folder with config.json (+ weights)")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a temporary public gradio.link (optional)")
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    demo = build_app(args.model_dir.resolve(), here, device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
