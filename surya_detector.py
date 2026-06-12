"""
surya_detector.py
-----------------
Standalone EfficientViT text-line detector, ported from datalab-to/surya.
No surya package required — weights loaded directly from vikp/surya_det3 on HF.

Architecture: EfficientVitLarge (depths=[1,1,1,6,6], widths=[32,64,128,256,512])
+ SegFormer-style MLP fuse head → 2-channel sigmoid heatmap.

Module names match the checkpoint exactly so no key remapping is needed.

Usage:
    det = SuryaLineDetector()          # downloads ~130 MB on first run
    boxes = det.detect(pil_image)      # list of [x0,y0,x1,y1] ints
    crops = det.line_crops(pil_image)  # list of PIL crops
"""

from __future__ import annotations

import os
from functools import partial
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SemanticSegmenterOutput

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────

class EfficientViTConfig(PretrainedConfig):
    model_type = "efficientvit_surya"

    def __init__(
        self,
        num_channels: int = 3,
        widths=(32, 64, 128, 256, 512),
        head_dim: int = 32,
        num_stages: int = 4,
        depths=(1, 1, 1, 6, 6),
        strides=(2, 2, 2, 2, 2),
        num_labels: int = 2,
        classifier_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-6,
        decoder_layer_hidden_size: int = 128,
        decoder_hidden_size: int = 512,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.widths = list(widths)
        self.head_dim = head_dim
        self.num_stages = num_stages
        self.depths = list(depths)
        self.strides = list(strides)
        self.num_labels = num_labels
        self.classifier_dropout_prob = classifier_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.decoder_layer_hidden_size = decoder_layer_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.initializer_range = initializer_range


# ── Building blocks ───────────────────────────────────────────────────────────

def _pad(k, s=1, d=1):
    return ((s - 1) + d * (k - 1)) // 2


class ConvNormAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, d=1, g=1, bias=False,
                 norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, _pad(k, s, d), d, g, bias=bias)
        self.norm = norm(out_c) if norm else nn.Identity()
        self.act  = act(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    """Two-layer conv block (used in Stem with block_type='large')."""
    def __init__(self, in_c, out_c, k=3, s=1, fewer_norm=False):
        super().__init__()
        BN = nn.BatchNorm2d
        if fewer_norm:
            self.conv1 = ConvNormAct(in_c, out_c, k, s, norm=None, act=nn.ReLU6, bias=True)
            self.conv2 = ConvNormAct(out_c, out_c, k, 1, norm=BN, act=None)
        else:
            self.conv1 = ConvNormAct(in_c, out_c, k, s, norm=BN, act=nn.ReLU6)
            self.conv2 = ConvNormAct(out_c, out_c, k, 1, norm=BN, act=None)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, er=6, fewer_norm=False):
        super().__init__()
        mid = round(in_c * er)
        BN  = nn.BatchNorm2d
        if fewer_norm:
            self.inverted_conv = ConvNormAct(in_c, mid, 1, norm=None, act=nn.ReLU6, bias=True)
            self.depth_conv    = ConvNormAct(mid, mid, k, s, g=mid, norm=None, act=nn.ReLU6, bias=True)
            self.point_conv    = ConvNormAct(mid, out_c, 1, norm=BN, act=None)
        else:
            self.inverted_conv = ConvNormAct(in_c, mid, 1, norm=BN, act=nn.ReLU6)
            self.depth_conv    = ConvNormAct(mid, mid, k, s, g=mid, norm=BN, act=nn.ReLU6)
            self.point_conv    = ConvNormAct(mid, out_c, 1, norm=BN, act=None)

    def forward(self, x):
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))


class FusedMBConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, er=6, fewer_norm=False):
        super().__init__()
        mid = round(in_c * er)
        BN  = nn.BatchNorm2d
        if fewer_norm:
            self.spatial_conv = ConvNormAct(in_c, mid, k, s, norm=None, act=nn.ReLU6, bias=True)
            self.point_conv   = ConvNormAct(mid, out_c, 1, norm=BN, act=None)
        else:
            self.spatial_conv = ConvNormAct(in_c, mid, k, s, norm=BN, act=nn.ReLU6)
            self.point_conv   = ConvNormAct(mid, out_c, 1, norm=BN, act=None)

    def forward(self, x):
        return self.point_conv(self.spatial_conv(x))


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention."""
    def __init__(self, in_c, out_c, head_dim=32, scales=(5,), eps=1e-5):
        super().__init__()
        self.eps = eps
        heads = in_c // head_dim
        total = heads * head_dim
        self.dim = head_dim

        self.qkv = ConvNormAct(in_c, 3 * total, 1, norm=None, act=None, bias=False)
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3 * total, 3 * total, sc,
                          padding=sc // 2, groups=3 * total, bias=False),
                nn.Conv2d(3 * total, 3 * total, 1, groups=3 * heads, bias=False),
            ) for sc in scales
        ])
        self.kernel_func = nn.ReLU(inplace=False)
        self.proj = ConvNormAct(total * (1 + len(scales)), out_c, 1,
                                norm=nn.BatchNorm2d, act=None, bias=False)

    def _attn(self, q, k, v):
        dtype = v.dtype
        q, k, v = q.float(), k.float(), v.float()
        kv  = k.transpose(-1, -2) @ v
        out = q @ kv
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        return out.to(dtype)

    def forward(self, x):
        B, _, H, W = x.shape
        qkv = self.qkv(x)
        ms  = [qkv] + [op(qkv) for op in self.aggreg]
        ms  = torch.cat(ms, 1).reshape(B, -1, 3 * self.dim, H * W).transpose(-1, -2)
        q, k, v = ms.chunk(3, -1)
        q, k = self.kernel_func(q), self.kernel_func(k)
        v    = F.pad(v, (0, 1), value=1.0)
        out  = self._attn(q, k, v)
        return self.proj(out.transpose(-1, -2).reshape(B, -1, H, W))


class ResidualBlock(nn.Module):
    def __init__(self, main, shortcut=None):
        super().__init__()
        self.main     = main
        self.shortcut = shortcut

    def forward(self, x):
        out = self.main(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class EfficientVitBlock(nn.Module):
    def __init__(self, c, head_dim=32, er=4, norm=nn.BatchNorm2d, act=nn.Hardswish):
        super().__init__()
        self.context_module = ResidualBlock(LiteMLA(c, c, head_dim), nn.Identity())
        self.local_module   = ResidualBlock(
            MBConv(c, c, er=er, fewer_norm=True), nn.Identity())

    def forward(self, x):
        return self.local_module(self.context_module(x))


# ── Stem ──────────────────────────────────────────────────────────────────────

class Stem(nn.Sequential):
    def __init__(self, in_c, out_c, depth, stride, norm, act):
        super().__init__()
        # First conv: downsampling
        self.add_module('in_conv', ConvNormAct(in_c, out_c, stride + 1, stride,
                                               norm=norm, act=act))
        # Residual blocks (expand_ratio=1, block_type='large' → ConvBlock)
        for i in range(depth):
            self.add_module(f'res{i}', ResidualBlock(
                ConvBlock(out_c, out_c, 3, 1), nn.Identity()))


# ── Encoder stages ────────────────────────────────────────────────────────────

class LargeStage(nn.Module):
    def __init__(self, in_c, out_c, depth, stride, norm, act, head_dim,
                 vit_stage=False, fewer_norm=False):
        super().__init__()
        er = 24 if vit_stage else 16
        fn = vit_stage or fewer_norm  # fewer_norm for this stage's down block

        # Down-sampling block (no skip connection)
        if fn:
            down = ResidualBlock(MBConv(in_c, out_c, stride + 1, stride, er=er,
                                        fewer_norm=True), None)
        else:
            down = ResidualBlock(FusedMBConv(in_c, out_c, stride + 1, stride, er=er,
                                             fewer_norm=False), None)

        blocks = [down]

        if vit_stage:
            for _ in range(depth):
                blocks.append(EfficientVitBlock(out_c, head_dim, er=6, norm=norm, act=act))
        else:
            for _ in range(depth):
                if fewer_norm:
                    inner = ResidualBlock(
                        MBConv(out_c, out_c, 3, 1, er=4, fewer_norm=True),
                        nn.Identity())
                else:
                    inner = ResidualBlock(
                        FusedMBConv(out_c, out_c, 3, 1, er=4, fewer_norm=False),
                        nn.Identity())
                blocks.append(inner)

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ── Full encoder ──────────────────────────────────────────────────────────────

class EfficientVitLarge(nn.Module):
    def __init__(self, cfg: EfficientViTConfig):
        super().__init__()
        norm = partial(nn.BatchNorm2d, eps=cfg.layer_norm_eps)
        act  = nn.Hardswish
        self.stem = Stem(cfg.num_channels, cfg.widths[0], cfg.depths[0],
                         cfg.strides[0], norm, act)
        self.stages = nn.ModuleList()
        in_c = cfg.widths[0]
        for i, (w, d, s) in enumerate(zip(cfg.widths[1:], cfg.depths[1:], cfg.strides[1:])):
            self.stages.append(LargeStage(
                in_c, w, d, s, norm, act, cfg.head_dim,
                vit_stage=(i >= 3), fewer_norm=(i >= 2)))
            in_c = w

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


# ── Decode head ───────────────────────────────────────────────────────────────

class DecodeMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class DecodeHead(nn.Module):
    def __init__(self, cfg: EfficientViTConfig):
        super().__init__()
        hdim = cfg.decoder_layer_hidden_size
        self.linear_c    = nn.ModuleList([DecodeMLP(w, hdim) for w in cfg.widths[1:]])
        self.linear_fuse = nn.Conv2d(hdim * cfg.num_stages, cfg.decoder_hidden_size, 1, bias=False)
        self.batch_norm  = nn.BatchNorm2d(cfg.decoder_hidden_size)
        self.activation  = nn.ReLU()
        self.dropout     = nn.Dropout(cfg.classifier_dropout_prob)
        self.classifier  = nn.Conv2d(cfg.decoder_hidden_size, cfg.num_labels, 1)

    def forward(self, feats):
        B  = feats[-1].shape[0]
        h0, w0 = feats[0].shape[2], feats[0].shape[3]
        outs = []
        for feat, mlp in zip(feats, self.linear_c):
            H, W = feat.shape[2], feat.shape[3]
            x = mlp(feat).permute(0, 2, 1).reshape(B, -1, H, W)
            x = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=False)
            outs.append(x)
        x = self.linear_fuse(torch.cat(outs[::-1], 1))
        x = self.activation(self.batch_norm(x))
        return self.classifier(self.dropout(x))


# ── Full segmentation model ───────────────────────────────────────────────────

class EfficientViTForSemanticSegmentation(PreTrainedModel):
    config_class = EfficientViTConfig

    def __init__(self, cfg: EfficientViTConfig):
        super().__init__(cfg)
        self.vit         = EfficientVitLarge(cfg)
        self.decode_head = DecodeHead(cfg)

    def forward(self, pixel_values):
        feats  = self.vit(pixel_values)
        logits = self.decode_head(feats)
        return SemanticSegmenterOutput(logits=torch.sigmoid(logits))


# ── Postprocessing (adapted from surya/detection/heatmap.py) ──────────────────

_TEXT_THRESH  = 0.6
_BLANK_THRESH = 0.35
_Y_EXPAND     = 0.05


def _dynamic_thresh(linemap, text_t, blank_t, ref=0.7):
    flat   = linemap.ravel()
    idx    = int(len(flat) * 0.9)
    top10  = float(np.mean(np.partition(flat, idx)[idx:]))
    scale  = float(np.clip(top10 / ref, 0.0, 1.0) ** 0.5)
    return (float(np.clip(text_t * scale, 0.15, 0.8)),
            float(np.clip(blank_t * scale, 0.1,  0.6)))


def _heatmap_to_boxes(linemap: np.ndarray) -> List[List[int]]:
    text_t, blank_t = _dynamic_thresh(linemap, _TEXT_THRESH, _BLANK_THRESH)
    H, W = linemap.shape
    mask  = (linemap > blank_t).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    boxes = []
    for k in range(1, n_lab):
        if stats[k, cv2.CC_STAT_AREA] < 10:
            continue
        x = stats[k, cv2.CC_STAT_LEFT]; y = stats[k, cv2.CC_STAT_TOP]
        w = stats[k, cv2.CC_STAT_WIDTH]; h = stats[k, cv2.CC_STAT_HEIGHT]
        roi = linemap[y:y+h, x:x+w]
        seg_vals = roi[labels[y:y+h, x:x+w] == k]
        if seg_vals.size == 0 or float(seg_vals.max()) < text_t:
            continue
        nit = max(1, int(np.sqrt(min(w, h))))
        sx, sy = max(0, x - nit - 1), max(0, y - nit - 1)
        ex, ey = min(W, x + w + nit + 1), min(H, y + h + nit + 1)
        seg = (labels[sy:ey, sx:ex] == k).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (nit + 1, nit + 1))
        seg = cv2.dilate(seg, kernel)
        ys, xs = np.nonzero(seg)
        xs += sx; ys += sy
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


# ── Public API ────────────────────────────────────────────────────────────────

_MODEL_HF_ID   = 'vikp/surya_det3'
_PROC_SIZE     = 1200
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class SuryaLineDetector:
    """EfficientViT full-page text-line detector, weights from vikp/surya_det3."""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._model: Optional[EfficientViTForSemanticSegmentation] = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg = EfficientViTConfig()
        weights_path = hf_hub_download(_MODEL_HF_ID, 'model.safetensors')
        state = load_file(weights_path)

        model = EfficientViTForSemanticSegmentation(cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'[SuryaDetector] {len(missing)} missing keys — first 5: {missing[:5]}')
        if unexpected:
            print(f'[SuryaDetector] {len(unexpected)} unexpected keys — first 5: {unexpected[:5]}')

        model.eval()
        self._model = model.to(self.device)
        print(f'[SuryaDetector] Loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) on {self.device}')

    def _preprocess(self, img: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        orig_size = img.size  # (W, H)
        img = img.convert('RGB')
        img.thumbnail((_PROC_SIZE, _PROC_SIZE), Image.LANCZOS)
        img = img.resize((_PROC_SIZE, _PROC_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0), orig_size

    def detect(self, img: Image.Image) -> List[List[int]]:
        """Return [[x0,y0,x1,y1], ...] in original image coordinates."""
        self._ensure_loaded()
        tensor, (ow, oh) = self._preprocess(img)
        with torch.no_grad():
            out = self._model(tensor.to(self.device))
        heatmap = out.logits[0, 0].cpu().numpy()  # channel 0 = text heatmap

        # Interpolate to proc size if model output is smaller
        if heatmap.shape != (_PROC_SIZE, _PROC_SIZE):
            t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, (_PROC_SIZE, _PROC_SIZE), mode='bilinear', align_corners=False)
            heatmap = t.squeeze().numpy()

        raw = _heatmap_to_boxes(heatmap)
        sx, sy = ow / _PROC_SIZE, oh / _PROC_SIZE
        result = []
        for x0, y0, x1, y1 in raw:
            rx0, ry0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
            rx1, ry1 = min(ow, int(x1 * sx)), min(oh, int(y1 * sy))
            exp = int((ry1 - ry0) * _Y_EXPAND)
            ry0, ry1 = max(0, ry0 - exp), min(oh, ry1 + exp)
            if rx1 > rx0 and ry1 > ry0:
                result.append([rx0, ry0, rx1, ry1])
        return result

    def line_crops(self, img: Image.Image) -> List[Image.Image]:
        return [img.crop(b) for b in self.detect(img)]
