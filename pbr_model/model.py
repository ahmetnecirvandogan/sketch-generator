"""Sketch + prompt → PBR maps / render model scaffold (issue #20).

A small U-Net-style encoder/decoder with text injection at the bottleneck.
Two variants matching the data loader (#19):

- **Variant A** (primary): three output heads
    - ``albedo``       — 3-channel image (B, 3, 512, 512) in [0, 1]
    - ``roughness``    — 1-channel image (B, 1, 512, 512) in [0, 1]
    - ``lighting_sh``  — 9 floats        (B, 9), unconstrained

- **Variant B** (ablation): one output head
    - ``render``       — 3-channel image (B, 3, 512, 512) in [0, 1]

Variant selectable via the ``variant`` arg, matching the data loader's flag.

This file contains **architecture only** — no optimizer, no training loop, no
loss. Per #20's spec: "Forward pass runs on a fake batch of size 2 without
crashing; output shapes match expected targets."

Text encoder note: the included ``TextEncoder`` is a *stub* (hash → embedding
lookup → linear projection). Swap with CLIP / HuggingFace transformers in
real training. The architecture upstream of the text encoder doesn't change.

CLI smoke test::

    python -m pbr_model.model --variant a --batch-size 2
    python -m pbr_model.model --variant b --batch-size 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """Conv → BN → ReLU twice. The basic U-Net residual unit."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool 2× downsample → DoubleConv."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Transposed-conv 2× upsample, concat skip, DoubleConv."""

    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c // 2 + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if skip's spatial dims are slightly different (handles odd input sizes).
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Text encoder (stub — replace with CLIP / HuggingFace in real training)
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Toy text encoder: hash each prompt → vocab id → learned embedding → linear.

    Deterministic enough for scaffold smoke tests; *not* what real training will use.
    Swap with a real text encoder (CLIP, T5, frozen BERT, etc.) — only requirement
    is output shape ``(B, embed_dim)``.
    """

    def __init__(self, embed_dim: int = 128, vocab_size: int = 10_000) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, prompts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        # 4-token toy hash so two different prompts produce different ids.
        ids = torch.tensor(
            [
                [hash(p[i::4]) % self.vocab_size for i in range(4)]
                for p in prompts
            ],
            device=device,
            dtype=torch.long,
        )
        embs = self.emb(ids).mean(dim=1)  # (B, D)
        return F.relu(self.proj(embs))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class PBRModel(nn.Module):
    """Sketch + prompt → PBR maps (Variant A) or combined render (Variant B).

    Args:
        variant: 'a' or 'b' — selects which output heads are built.
        base_channels: U-Net width at the top resolution (32 in the scaffold).
        text_dim: dimensionality of the text encoder's output features.
    """

    def __init__(
        self,
        variant: Literal["a", "b"] = "a",
        base_channels: int = 32,
        text_dim: int = 128,
    ) -> None:
        super().__init__()
        if variant not in ("a", "b"):
            raise ValueError(f"variant must be 'a' or 'b', got {variant!r}")
        self.variant: Literal["a", "b"] = variant

        c = base_channels
        # --- Encoder ---
        self.in_conv = DoubleConv(3, c)        # input: sketch (B, 3, H, W)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)       # bottleneck (B, 16c, H/16, W/16)

        # --- Text injection at the bottleneck ---
        self.text_enc = TextEncoder(embed_dim=text_dim)
        self.text_proj = nn.Linear(text_dim, c * 16)

        # --- Decoder ---
        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)

        # --- Heads ---
        if variant == "a":
            self.head_albedo = nn.Conv2d(c, 3, kernel_size=1)
            self.head_roughness = nn.Conv2d(c, 1, kernel_size=1)
            self.head_lighting = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c * 16, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 9),
            )
        else:  # variant b
            self.head_render = nn.Conv2d(c, 3, kernel_size=1)

    def forward(self, sketch: torch.Tensor, prompt: list[str]) -> dict:
        if sketch.dim() != 4 or sketch.size(1) != 3:
            raise ValueError(f"sketch must be (B, 3, H, W), got {tuple(sketch.shape)}")
        if not isinstance(prompt, list) or len(prompt) != sketch.size(0):
            raise ValueError(
                f"prompt must be list[str] of len {sketch.size(0)} (matching batch), "
                f"got {type(prompt).__name__} len={len(prompt) if isinstance(prompt, list) else 'n/a'}"
            )

        # Encoder
        x0 = self.in_conv(sketch)              # (B,   c, H,    W)
        x1 = self.down1(x0)                    # (B,  2c, H/2,  W/2)
        x2 = self.down2(x1)                    # (B,  4c, H/4,  W/4)
        x3 = self.down3(x2)                    # (B,  8c, H/8,  W/8)
        x4 = self.down4(x3)                    # (B, 16c, H/16, W/16)  — bottleneck

        # Inject text at bottleneck (per-sample channel-wise bias)
        text_feat = self.text_enc(prompt)      # (B, text_dim)
        text_feat = self.text_proj(text_feat)  # (B, 16c)
        x4 = x4 + text_feat[:, :, None, None]  # broadcast across spatial

        # Decoder with skips
        d3 = self.up1(x4, x3)                  # (B, 8c, H/8, W/8)
        d2 = self.up2(d3, x2)                  # (B, 4c, H/4, W/4)
        d1 = self.up3(d2, x1)                  # (B, 2c, H/2, W/2)
        d0 = self.up4(d1, x0)                  # (B,  c, H,   W)

        # Heads
        out: dict = {}
        if self.variant == "a":
            out["albedo"] = torch.sigmoid(self.head_albedo(d0))      # (B, 3, H, W)
            out["roughness"] = torch.sigmoid(self.head_roughness(d0))  # (B, 1, H, W)
            out["lighting_sh"] = self.head_lighting(x4)              # (B, 9)
        else:
            out["render"] = torch.sigmoid(self.head_render(d0))      # (B, 3, H, W)
        return out


def make_model(
    variant: Literal["a", "b"] = "a",
    base_channels: int = 32,
    text_dim: int = 128,
) -> PBRModel:
    """Factory for ``PBRModel``. Mirrors ``pbr_model.dataset.make_dataloader``."""
    return PBRModel(variant=variant, base_channels=base_channels, text_dim=text_dim)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    model = make_model(variant=args.variant, base_channels=args.base_channels)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PBRModel(variant={args.variant!r}, base_channels={args.base_channels}) | {n_params:,} params")

    sketch = torch.randn(args.batch_size, 3, args.size, args.size)
    prompts = [f"cloth sample {i}" for i in range(args.batch_size)]

    model.eval()
    with torch.no_grad():
        outputs = model(sketch, prompts)

    print(f"forward(sketch={tuple(sketch.shape)}, prompt=list[str]×{args.batch_size}):")
    for k, v in outputs.items():
        print(f"  {k:>12}: {tuple(v.shape)} {v.dtype}  range=[{v.min().item():.3f}, {v.max().item():.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=512, help="square input H=W")
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()
    _smoke_test(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
