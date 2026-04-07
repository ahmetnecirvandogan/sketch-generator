"""Bezier Pivot-based Deformation (BPD) augmentation for conditioning sketches.

Implements the sketch data augmentation from DeepSketch2Wear (Chen et al., 2026)
as a smooth grid warp applied to the stroke layer of a conditioning image.
This simulates the variability of human freehand drawing while keeping the
text annotation block pixel-exact.

The paper's formulation:
  - Divide the sketch into 32×32 patches
  - Fit cubic Bezier curves to strokes in each patch
  - Apply displacement θ = floor(row / C) × K to control points
    (amplitude grows with row position, i.e. larger shifts further down)

Here we implement this at the image level via a sparse control-point grid
with bilinear interpolation — equivalent result without per-stroke Bezier fitting.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_bpd(
    img_bgr: np.ndarray,
    *,
    text_pad_top: int = 0,
    text_pad_left: int = 0,
    grid_spacing: int = 32,
    max_displacement: float = 5.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply BPD-style warp to a conditioning sketch image.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR uint8 conditioning image (white background, dark strokes).
    text_pad_top, text_pad_left : int
        Pixel dimensions of the reserved text annotation block (top-left).
        This region is excluded from warping and restored pixel-exact.
    grid_spacing : int
        Spacing between warp control points in pixels. The paper uses 32×32
        patches — matching this default keeps the deformation scale consistent.
    max_displacement : float
        Maximum warp amplitude (pixels) at the bottom of the image. Amplitude
        scales linearly from ~40% at the top to 100% at the bottom, matching
        the paper's row-proportional θ formulation.
    seed : int | None
        Optional RNG seed for reproducibility across augmentation variants.

    Returns
    -------
    np.ndarray
        Warped BGR uint8 image, same shape as input.
    """
    h, w = img_bgr.shape[:2]
    rng = np.random.default_rng(seed)

    # Build sparse grid of control points (paper's 32×32 patch grid).
    gy_pts = np.arange(0, h + grid_spacing, grid_spacing, dtype=np.float32)
    gx_pts = np.arange(0, w + grid_spacing, grid_spacing, dtype=np.float32)
    ny, nx = len(gy_pts), len(gx_pts)

    dx_grid = np.zeros((ny, nx), dtype=np.float32)
    dy_grid = np.zeros((ny, nx), dtype=np.float32)

    for iy, gy in enumerate(gy_pts):
        # Row-proportional amplitude: 40% at top → 160% at bottom.
        # Mirrors θ = floor(row/C) × K — more displacement lower in the image.
        row_scale = 0.4 + 1.2 * float(gy) / max(1.0, float(h))
        amp = max_displacement * row_scale
        for ix, gx in enumerate(gx_pts):
            # Leave the text annotation zone unperturbed.
            if float(gx) <= text_pad_left and float(gy) <= text_pad_top:
                continue
            dx_grid[iy, ix] = rng.uniform(-amp, amp)
            dy_grid[iy, ix] = rng.uniform(-amp, amp)

    # Interpolate displacement field to full resolution.
    dx_full = cv2.resize(dx_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy_grid, (w, h), interpolation=cv2.INTER_CUBIC)

    # Zero out displacement in the text block and blend the boundary smoothly.
    if text_pad_top > 0 or text_pad_left > 0:
        tp = min(text_pad_top, h)
        lp = min(text_pad_left, w)
        dx_full[:tp, :lp] = 0.0
        dy_full[:tp, :lp] = 0.0
        # Smooth transition to avoid hard seam at text-block edge.
        blend = min(16, h - tp, w - lp)
        if blend > 0:
            ramp = np.linspace(0.0, 1.0, blend, dtype=np.float32)
            dx_full[tp : tp + blend, :lp] *= ramp[:, np.newaxis]
            dy_full[tp : tp + blend, :lp] *= ramp[:, np.newaxis]
            dx_full[:tp, lp : lp + blend] *= ramp[np.newaxis, :]
            dy_full[:tp, lp : lp + blend] *= ramp[np.newaxis, :]

    # Build source-coordinate maps for cv2.remap.
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    src_x = np.clip(map_x + dx_full, 0.0, w - 1).astype(np.float32)
    src_y = np.clip(map_y + dy_full, 0.0, h - 1).astype(np.float32)

    warped = cv2.remap(
        img_bgr, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Restore text block pixel-exact (annotations must not be deformed).
    if text_pad_top > 0 and text_pad_left > 0:
        tp = min(text_pad_top, h)
        lp = min(text_pad_left, w)
        warped[:tp, :lp] = img_bgr[:tp, :lp]

    return warped
