"""Render → conditioning sketch (edges, silhouette, hatching, glyphs).

Returns a raw render-sized BGR canvas so the sketch is pixel-aligned with the
beauty render and PBR maps. The decorated text-band variant has been retired
along with the legacy ``dataset/conditioning/`` output.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


from cloth_pipeline.sketch.constants import SKETCH_BGR, USE_TEXTURE_STROKES
from cloth_pipeline.sketch.drawing import (
    albedo_pattern_stroke_mask,
    draw_depth_layer_boundary,
    draw_occlusion_edges,
    draw_shade_marks,
    draw_wobbly_contour,
)
from cloth_pipeline.sketch.edges import detect_edges
from cloth_pipeline.sketch.features import find_feature_points
from cloth_pipeline.sketch.segmentation import get_object_mask


def generate_sketch(
    render_path:    str,
    *,
    alpha_mask_path: str | None = None,
    albedo_map_path: str | None = None,
    albedo_tiling:   tuple[float, float] | None = None,
    pattern_name:    str | None = None,
) -> np.ndarray:
    """
    Render → render-sized BGR sketch canvas (no text band, no rescaling).

    The returned canvas matches ``render_path``'s width/height exactly, so the
    sketch is pixel-aligned with the beauty render and the PBR maps written by
    Stage 1 to ``outputs/<mesh>/view_<idx>/``.
    """
    # Load as BGRA so we can extract the alpha channel saved by generate_dataset.
    # cv2.IMREAD_UNCHANGED preserves all 4 channels when they exist.
    img_raw = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Could not read: {render_path}")

    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        img_bgr = img_raw[:, :, :3]          # drop alpha for colour processing
        alpha   = img_raw[:, :, 3]           # uint8 [0-255], 255 = cloth pixel
    else:
        img_bgr = img_raw                    # legacy RGB-only render
        alpha   = None

    # Prefer dedicated Stage-1 mask when present (independent of render alpha).
    if alpha_mask_path and os.path.isfile(alpha_mask_path):
        alpha_mask = cv2.imread(alpha_mask_path, cv2.IMREAD_GRAYSCALE)
        if alpha_mask is not None and alpha_mask.shape[:2] == img_bgr.shape[:2]:
            alpha = alpha_mask

    h, w = img_bgr.shape[:2]

    _sample_dir = os.path.dirname(render_path)
    _norm_base  = os.path.join(_sample_dir, "normals")
    normal_path = (
        _norm_base + ".png"
        if os.path.exists(_norm_base + ".png")
        else _norm_base + ".npy"
    )
    depth_path = os.path.join(_sample_dir, "depth.npy")

    # White canvas (BGR)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # ── B: Segmentation mask  (needed before A so we can mask edges) ─────────
    # Pass alpha so the mask is built from renderer coverage, not brightness.
    # This ensures shadowed cloth areas are never excluded from the silhouette.
    seg_mask = get_object_mask(img_bgr, alpha=alpha)

    # ── A: Edge detection (interior structural lines) ────────────────────────
    # Pass normal_path so the detector can use the geometry-only normal-map
    # gradient when the .npy file exists, skipping fine micro-fibre noise in RGB.
    edges = detect_edges(img_bgr, seg_mask=seg_mask, normal_path=normal_path)
    # Keep structural edges only on the eroded interior so they don't stack
    # with the wobbly silhouette into a fat outer edge.
    _k = np.ones((5, 5), np.uint8)
    interior_for_edges = cv2.erode(seg_mask, _k, iterations=1)
    edges = cv2.bitwise_and(edges, interior_for_edges)
    canvas[edges > 0] = SKETCH_BGR

    # ── A−: Optional albedo pattern strokes ───────────────────────────────────
    # Disabled by default to avoid tiled/grid artifacts in conditioning sketches.
    if USE_TEXTURE_STROKES and albedo_map_path and os.path.isfile(albedo_map_path):
        tex_bgr = cv2.imread(albedo_map_path, cv2.IMREAD_COLOR)
        if tex_bgr is not None:
            tu, tv = 4.0, 4.0
            if albedo_tiling is not None and len(albedo_tiling) >= 2:
                tu = float(albedo_tiling[0])
                tv = float(albedo_tiling[1])
            albedo_pat = albedo_pattern_stroke_mask(
                tex_bgr, h, w, tu, tv, interior_for_edges, pattern_name
            )
            canvas[albedo_pat > 0] = SKETCH_BGR

    # ── A+: Depth-aware occlusion emphasis (regional threshold) ──────────────
    canvas = draw_occlusion_edges(canvas, seg_mask, depth_path, SKETCH_BGR, thickness=1)

    # ── A++: Depth layer boundary — marks the separation between overlapping
    # cloth tails in the lower half even when the mask merges them ────────────
    canvas = draw_depth_layer_boundary(canvas, seg_mask, depth_path, SKETCH_BGR, thickness=1)

    # ── A+++: Wobbly silhouette — outer boundary + inner hole boundaries ──────
    # RETR_TREE returns both outer contours and inner holes (gaps between tails).
    # Drawing hole boundaries explicitly shows where cloth ends are disconnected.
    all_cnts, hierarchy = cv2.findContours(
        seg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    if all_cnts and hierarchy is not None:
        hier = hierarchy[0]
        # Outer contour (no parent)
        outer_candidates = [all_cnts[i] for i in range(len(all_cnts)) if hier[i][3] == -1]
        if outer_candidates:
            draw_wobbly_contour(
                canvas,
                max(outer_candidates, key=cv2.contourArea),
                SKETCH_BGR, base_thickness=2.0, wobble_amp=0.0, wobble_freq=2,
            )
        # Inner hole contours (have a parent) — gaps between separate cloth parts
        for i, cnt in enumerate(all_cnts):
            if hier[i][3] != -1 and cv2.contourArea(cnt) > 150:
                draw_wobbly_contour(
                    canvas, cnt, SKETCH_BGR,
                    base_thickness=1.0, wobble_amp=0.0, wobble_freq=2,
                )

    features_pts = find_feature_points(img_bgr, seg_mask)
    draw_shade_marks(canvas, features_pts, SKETCH_BGR)

    return canvas
