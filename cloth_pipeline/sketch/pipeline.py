"""Render → conditioning sketch (edges, silhouette, hatching, labels)."""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cloth_pipeline.paths import DATASET_DIR
from cloth_pipeline.sketch.constants import SKETCH_BGR, SKETCH_RGB
from cloth_pipeline.sketch.drawing import (
    draw_annotations,
    draw_wobbly_contour,
    draw_wool_texture,
)
from cloth_pipeline.sketch.edges import detect_edges
from cloth_pipeline.sketch.features import detect_dominant_color, find_feature_points
from cloth_pipeline.sketch.segmentation import get_object_mask
from cloth_pipeline.sketch.shadows import compute_shadow_mask, draw_hatching

def generate_sketch(
    render_path:  str,
    obj_name:     str,
    texture_type: str,
    keyword:      str,
) -> np.ndarray:
    """
    Full four-step pipeline.
    Returns a BGR uint8 image ready for cv2.imwrite.
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

    h, w = img_bgr.shape[:2]

    # Derive companion normal-map path: try PNG first, then NPY fallback.
    _base_name  = os.path.basename(render_path)
    _frame_str  = os.path.splitext(_base_name)[0].split("_")[-1]
    _norm_base  = os.path.join(DATASET_DIR, "normals", f"normals_{_frame_str}")
    normal_path = (
        _norm_base + ".png"
        if os.path.exists(_norm_base + ".png")
        else _norm_base + ".npy"
    )

    # White canvas (BGR)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # ── B: Segmentation mask  (needed before A so we can mask edges) ─────────
    # Pass alpha so the mask is built from renderer coverage, not brightness.
    # This ensures shadowed cloth areas are never excluded from the silhouette.
    seg_mask = get_object_mask(img_bgr, alpha=alpha)

    # ── A: Edge detection (interior structural lines) ────────────────────────
    # Pass normal_path so the detector can use the geometry-only normal-map
    # gradient when the .npy file exists, skipping wool micro-texture noise.
    edges = detect_edges(img_bgr, seg_mask=seg_mask, normal_path=normal_path)
    # HED / normal / Canny all fire strongly on the object rim; those pixels stack
    # with the wobbly silhouette and read as a fat outer edge.  Keep structural
    # edges only on the eroded interior so the outline is a single thin stroke.
    _k = np.ones((5, 5), np.uint8)
    interior_for_edges = cv2.erode(seg_mask, _k, iterations=1)
    edges = cv2.bitwise_and(edges, interior_for_edges)
    canvas[edges > 0] = SKETCH_BGR

    # ── A+: Wobbly outer silhouette from segmentation mask contour ───────────
    # CHAIN_APPROX_NONE → full boundary; draw_wobbly_contour resamples to one
    # continuous polyline. base_thickness=1.5 → 1 px core + half-weight outer ring.
    outer_cnts, _ = cv2.findContours(
        seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if outer_cnts:
        draw_wobbly_contour(
            canvas,
            max(outer_cnts, key=cv2.contourArea),
            SKETCH_BGR, base_thickness=1.5, wobble_amp=1.0, wobble_freq=2,
        )

    # ── C: Shadow hatching — single 45° direction only ────────────────────────
    shadow_mask = compute_shadow_mask(img_bgr, seg_mask, percentile=18.0)

    # Keep only the LARGEST connected shadow region so scattered fringe gaps
    # don't pollute the hatching with noise across the bottom.
    n_comp, lbl_comp, stats_comp, _ = cv2.connectedComponentsWithStats(shadow_mask)
    if n_comp > 1:
        biggest    = 1 + int(np.argmax(stats_comp[1:, cv2.CC_STAT_AREA]))
        shadow_mask = ((lbl_comp == biggest).astype(np.uint8) * 255)

    # Override shadow centroid now (before any mask manipulation below).
    shad_ys, shad_xs = np.where(shadow_mask > 0)

    # Exclude the highlight circle from hatching so hatching and highlight
    # occupy distinct semantic regions (Issue 3).
    features_pre  = find_feature_points(img_bgr, seg_mask)
    h_pt_pre      = features_pre.get("highlight")
    if h_pt_pre is not None:
        excl = np.zeros_like(shadow_mask)
        cv2.circle(excl, h_pt_pre, 24, 255, -1)   # 24 px = circle radius + margin
        shadow_mask = cv2.bitwise_and(shadow_mask, cv2.bitwise_not(excl))

    canvas = draw_hatching(canvas, shadow_mask, SKETCH_BGR,
                           spacing=22, angle_deg=45.0, thickness=1)

    # ── Colour + feature point detection for text / arrows ───────────────────
    dominant_color = detect_dominant_color(img_bgr, seg_mask)
    features       = find_feature_points(img_bgr, seg_mask)

    # Override shadow centroid with the filtered/hatched zone centroid.
    if len(shad_ys) > 0:
        features["shadow"] = (int(np.mean(shad_xs)), int(np.mean(shad_ys)))

    # ── Mid-tone wool texture stippling (dots + squiggles) ───────────────────
    gray_img      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    obj_pixels    = gray_img[seg_mask > 0]
    if obj_pixels.size > 0:
        lo_pct = float(np.percentile(obj_pixels, 35))
        hi_pct = float(np.percentile(obj_pixels, 70))
        mid_mask = (
            (gray_img >= lo_pct) & (gray_img <= hi_pct) & (seg_mask > 0)
        ).astype(np.uint8) * 255
    else:
        mid_mask = np.zeros_like(seg_mask)

    # Convert to PIL early so we can draw vector annotations on the same surface.
    pil  = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Wool texture marks drawn directly on the PIL canvas before labels.
    draw_wool_texture(pil, mid_mask, SKETCH_RGB, density=0.006)

    # ── D: Annotation ─────────────────────────────────────────────────────────
    tex_label  = (texture_type
                  .replace(" texture", "").replace(" Texture", "")
                  .replace("Coarse ", "").replace("coarse ", "")
                  .strip())
    short_name = obj_name.replace("Wool ", "").replace("wool ", "")
    text_lines = [
        f"Text: {short_name} ({dominant_color})",
        f"Texture: {tex_label}",
        f"Key word: {keyword}",
    ]

    draw_annotations(
        draw, text_lines, features,
        canvas_wh=(w, h),
        color=SKETCH_RGB,
    )
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Soften aliasing; sigma between “hairline” and original heavy bleed.
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.42)
    return canvas
