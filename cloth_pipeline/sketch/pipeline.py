"""Render → conditioning sketch (edges, silhouette, hatching, labels)."""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cloth_pipeline.paths import DATASET_DIR
from cloth_pipeline.sketch.constants import SKETCH_BGR, SKETCH_RGB
from cloth_pipeline.sketch.drawing import (
    albedo_pattern_stroke_mask,
    draw_annotations,
    draw_wobbly_contour,
    measure_top_left_text_pad,
)
from cloth_pipeline.sketch.edges import detect_edges
from cloth_pipeline.sketch.features import detect_dominant_color, find_feature_points
from cloth_pipeline.sketch.segmentation import get_object_mask
from cloth_pipeline.sketch.shadows import (
    compute_shadow_mask,
    draw_hatching,
    shadow_mask_darkest_fraction,
)

def generate_sketch(
    render_path:    str,
    obj_name:       str,
    material_label: str,
    keyword:        str,
    *,
    albedo_map_path: str | None = None,
    albedo_tiling:   tuple[float, float] | None = None,
    pattern_name:    str | None = None,
) -> np.ndarray:
    """
    Full render → conditioning sketch pipeline.

    When ``albedo_map_path`` / ``albedo_tiling`` / ``pattern_name`` are set (from
    dataset metadata), the same procedural albedo PNG used for Mitsuba ``base_color``
    is tiled and converted to sparse in-mask strokes so pattern is separate from
    lighting-dependent structure in the RGB render.

    Fabric preset is written in the top-left text block (``material_label``), not
    as strokes on the cloth. The output canvas adds equal padding on opposite sides
    (right = left, bottom = top) so the ``w×h`` sketch stays **centred** like the
    source render, while the top-left quadrant stays clear for captions.

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
    # gradient when the .npy file exists, skipping fine micro-fibre noise in RGB.
    edges = detect_edges(img_bgr, seg_mask=seg_mask, normal_path=normal_path)
    # HED / normal / Canny all fire strongly on the object rim; those pixels stack
    # with the wobbly silhouette and read as a fat outer edge.  Keep structural
    # edges only on the eroded interior so the outline is a single thin stroke.
    _k = np.ones((5, 5), np.uint8)
    interior_for_edges = cv2.erode(seg_mask, _k, iterations=1)
    edges = cv2.bitwise_and(edges, interior_for_edges)
    canvas[edges > 0] = SKETCH_BGR

    # ── A−: Albedo pattern from the same bitmap used as Mitsuba base_color ─────
    # Tied to material / pattern identity; not view- or lighting-dependent.
    albedo_pat = None
    if albedo_map_path and os.path.isfile(albedo_map_path):
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

    # When luminance is almost flat (common on translucent / even-lit fabrics),
    # the percentile threshold ties across most pixels and morph close merges
    # into one giant region — hatching then reads as a solid fill.  Fall back
    # to a fixed darkest fraction of object pixels (same intent as ~18%).
    _obj_area = int(np.count_nonzero(seg_mask > 0))
    _sh_area  = int(np.count_nonzero(shadow_mask > 0))
    _max_frac = 0.44
    if _obj_area > 0 and _sh_area > _max_frac * _obj_area:
        shadow_mask = shadow_mask_darkest_fraction(img_bgr, seg_mask, 0.18)
        n_comp, lbl_comp, stats_comp, _ = cv2.connectedComponentsWithStats(
            shadow_mask
        )
        if n_comp > 1:
            biggest = 1 + int(np.argmax(stats_comp[1:, cv2.CC_STAT_AREA]))
            shadow_mask = ((lbl_comp == biggest).astype(np.uint8) * 255)
        _sh_area = int(np.count_nonzero(shadow_mask > 0))
        # If the ranked mask still exploded (ties after close), hard-cap by eroding.
        _erk = np.ones((3, 3), np.uint8)
        _cap = max(int(_obj_area * _max_frac), 1)
        while _sh_area > _cap:
            prev = _sh_area
            shadow_mask = cv2.erode(shadow_mask, _erk, iterations=1)
            _sh_area = int(np.count_nonzero(shadow_mask > 0))
            if _sh_area == prev or _sh_area == 0:
                break

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

    # ── D: Annotation + reserved top-left band ───────────────────────────────
    mat_display = (
        material_label.replace(" texture", "").replace(" Texture", "")
        .replace("Coarse ", "").replace("coarse ", "")
        .strip()
    )
    short_name = obj_name.replace("Wool ", "").replace("wool ", "")
    text_lines = [
        f"Material: {mat_display}",
        f"Object: {short_name} ({dominant_color})",
        f"Key word: {keyword}",
    ]
    pad_left, pad_top = measure_top_left_text_pad(text_lines)
    # Mirror padding so the render-sized sketch block stays canvas-centred
    # (pixel layout inside the block still matches the Mitsuba frame 1:1).
    pad_right = pad_left
    pad_bottom = pad_top
    out_w = pad_left + w + pad_right
    out_h = pad_top + h + pad_bottom
    full = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    full[pad_top : pad_top + h, pad_left : pad_left + w] = canvas

    def _shift(pt: tuple[int, int] | None) -> tuple[int, int] | None:
        if pt is None:
            return None
        return (int(pt[0] + pad_left), int(pt[1] + pad_top))

    features_draw = {
        "highlight": _shift(features.get("highlight")),
        "midtone": _shift(features.get("midtone")),
        "shadow": _shift(features.get("shadow")),
    }

    pil = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    draw_annotations(
        draw,
        text_lines,
        features_draw,
        canvas_wh=(out_w, out_h),
        color=SKETCH_RGB,
        sketch_content_top=pad_top,
    )
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Soften aliasing; sigma between “hairline” and original heavy bleed.
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.42)
    return canvas
