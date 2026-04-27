"""Render → conditioning sketch (edges, silhouette, hatching, labels)."""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cloth_pipeline.paths import DATASET_DIR
from cloth_pipeline.sketch.constants import SKETCH_BGR, SKETCH_RGB, USE_TEXTURE_STROKES
from cloth_pipeline.sketch.drawing import (
    albedo_pattern_stroke_mask,
    draw_annotations,
    draw_depth_layer_boundary,
    draw_occlusion_edges,
    draw_shade_marks,
    draw_wobbly_contour,
    measure_top_left_text_pad,
)
from cloth_pipeline.sketch.edges import detect_edges
from cloth_pipeline.sketch.features import detect_dominant_color, find_feature_points
from cloth_pipeline.sketch.segmentation import get_object_mask


def _nonwhite_bbox(img_bgr: np.ndarray, threshold: int = 250) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < threshold)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _place_scaled_content(
    content_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
) -> np.ndarray:
    """
    Crop to the inked sketch region, scale it up moderately, and place it in
    the drawable area that remains after reserving the top-left text block.
    """
    full = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    bbox = _nonwhite_bbox(content_bgr)
    if bbox is None:
        full[pad_top : pad_top + content_bgr.shape[0], pad_left : pad_left + content_bgr.shape[1]] = content_bgr
        return full

    x0, y0, x1, y1 = bbox
    crop = content_bgr[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = crop.shape[:2]

    avail_x0 = max(18, pad_left + 12)
    avail_y0 = max(18, pad_top + 10)
    avail_x1 = max(avail_x0 + 1, out_w - pad_right + max(18, pad_right // 3))
    avail_y1 = max(avail_y0 + 1, out_h - pad_bottom + max(18, pad_bottom // 3))
    avail_w = max(1, avail_x1 - avail_x0)
    avail_h = max(1, avail_y1 - avail_y0)

    scale = min(avail_w / float(max(1, cw)), avail_h / float(max(1, ch)), 1.45)
    if scale <= 0:
        scale = 1.0
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    placed = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    off_x = avail_x0 + max(0, (avail_w - new_w) // 2)
    off_y = avail_y0 + max(0, (avail_h - new_h) // 2)
    region = full[off_y : off_y + new_h, off_x : off_x + new_w]
    gray = cv2.cvtColor(placed, cv2.COLOR_BGR2GRAY)
    ink = gray < 250
    region[ink] = placed[ink]
    return full

def generate_sketch(
    render_path:    str,
    obj_name:       str,
    material_label: str,
    texture_label:  str,
    include_text:   bool = True,
    *,
    alpha_mask_path: str | None = None,
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

    # Prefer dedicated Stage-1 mask when present (independent of render alpha).
    if alpha_mask_path and os.path.isfile(alpha_mask_path):
        alpha_mask = cv2.imread(alpha_mask_path, cv2.IMREAD_GRAYSCALE)
        if alpha_mask is not None and alpha_mask.shape[:2] == img_bgr.shape[:2]:
            alpha = alpha_mask

    h, w = img_bgr.shape[:2]

    # Derive companion data paths from frame string.
    _base_name  = os.path.basename(render_path)
    _frame_str  = os.path.splitext(_base_name)[0].split("_")[-1]
    _norm_base  = os.path.join(DATASET_DIR, "normals", f"normals_{_frame_str}")
    normal_path = (
        _norm_base + ".png"
        if os.path.exists(_norm_base + ".png")
        else _norm_base + ".npy"
    )
    depth_path = os.path.join(DATASET_DIR, "depth", f"depth_{_frame_str}.npy")

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

    # For conditioning-only mode, keep output exactly render-aligned:
    # no annotation text, no extra canvas, no scaling/repositioning.
    if not include_text:
        return canvas

    # ── Colour detection for text ─────────────────────────────────────────────
    dominant_color = detect_dominant_color(img_bgr, seg_mask)

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
        f"Texture: {texture_label}",
    ]
    pad_left, pad_top = measure_top_left_text_pad(text_lines)
    # Keep a generous text band at top-left, but trim the opposite margins so
    # the scarf takes up more of the page than in earlier versions.
    pad_right = max(44, int(round(pad_left * 0.42)))
    pad_bottom = max(40, int(round(pad_top * 0.35)))
    out_w = pad_left + w + pad_right
    out_h = pad_top + h + pad_bottom
    full = _place_scaled_content(
        canvas,
        out_w,
        out_h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )

    pil = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    draw_annotations(
        draw,
        text_lines,
        {},
        canvas_wh=(out_w, out_h),
        color=SKETCH_RGB,
        sketch_content_top=pad_top,
    )
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return canvas
