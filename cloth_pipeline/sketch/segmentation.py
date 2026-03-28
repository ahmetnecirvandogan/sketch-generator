"""SAM and threshold segmentation; dashed organic boundary helper."""

import math
import os

import cv2
import numpy as np
from PIL import ImageDraw

from cloth_pipeline.sketch.constants import SAM_CHECKPOINT, SAM_MODEL_TYPE

# ─────────────────────────────────────────────────────────────────────────────
# STEP B — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

_sam_predictor = None


def _load_sam():
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print(
            "[B] segment-anything not installed — using threshold segmentation.\n"
            "    (pip install git+https://github.com/facebookresearch/segment-anything.git)"
        )
        _sam_predictor = False
        return None
    if not os.path.exists(SAM_CHECKPOINT):
        print(
            f"[B] SAM checkpoint not found at '{SAM_CHECKPOINT}' — "
            "using threshold segmentation."
        )
        _sam_predictor = False
        return None
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    _sam_predictor = SamPredictor(sam)
    print("[B] SAM predictor loaded.")
    return _sam_predictor


def _threshold_mask(
    img_bgr: np.ndarray,
    alpha:   "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Returns a binary uint8 mask (255 = object, 0 = background).

    Priority:
      1. Alpha channel from the RGBA render (1.0 = cloth, 0.0 = background).
         This is pixel-perfect regardless of shadow darkness — even a cloth
         pixel rendered as pure black (deep shadow) has alpha = 1.
      2. Brightness threshold fallback for legacy RGB-only renders.
    """
    if alpha is not None and alpha.max() > 0:
        _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        k    = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        return mask
    # Legacy fallback — brightness threshold on the beauty render
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 4, 255, cv2.THRESH_BINARY)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def get_object_mask(
    img_bgr: np.ndarray,
    alpha:   "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Returns binary uint8 mask (255 = object foreground).
    Uses SAM with rough-centroid point-prompt when available;
    otherwise falls back to _threshold_mask (alpha-based when available).
    """
    predictor = _load_sam()
    if predictor:
        rough = _threshold_mask(img_bgr, alpha=alpha)
        ys, xs = np.where(rough > 0)
        cx = int(np.mean(xs)) if len(xs) else img_bgr.shape[1] // 2
        cy = int(np.mean(ys)) if len(ys) else img_bgr.shape[0] // 2
        predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks[np.argmax(scores)].astype(np.uint8) * 255
    return _threshold_mask(img_bgr, alpha=alpha)


def draw_organic_dashed_boundary(
    draw:       ImageDraw.Draw,
    seg_mask:   np.ndarray,
    color:      tuple,
    dilation:   int = 40,
    dash_px:    int = 18,
    gap_px:     int = 9,
    line_width: int = 2,
) -> "tuple | None":
    """
    Puffs the seg_mask out with a large elliptical dilation, finds the contour
    of that dilated shape, and draws a dashed line along its curved organic
    path.  The result is a loose "aura" that follows the silhouette of the
    garment rather than a rigid rectangular bounding box.

    Returns boundary_top (topmost contour point) as the arrow anchor for the
    "Segmentation Mask" label.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    dilated = cv2.dilate(seg_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    # Simplify with approxPolyDP to reduce point count while keeping the
    # organic shape; epsilon = 0.3 % of perimeter keeps curves smooth.
    epsilon = 0.003 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    pts = contour.reshape(-1, 2).tolist()
    n   = len(pts)
    if n < 3:
        return None

    # Topmost point as arrow anchor
    top_idx      = min(range(n), key=lambda i: pts[i][1])
    boundary_top = (pts[top_idx][0], pts[top_idx][1])

    # Walk contour with arc-length dashing
    arc_acc = 0.0
    drawing = True
    seg_lim = float(dash_px)
    prev    = (float(pts[0][0]), float(pts[0][1]))

    for k in range(1, n + 1):
        curr      = (float(pts[k % n][0]), float(pts[k % n][1]))
        step      = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        remaining = step

        while remaining > 0:
            budget = seg_lim - arc_acc
            if remaining >= budget:
                t  = (budget / step) if step > 0 else 0.0
                ip = (
                    prev[0] + t * (curr[0] - prev[0]),
                    prev[1] + t * (curr[1] - prev[1]),
                )
                if drawing:
                    draw.line([prev, ip], fill=color, width=line_width)
                drawing   = not drawing
                arc_acc   = 0.0
                seg_lim   = float(dash_px) if drawing else float(gap_px)
                remaining -= budget
                prev       = ip
            else:
                if drawing:
                    draw.line([prev, curr], fill=color, width=line_width)
                arc_acc  += remaining
                remaining = 0
        prev = curr

    return boundary_top

