"""Shadow percentile mask and diagonal hatching."""

import math
import random

import cv2
import numpy as np

from cloth_pipeline.sketch.tones import smooth_lighting_luma

def compute_shadow_mask(
    img_bgr:    np.ndarray,
    seg_mask:   np.ndarray,
    percentile: float = 18.0,
) -> np.ndarray:
    """
    Shadow mask from low-frequency lighting, not raw albedo texture.

    Pixels whose smoothed luminance is below the *percentile*-th value over
    object pixels are marked as shadow.
    """
    luma = smooth_lighting_luma(img_bgr, seg_mask)
    object_pixels = luma[seg_mask > 0]
    if object_pixels.size == 0:
        return np.zeros_like(luma, dtype=np.uint8)

    thresh = float(np.percentile(object_pixels, percentile))
    p25 = float(np.percentile(object_pixels, 25))
    p50 = float(np.percentile(object_pixels, 50))
    p75 = float(np.percentile(object_pixels, 75))
    iqr = max(1.0, p75 - p25)

    # Local-darkness response: keeps pockets that are darker than nearby cloth.
    # This suppresses broad tonal drift (e.g. lower garment gently darker overall).
    h, w = luma.shape[:2]
    k = int(round(min(h, w) * 0.12))
    k = max(25, min(91, k))
    if k % 2 == 0:
        k += 1
    trend = cv2.GaussianBlur(luma, (k, k), 0).astype(np.float32)
    response = trend - luma.astype(np.float32)
    resp_vals = response[seg_mask > 0]
    resp_thresh = float(np.percentile(resp_vals, 84)) if resp_vals.size > 0 else 0.0

    # Reject weak tonal drift so only clearly visible camera-facing shadows get '#'.
    min_drop = max(12.0, 1.0 * iqr)
    shadow_bool = (
        (luma <= thresh)
        & ((p50 - luma.astype(np.float32)) >= min_drop)
        & (response >= resp_thresh)
    )
    shadow = (shadow_bool.astype(np.uint8) * 255)
    shadow = cv2.bitwise_and(shadow, seg_mask)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    # 5×5 close links nearby shadow without swallowing the whole garment when
    # luminance is compressed (e.g. translucent chiffon, narrow histogram).
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Remove tiny speckles that become stray '#' markers.
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(shadow)
    if n > 1:
        min_area = max(16, int(round(0.0018 * int((seg_mask > 0).sum()))))
        ys_obj = np.where(seg_mask > 0)[0]
        y_bottom_gate = float(np.percentile(ys_obj, 68)) if ys_obj.size > 0 else float(h * 0.68)
        strong_resp = float(np.percentile(resp_vals, 93)) if resp_vals.size > 0 else resp_thresh
        keep = np.zeros_like(shadow, dtype=np.uint8)
        for i in range(1, n):
            area_i = int(stats[i, cv2.CC_STAT_AREA])
            if area_i < min_area:
                continue
            cx_i = float(stats[i, cv2.CC_STAT_LEFT] + 0.5 * stats[i, cv2.CC_STAT_WIDTH])
            cy_i = float(stats[i, cv2.CC_STAT_TOP] + 0.5 * stats[i, cv2.CC_STAT_HEIGHT])
            comp = lbl == i
            mean_resp = float(response[comp].mean()) if np.any(comp) else 0.0
            # Lower-tail shadows are accepted only when they have strong local contrast.
            if cy_i >= y_bottom_gate and mean_resp < strong_resp:
                continue
            _ = cx_i  # keeps explicit geometry vars for readability/debugging
            keep[comp] = 255
        shadow = keep
    return shadow


def shadow_mask_darkest_fraction(
    img_bgr: np.ndarray,
    seg_mask: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """
    Marks exactly the ``fraction`` darkest object pixels (stable sort), then
    applies a light open/close.  Use when percentile thresholds tie across most
    of the cloth and would otherwise hatch nearly the entire silhouette.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(seg_mask > 0)
    if ys.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    vals = gray[ys, xs]
    n = int(vals.size)
    k = max(1, min(n, int(round(n * fraction))))
    order = np.argsort(vals, kind="stable")
    out = np.zeros_like(gray, dtype=np.uint8)
    out[ys[order[:k]], xs[order[:k]]] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out


def draw_hatching(
    canvas:    np.ndarray,
    mask:      np.ndarray,
    color_bgr: tuple,
    spacing:   int   = 12,
    angle_deg: float = 45.0,
    thickness: int   = 1,
) -> np.ndarray:
    """
    Draws parallel diagonal lines at *angle_deg* across the full canvas,
    then masks them to *mask* so ink falls ONLY where the shadow pixels are.

    Each line receives a tiny random nudge in angle (±1.5°) and spacing
    (±2 px) to produce the loose, clustered feel of human hatching rather
    than a perfectly uniform grid.  Semantic alignment with the render is
    preserved because the mask boundary never changes.
    """
    h, w      = canvas.shape[:2]
    hatch_lay = np.zeros((h, w), dtype=np.uint8)
    diag      = int(math.hypot(w, h)) + 1

    offset = -float(diag)
    while offset < diag:
        actual_angle = angle_deg + random.gauss(0, 1.5)
        rad = math.radians(actual_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        x1 = int(-diag * cos_a - offset * sin_a)
        y1 = int(-diag * sin_a + offset * cos_a)
        x2 = int( diag * cos_a - offset * sin_a)
        y2 = int( diag * sin_a + offset * cos_a)
        cv2.line(hatch_lay, (x1, y1), (x2, y2), 255, thickness)

        offset += spacing + random.uniform(-2.0, 2.0)

    applied = cv2.bitwise_and(hatch_lay, mask)
    canvas[applied > 0] = color_bgr
    return canvas

