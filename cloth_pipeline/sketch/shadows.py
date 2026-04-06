"""Shadow percentile mask and diagonal hatching."""

from __future__ import annotations

import math
import os
import random

import cv2
import numpy as np

from cloth_pipeline.sketch.tones import smooth_lighting_luma


def compute_shadow_mask(
    img_bgr:    np.ndarray,
    seg_mask:   np.ndarray,
    percentile: float = 18.0,
    *,
    normal_path: str | None = None,
    depth_path: str | None = None,
    cam_origin: list | tuple | None = None,
    cam_target: list | tuple | None = None,
    fov_deg: float = 40.0,
) -> np.ndarray:
    """
    Shadow mask from low-frequency lighting plus raw-luma guards.

    - **Lit exclusion:** upper quantiles on smoothed luma and raw grayscale
      suppress ``#`` on brightly lit regions (fixes false positives on even
      fabrics and sunlit sides).
    - **Flat lighting:** when the object histogram is narrow (studio fill),
      broad shadow is tightened so only clearly dark pixels qualify.
    - **Degenerate bimodal:** when ``p86 - gap`` would exclude almost
      everything, fall back to a lower-tail band so uniform dark interiors
      still get markers.
    """
    luma = smooth_lighting_luma(img_bgr, seg_mask)
    raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    object_pixels = luma[seg_mask > 0]
    raw_obj = raw[seg_mask > 0].astype(np.float32)
    if object_pixels.size == 0:
        return np.zeros_like(luma, dtype=np.uint8)

    thresh = float(np.percentile(object_pixels, percentile))
    p25 = float(np.percentile(object_pixels, 25))
    p50 = float(np.percentile(object_pixels, 50))
    p75 = float(np.percentile(object_pixels, 75))
    p87 = float(np.percentile(object_pixels, 87.0))
    p90 = float(np.percentile(object_pixels, 90.0))
    p92 = float(np.percentile(object_pixels, 92.0))
    iqr = max(1.0, p75 - p25)

    h, w = luma.shape[:2]
    k = int(round(min(h, w) * 0.12))
    k = max(25, min(91, k))
    if k % 2 == 0:
        k += 1
    trend = cv2.GaussianBlur(luma, (k, k), 0).astype(np.float32)
    response = trend - luma.astype(np.float32)
    resp_vals = response[seg_mask > 0]
    resp_thresh = float(np.percentile(resp_vals, 78)) if resp_vals.size > 0 else 0.0
    if p50 > 160.0:
        resp_thresh = resp_thresh * 1.14
    # Mid–high key: weave / micro-crease response reads as shadow though the cloth is evenly lit (e.g. 38).
    if 118.0 < p50 < 176.0:
        resp_thresh = resp_thresh * 1.14
        min_drop = max(8.5, 0.85 * iqr) * 1.10
    else:
        min_drop = max(8.5, 0.85 * iqr)
    if p50 > 155.0:
        min_drop = min_drop * 1.12
    crease_shadow = (
        (luma <= thresh)
        & ((p50 - luma.astype(np.float32)) >= min_drop)
        & (response >= resp_thresh)
    )
    if 118.0 < p50 < 176.0:
        crease_shadow = crease_shadow & (
            luma.astype(np.float32) < (p50 - 0.11 * float(iqr))
        )

    lmin = float(object_pixels.min())
    lmax = float(object_pixels.max())
    span = max(1.0, lmax - lmin)

    broad_median = (
        (luma <= float(np.percentile(object_pixels, min(44.0, percentile + 24.0))))
        & (luma.astype(np.float32) < p50 - max(5.0, 0.2 * iqr))
    )

    knee = p90 - p75
    if knee > 28.0:
        split = p75 + 0.32 * knee
        broad_bimodal = (luma.astype(np.float32) < split) & (seg_mask > 0)
    else:
        p_hi_lit = float(np.percentile(object_pixels, 86.0))
        gap = max(10.0, 0.06 * span, 0.45 * max(iqr, 1.0))
        cut = p_hi_lit - gap
        broad_bimodal = (luma.astype(np.float32) < cut) & (seg_mask > 0)
        # Degenerate: cut small or almost nothing selected — uniform dark interiors
        # (heavy black tail; crease response ~ 0).
        obj_n = int((seg_mask > 0).sum())
        if cut <= 8.0 or np.count_nonzero(broad_bimodal) < max(80, int(0.008 * obj_n)):
            p72 = float(np.percentile(object_pixels, 72.0))
            broad_bimodal = (
                (luma.astype(np.float32) < min(p92 - 0.5, p72 + 1e-3))
                & (luma.astype(np.float32) <= p72)
                & (seg_mask > 0)
            )

    if p50 < 22.0 or iqr < 0.07 * span:
        broad_shadow = broad_bimodal
    else:
        broad_shadow = broad_median | broad_bimodal

    # Flat lighting: narrow histogram → restrict broad shadow to darker band.
    spread_smooth = max(1.0, p92 - p25)
    flat_light = spread_smooth < 40.0
    if flat_light:
        dark_ceiling = p25 + 0.48 * (p90 - p25)
        # High-key studio renders: most of the cloth is mid-bright — only deepest tones.
        if p50 > 150:
            dark_ceiling = min(dark_ceiling, p25 + 0.20 * (p90 - p25))
        if p50 > 120:
            dark_ceiling = min(dark_ceiling, p50 + 0.12 * (p90 - p25))
        broad_shadow = broad_shadow & (luma.astype(np.float32) < dark_ceiling)
        crease_shadow = crease_shadow & (luma.astype(np.float32) < p50 + 0.28 * (p90 - p50))
        if p50 > 150:
            crease_shadow = crease_shadow & (
                luma.astype(np.float32) < p25 + 0.32 * (p90 - p25)
            )
        if p50 > 175:
            crease_shadow = crease_shadow & (
                luma.astype(np.float32) < p25 + 0.16 * (p90 - p25)
            )

    shadow_bool = crease_shadow | broad_shadow

    # Near-white, evenly lit renders: keep only the deepest tones.
    if p50 > 178.0:
        deep_cut = p25 + 0.14 * (p90 - p25)
        shadow_bool = shadow_bool & (luma.astype(np.float32) < deep_cut)
    # Studio-bright swatches: broad shadow is almost always wrong — creases only.
    if p50 > 182.0:
        shadow_bool = crease_shadow
        crease_tight = p25 + 0.11 * max(1.0, (p90 - p25))
        if p50 > 188.0:
            crease_tight = p25 + 0.065 * max(1.0, (p90 - p25))
        shadow_bool = shadow_bool & (luma.astype(np.float32) < crease_tight)

    # Shading normals: camera-facing, moderately bright = lit shell, not deep shadow.
    h0, w0 = luma.shape[:2]
    if (
        normal_path
        and depth_path
        and cam_origin is not None
        and cam_target is not None
        and len(cam_origin) >= 3
        and len(cam_target) >= 3
        and os.path.isfile(depth_path)
        and os.path.isfile(normal_path)
        and p50 > 42.0
    ):
        try:
            from cloth_pipeline.sketch.visibility import (
                load_world_normals,
                ray_directions_world,
            )

            normals = load_world_normals(normal_path, h0, w0)
            depth = np.load(depth_path).astype(np.float32)
            if depth.shape[:2] != (h0, w0):
                depth = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_NEAREST)
            if normals is not None:
                rays = ray_directions_world(cam_origin, cam_target, float(fov_deg), w0, h0)
                v_to_cam = -rays
                valid = np.isfinite(depth) & (depth > 1e-6)
                cloth = (seg_mask > 0) & valid
                if np.any(cloth):
                    ndot_probe = np.sum(normals * v_to_cam, axis=-1)
                    med = float(np.median(ndot_probe[cloth]))
                    if med < 0.0:
                        normals = -normals
                ndot = np.sum(normals * v_to_cam, axis=-1)
                sunlit_shell = (ndot > 0.13) & (
                    luma.astype(np.float32)
                    >= (p25 + 0.09 * max(1.0, (p90 - p25)))
                )
                if p50 > 85.0:
                    sunlit_shell = sunlit_shell | (
                        (ndot > 0.19)
                        & (luma.astype(np.float32) >= float(np.percentile(object_pixels, 36.0)))
                    )
                shadow_bool = shadow_bool & (~(sunlit_shell & valid & (seg_mask > 0)))
        except Exception:
            pass

    # Mid–high key: only the darker tail reads as real shadow; smooth luma can still
    # dip in flat-lit weave (false creases) until this cap (e.g. 38).
    if 128.0 < p50 < 172.0:
        _cap = min(
            float(p50 - 0.28 * float(iqr)),
            float(np.percentile(object_pixels, 26.0)),
        )
        shadow_bool = shadow_bool & (luma.astype(np.float32) < _cap)

    # --- Lit exclusion (bright objects: raw + smooth; dark objects: avoid killing shadow). ---
    raw_p78 = float(np.percentile(raw_obj, 78.0))
    raw_p90 = float(np.percentile(raw_obj, 90.0))
    if p50 > 120.0:
        upper_smooth = luma.astype(np.float32) >= float(
            np.percentile(object_pixels, 82.0)
        )
    else:
        upper_smooth = luma.astype(np.float32) >= p87
    midlit = (luma.astype(np.float32) >= (p50 + 0.16 * spread_smooth)) & (
        luma.astype(np.float32) >= p75
    )
    if flat_light:
        lit_cut = p25 + 0.40 * (p92 - p25)
        midlit = midlit | (luma.astype(np.float32) >= lit_cut)
    # On nearly black cloth, raw p78 sits in the shadow range — only exclude extreme highlights.
    if p90 < 38.0:
        lit_raw = raw.astype(np.float32) >= float(np.percentile(raw_obj, 96.0))
    else:
        lit_raw = raw.astype(np.float32) >= raw_p78
    if raw_p90 > 200.0:
        lit_raw = lit_raw | (raw.astype(np.float32) >= (raw_p90 - 12.0))
    clearly_lit = upper_smooth | midlit | lit_raw
    # Sunlit mid-tones on bright scarves: exclude upper-mid quantile as shadow.
    if p50 > 100.0:
        clearly_lit = clearly_lit | (
            luma.astype(np.float32) >= float(np.percentile(object_pixels, 66.0))
        )
    if p50 > 162.0:
        clearly_lit = clearly_lit | (
            luma.astype(np.float32) >= float(np.percentile(object_pixels, 54.0))
        )
    elif p50 > 95.0:
        clearly_lit = clearly_lit | (
            luma.astype(np.float32) >= (p50 + 0.11 * spread_smooth)
        )
    # Even studio mid–high key: pull the lower-mid tonal band out of shadow (false # in flat-lit cloth).
    if 118.0 < p50 < 176.0:
        clearly_lit = clearly_lit | (
            luma.astype(np.float32) >= (p25 + 0.14 * max(1.0, (p90 - p25)))
        )
    shadow_bool = shadow_bool & (~clearly_lit)

    shadow = (shadow_bool.astype(np.uint8) * 255)
    shadow = cv2.bitwise_and(shadow, seg_mask)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(shadow)
    if n > 1:
        min_area = max(12, int(round(0.0012 * int((seg_mask > 0).sum()))))
        keep = np.zeros_like(shadow, dtype=np.uint8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                keep[lbl == i] = 255
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
