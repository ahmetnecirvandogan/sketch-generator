"""Highlight / mid-tone / shadow feature points and dominant colour name."""

from __future__ import annotations

import cv2
import numpy as np


def _interior_representative(mask: np.ndarray) -> tuple[int, int] | None:
    """
    A point guaranteed to lie on the blob.  Uses the centroid when it falls
    inside the mask; otherwise the distance-transform peak (avoids centroids
    landing outside concave highlight/shadow regions).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = int(round(float(np.mean(xs))))
    cy = int(round(float(np.mean(ys))))
    h, w = mask.shape[:2]
    if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
        return cx, cy
    binm = (mask > 0).astype(np.uint8)
    d = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
    if float(d.max()) <= 0:
        return int(xs[0]), int(ys[0])
    yi, xi = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(xi), int(yi)


def _shade_mark_positions(
    band_mask: np.ndarray,
    seg_mask: np.ndarray,
    garment_px: int,
    *,
    max_blobs: int = 8,
    min_area_frac: float = 0.0035,
    min_area_abs: int = 120,
) -> list[tuple[int, int]]:
    """
    One marker per significant connected component inside the garment interior
    (eroded mask so points do not sit on the outer silhouette cut line).
    """
    k5 = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(seg_mask, k5, iterations=1)
    if not np.any(eroded):
        eroded = cv2.erode(seg_mask, np.ones((3, 3), np.uint8), iterations=1)
    # Prefer eroded interior so markers sit on cloth, not on the cut silhouette.
    interior_tries: list[np.ndarray] = [x for x in (eroded, seg_mask) if np.any(x)]

    min_area = max(min_area_abs, int(garment_px * min_area_frac))
    k3 = np.ones((3, 3), np.uint8)

    for interior in interior_tries:
        m = cv2.bitwise_and(band_mask, interior)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        if not np.any(m):
            continue
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        candidates: list[tuple[int, tuple[int, int]]] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            blob = np.where(lbl == i, np.uint8(255), np.uint8(0))
            pt = _interior_representative(blob)
            if pt is not None:
                candidates.append((area, pt))
        if candidates:
            candidates.sort(key=lambda t: -t[0])
            return [pt for _, pt in candidates[:max_blobs]]
    return []


def find_feature_points(
    img_bgr:  np.ndarray,
    seg_mask: np.ndarray,
) -> dict:
    """
    Returns:

      • ``highlights`` — list of ``(x, y)`` on the garment for sparkle markers
        (one per major bright blob above the 90th percentile).
      • ``shadows`` — list of ``(x, y)`` for crescent markers (darkest ~20th
        percentile blobs).
      • ``midtone`` — single centroid in the 40th–60th band, or ``None``.

    Markers are chosen inside an eroded object mask so they stay on the cloth,
    not on the background outside the silhouette.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[seg_mask > 0]
    if len(pixels) == 0:
        return {"highlights": [], "shadows": [], "midtone": None}

    p20 = float(np.percentile(pixels, 20))
    p40 = float(np.percentile(pixels, 40))
    p60 = float(np.percentile(pixels, 60))
    p90 = float(np.percentile(pixels, 90))

    def _masked(lo, hi):
        m = cv2.bitwise_and(
            (gray >= lo).astype(np.uint8) * 255,
            (gray <= hi).astype(np.uint8) * 255,
        )
        return cv2.bitwise_and(m, seg_mask)

    def _centroid(m):
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            return None
        return int(np.mean(xs)), int(np.mean(ys))

    garment_px = int(np.count_nonzero(seg_mask > 0))
    bright = _masked(p90, 255)
    dark = _masked(0, p20)

    return {
        "highlights": _shade_mark_positions(bright, seg_mask, garment_px),
        "shadows": _shade_mark_positions(dark, seg_mask, garment_px),
        "midtone": _centroid(_masked(p40, p60)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOMINANT COLOUR DETECTION  (for text block label)
# ─────────────────────────────────────────────────────────────────────────────

def detect_dominant_color(img_bgr: np.ndarray, seg_mask: np.ndarray) -> str:
    """
    Returns a human-readable colour name (e.g. "Pink", "Blue") from the
    mean BGR value of pixels inside the segmentation mask.
    """
    pixels = img_bgr[seg_mask > 0]
    if len(pixels) == 0:
        return ""

    b, g, r = [float(v) for v in pixels.mean(axis=0)]
    nr, ng, nb = r / 255.0, g / 255.0, b / 255.0
    mx    = max(nr, ng, nb)
    mn    = min(nr, ng, nb)
    delta = mx - mn

    if delta < 0.12:
        if mx > 0.85: return "White"
        if mx > 0.40: return "Gray"
        return "Black"

    if mx == nr:
        h = 60.0 * (((ng - nb) / delta) % 6)
    elif mx == ng:
        h = 60.0 * ((nb - nr) / delta + 2)
    else:
        h = 60.0 * ((nr - ng) / delta + 4)
    if h < 0:
        h += 360.0

    sat = delta / mx if mx > 0 else 0.0

    # Pink: hue near red but desaturated or bright
    if h < 20 or h >= 340:
        return "Pink" if (sat < 0.55 or mx > 0.75) else "Red"
    if  20 <= h <  45: return "Orange"
    if  45 <= h <  70: return "Yellow"
    if  70 <= h < 165: return "Green"
    if 165 <= h < 200: return "Cyan"
    if 200 <= h < 265: return "Blue"
    if 265 <= h < 295: return "Purple"
    return "Pink"

