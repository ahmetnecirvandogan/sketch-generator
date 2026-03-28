"""Highlight / texture / shadow feature points and dominant colour name."""

import cv2
import numpy as np

def find_feature_points(
    img_bgr:  np.ndarray,
    seg_mask: np.ndarray,
) -> dict:
    """
    Returns a dict with keys 'highlight', 'texture', 'shadow' → (x, y) or None.

    • highlight — centroid of the LARGEST connected region above the 90th
                  brightness percentile.  The largest-component filter prevents
                  scattered fringe tips (which are individually bright) from
                  pulling the centroid away from the main body highlight.
    • texture   — centroid of pixels in the 40th–60th brightness percentile
    • shadow    — raw centroid (will be overridden in generate_sketch with the
                  centroid of the hatching-filtered shadow mask)
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[seg_mask > 0]
    if len(pixels) == 0:
        return {"highlight": None, "texture": None, "shadow": None}

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

    def _largest_component(m):
        """Keep only the biggest connected blob to discard scattered outliers."""
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return m
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((lbl == biggest).astype(np.uint8) * 255)

    return {
        "highlight": _centroid(_largest_component(_masked(p90, 255))),
        "texture":   _centroid(_masked(p40, p60)),
        "shadow":    _centroid(_masked(0,   p20)),   # overridden in generate_sketch
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

