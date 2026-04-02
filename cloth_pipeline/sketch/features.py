"""Highlight / mid-tone / shadow feature points and dominant colour name."""

import cv2
import numpy as np

from cloth_pipeline.sketch.tones import smooth_lighting_luma


def highlight_region_mask(
    img_bgr: np.ndarray,
    seg_mask: np.ndarray,
    *,
    keep_largest: bool = True,
    percentile: float = 90.0,
    min_rise_scale: float = 0.75,
    min_area_frac: float = 0.0015,
) -> np.ndarray:
    """
    Binary mask of the main highlight from low-frequency lighting.

    Uses smoothed luminance so texture pattern does not create fake highlights.
    By default returns only the largest connected region. For marker placement,
    set ``keep_largest=False`` to keep multiple visible bright regions.
    """
    luma = smooth_lighting_luma(img_bgr, seg_mask)
    pixels = luma[seg_mask > 0]
    if len(pixels) == 0:
        return np.zeros_like(seg_mask, dtype=np.uint8)

    p90 = float(np.percentile(pixels, percentile))
    p25 = float(np.percentile(pixels, 25))
    p50 = float(np.percentile(pixels, 50))
    p75 = float(np.percentile(pixels, 75))
    iqr = max(1.0, p75 - p25)
    # Keep only clearly bright regions (camera-visible highlights), not mild
    # albedo variation that survives smoothing.
    min_rise = max(8.0, float(min_rise_scale) * iqr)
    bright_bool = (luma >= p90) & ((luma.astype(np.float32) - p50) >= min_rise)
    bright = cv2.bitwise_and((bright_bool.astype(np.uint8) * 255), seg_mask)

    obj_count = int((seg_mask > 0).sum())
    # When the lit shell is bright but histogram compression hides p90 tails,
    # union a milder upper-quantile band (still above median + k·IQR).
    if obj_count > 0 and np.count_nonzero(bright) < max(80, int(0.012 * obj_count)):
        p_relax = max(72.0, percentile - 12.0)
        pr = float(np.percentile(pixels, p_relax))
        min_rise2 = max(4.5, 0.38 * iqr)
        extra = (
            (luma >= pr)
            & ((luma.astype(np.float32) - p50) >= min_rise2)
            & (seg_mask > 0)
        )
        bright = cv2.bitwise_or(
            bright, (extra.astype(np.uint8) * 255)
        )

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n <= 1:
        return bright
    if keep_largest:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((lbl == biggest).astype(np.uint8) * 255)

    out = np.zeros_like(bright, dtype=np.uint8)
    min_area = max(12, int(round(float(min_area_frac) * int((seg_mask > 0).sum()))))
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            out[lbl == i] = 255
    return out


def find_feature_points(
    img_bgr:  np.ndarray,
    seg_mask: np.ndarray,
) -> dict:
    """
    Returns a dict with keys 'highlight', 'midtone', 'shadow' → (x, y) or None.

    • highlight — centroid of the LARGEST connected region above the 90th
                  brightness percentile.  The largest-component filter prevents
                  scattered fringe tips (which are individually bright) from
                  pulling the centroid away from the main body highlight.
    • midtone   — centroid of pixels in the 40th–60th brightness percentile
    • shadow    — raw centroid (will be overridden in generate_sketch with the
                  centroid of the hatching-filtered shadow mask)
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[seg_mask > 0]
    if len(pixels) == 0:
        return {"highlight": None, "midtone": None, "shadow": None}

    p20 = float(np.percentile(pixels, 20))
    p40 = float(np.percentile(pixels, 40))
    p60 = float(np.percentile(pixels, 60))
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

    return {
        "highlight": _centroid(highlight_region_mask(img_bgr, seg_mask)),
        "midtone":   _centroid(_masked(p40, p60)),
        "shadow":    _centroid(_masked(0,   p20)),   # overridden in generate_sketch
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOMINANT COLOUR DETECTION  (for text block label)
# ─────────────────────────────────────────────────────────────────────────────

def detect_dominant_color(img_bgr: np.ndarray, seg_mask: np.ndarray) -> str:
    """
    Returns a human-readable colour name (e.g. "Pink", "Blue") from mean BGR
    on **lit** cloth pixels. Deep shadow pixels are ignored so a green/white
    checkerboard with large black folded regions is not labelled "Black".
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    obj = seg_mask > 0
    vals = gray[obj]
    if vals.size == 0:
        return ""

    # Drop the darkest majority (fold shadow) so mean hue reflects albedo.
    # Use strict > so a huge ``luma==0`` plateau is excluded when p55==0.
    thr = float(np.percentile(vals, 55.0))
    sel = obj & (gray > thr)
    min_keep = max(64, int(round(0.02 * float(vals.size))))
    if np.count_nonzero(sel) < min_keep:
        thr = float(np.percentile(vals, 40.0))
        sel = obj & (gray > thr)
    if np.count_nonzero(sel) < max(32, int(round(0.01 * float(vals.size)))):
        sel = obj

    pixels = img_bgr[sel]
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

