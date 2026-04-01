"""Low-frequency lighting proxies for highlight/shadow detection."""

from __future__ import annotations

import cv2
import numpy as np


def smooth_lighting_luma(
    img_bgr: np.ndarray,
    seg_mask: np.ndarray,
    *,
    blur_frac: float = 0.06,
    min_ksize: int = 21,
    max_ksize: int = 71,
) -> np.ndarray:
    """
    Return a smoothed luminance map that suppresses high-frequency albedo texture.

    This keeps broad, camera-visible lighting changes (real highlight/shadow)
    while damping checker/grid patterns from the base color map.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    k = int(round(min(h, w) * float(blur_frac)))
    k = max(int(min_ksize), min(int(max_ksize), k))
    if k % 2 == 0:
        k += 1

    obj = seg_mask > 0
    if not np.any(obj):
        return cv2.GaussianBlur(gray, (k, k), 0)

    fill_val = int(np.median(gray[obj]))
    padded = np.full_like(gray, fill_val, dtype=np.uint8)
    padded[obj] = gray[obj]
    return cv2.GaussianBlur(padded, (k, k), 0)

