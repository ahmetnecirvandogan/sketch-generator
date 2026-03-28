"""Shadow percentile mask and diagonal hatching."""

import math
import random

import cv2
import numpy as np

def compute_shadow_mask(
    img_bgr:    np.ndarray,
    seg_mask:   np.ndarray,
    percentile: float = 25.0,
) -> np.ndarray:
    """
    Pixels whose luminance is below the *percentile*-th value of all object
    pixels are marked as shadow.  The threshold self-calibrates to each
    render's unique lighting.
    """
    gray          = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    object_pixels = gray[seg_mask > 0]
    if object_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    thresh = float(np.percentile(object_pixels, percentile))
    shadow = ((gray <= thresh).astype(np.uint8) * 255)
    shadow = cv2.bitwise_and(shadow, seg_mask)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return shadow


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

