"""
generate_sketches.py
--------------------
Stage 2 of the Neural-Contours-Cloth ControlNet dataset pipeline.

Converts each Mitsuba render into a precisely-aligned sketch conditioning
image via a four-step computer-vision pipeline:

  A. Edge Detection  — HED neural edge detector (OpenCV DNN) if model files
                       are present; otherwise bilateral-filtered adaptive Canny
                       (bilateral filter preserves structural cloth folds while
                       suppressing wool micro-texture noise).
                       Edges are restricted to the object mask so background
                       pixels never produce spurious lines.

  B. Segmentation    — SAM (segment-anything) if checkpoint is present;
                       otherwise threshold + morphological cleanup.
                       The boundary is drawn as a PADDED CONVEX HULL — a
                       loose dashed outline ~40 px outside the cloth shape,
                       labeled "Segmentation Mask" with an arrow.

  C. Shadow Hatching — Pixels below the 25th brightness percentile of the
                       object are identified as shadow zones. Programmatic
                       45° diagonal hatching is drawn ONLY in those zones
                       (single direction — no cross-hatch grid).

  D. Annotation      — 4 labeled arrows (Segmentation Mask, Highlight,
                       textured, Shadow) + 3-line handwritten text block
                       (object name with dominant colour, texture type,
                       keyword).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIONAL MODEL FILES  (place in project root or set env-vars)

  HED neural edge detector:
    HED_MODEL_DIR (default: project root)
    Files needed:
      • deploy.prototxt
      • hed_pretrained_bsds.caffemodel
    Download: https://github.com/s9xie/hed

  SAM segmentation model:
    SAM_CHECKPOINT (default: <project root>/sam_vit_h_4b8939.pth)
    pip install git+https://github.com/facebookresearch/segment-anything.git
    Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

  Without these files the script uses bilateral Canny + threshold, which is
  robust for clean Mitsuba renders on a black background.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import json
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
RENDERS_DIR   = os.path.join(DATASET_DIR, "renders")
CONDITION_DIR = os.path.join(DATASET_DIR, "conditioning")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")

os.makedirs(CONDITION_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR (teal-green marker)
# ─────────────────────────────────────────────────────────────────────────────
SKETCH_RGB = (34, 139, 69)    # PIL  / RGB
SKETCH_BGR = (69, 139, 34)    # OpenCV / BGR

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PATHS  (overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────
_HED_DIR      = os.environ.get("HED_MODEL_DIR", BASE_DIR)
HED_PROTO     = os.path.join(_HED_DIR, "deploy.prototxt")
HED_WEIGHTS   = os.path.join(_HED_DIR, "hed_pretrained_bsds.caffemodel")

SAM_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT", os.path.join(BASE_DIR, "sam_vit_h_4b8939.pth")
)
SAM_MODEL_TYPE = "vit_h"

# ─────────────────────────────────────────────────────────────────────────────
# FONT RESOLUTION  (searches macOS system fonts; falls back to PIL default)
# ─────────────────────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    os.path.join(BASE_DIR, "handwriting.ttf"),
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
]

def _resolve_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except IOError:
                continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# STEP A — EDGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class _HEDCropLayer:
    """OpenCV DNN custom layer for HED's spatial crop operation."""
    def __init__(self, params, blobs):
        self.xstart = self.ystart = self.xend = self.yend = 0

    def getMemoryShapes(self, inputs):
        src, tgt = inputs[0], inputs[1]
        batch, channels = src[0], src[1]
        h, w = tgt[2], tgt[3]
        self.ystart = (src[2] - h) // 2
        self.xstart = (src[3] - w) // 2
        self.yend   = self.ystart + h
        self.xend   = self.xstart + w
        return [[batch, channels, h, w]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

try:
    cv2.dnn_registerLayer("Crop", _HEDCropLayer)
except Exception:
    pass

_hed_net = None


def _load_hed():
    global _hed_net
    if _hed_net is not None:
        return _hed_net
    if os.path.exists(HED_PROTO) and os.path.exists(HED_WEIGHTS):
        _hed_net = cv2.dnn.readNetFromCaffe(HED_PROTO, HED_WEIGHTS)
        print("[A] HED neural edge detector loaded.")
    else:
        print(
            f"[A] HED model not found in '{_HED_DIR}' — using bilateral Canny.\n"
            "    (Download deploy.prototxt + hed_pretrained_bsds.caffemodel to enable HED)"
        )
        _hed_net = False
    return _hed_net or None


def detect_edges(
    img_bgr:     np.ndarray,
    seg_mask:    "np.ndarray | None" = None,
    normal_path: "str | None"        = None,
) -> np.ndarray:
    """
    Returns a uint8 binary edge map (255 = edge, 0 = background).

    Priority order:
      1. HED neural edge detector (if model files present)
      2. Normal-map gradient (if a .npy file is given and exists) — detects
         only geometric fold edges by measuring how rapidly the surface normal
         changes direction.  Wool fibre texture has near-zero normal gradient
         so it is completely invisible to this detector.
      3. Bilateral-filtered Canny fallback on the RGB render (d=15 to
         aggressively suppress micro-texture before Canny).

    Edges are AND-masked to seg_mask so the black background never contributes.
    """
    h, w = img_bgr.shape[:2]
    net  = _load_hed()

    if net:
        blob = cv2.dnn.blobFromImage(
            img_bgr, scalefactor=1.0, size=(w, h),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False, crop=False,
        )
        net.setInput(blob)
        hed_out  = net.forward()
        edge_u8  = (np.clip(hed_out[0, 0], 0.0, 1.0) * 255).astype(np.uint8)
        _, edges = cv2.threshold(edge_u8, 50, 255, cv2.THRESH_BINARY)

    elif normal_path and os.path.exists(normal_path):
        # Load normal map — supports both formats saved by generate_dataset.py:
        #   .png  → uint8 BGR, loaded with cv2.imread then normalised to float
        #   .npy  → float32 (H, W, 3), values ≈ [-1, 1], loaded with np.load
        if normal_path.endswith(".png"):
            raw = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raw = np.zeros((h, w, 3), dtype=np.uint8)
            normals = raw.astype(np.float32) / 255.0   # [0, 1]
        else:
            normals = np.load(normal_path).astype(np.float32)   # [-1, 1]

        if normals.ndim == 2:
            normals = normals[:, :, np.newaxis]
        if normals.shape[:2] != (h, w):
            normals = cv2.resize(normals, (w, h))

        # Moderate Gaussian smoothing suppresses micro-texture while
        # preserving the main fold transitions in the normal map.
        edge_acc = np.zeros((h, w), dtype=np.float32)
        for c in range(min(3, normals.shape[2])):
            ch_smooth = cv2.GaussianBlur(normals[:, :, c], (11, 11), 2.5)
            gx = cv2.Sobel(ch_smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(ch_smooth, cv2.CV_32F, 0, 1, ksize=3)
            edge_acc += gx ** 2 + gy ** 2
        edge_acc = np.sqrt(edge_acc)
        mx       = edge_acc.max()
        edge_u8  = (edge_acc / mx * 255).astype(np.uint8) if mx > 0 else np.zeros((h, w), np.uint8)
        edges    = cv2.Canny(edge_u8, 60, 150)
        # Keep only the top-12 largest edge components (>20 px) so tiny
        # fragments are removed but meaningful fold lines are preserved.
        n_e, lbl_e, stats_e, _ = cv2.connectedComponentsWithStats(edges)
        if n_e > 2:
            areas   = stats_e[1:, cv2.CC_STAT_AREA]
            order   = np.argsort(areas)[::-1]
            keep    = set(
                int(order[i]) + 1
                for i in range(min(12, len(order)))
                if areas[order[i]] >= 20
            )
            clean   = np.zeros_like(edges)
            for lbl_id in keep:
                clean[lbl_e == lbl_id] = 255
            edges = clean
        print(f"[A] Normal-map edge detection active ({os.path.splitext(normal_path)[1]}, major fold lines).")

    else:
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # d=15, sigmaColor=80: more aggressive than before to wipe out the
        # fine wool fibre texture that was causing chaotic scratch noise.
        filtered = cv2.bilateralFilter(gray, d=15, sigmaColor=80, sigmaSpace=80)
        v        = float(np.median(filtered[filtered > 0])) if np.any(filtered > 0) else 128.0
        edges    = cv2.Canny(filtered,
                             int(max(0,   0.5 * v)),
                             int(min(255, 1.5 * v)))

    if seg_mask is not None:
        dilated = cv2.dilate(seg_mask, np.ones((5, 5), np.uint8), iterations=1)
        edges   = cv2.bitwise_and(edges, dilated)
    return edges


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


# ─────────────────────────────────────────────────────────────────────────────
# STEP C — SHADOW DETECTION → HATCHING
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE POINT DETECTION  (for annotation arrows)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# ORGANIC RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_wobbly_contour(
    canvas:         np.ndarray,
    contour:        np.ndarray,
    color_bgr:      tuple,
    base_thickness: int   = 3,
    wobble_amp:     float = 1.0,
    wobble_freq:    int   = 2,
) -> None:
    """
    Traces the contour with a smooth, low-frequency sine-wave wobble applied
    perpendicular to the local path direction.

    Unlike per-point Gaussian noise (which gives jagged, spiky digital
    artifacts), the wobble is parameterised by cumulative arc-length so
    adjacent points receive nearly identical displacements — the result
    looks like a confident but slightly imperfect marker stroke drawn by
    a human hand.

    Three harmonics (freq, 2×freq, 3×freq) with independent random phase
    offsets are summed so the shape is irregular without any periodicity.
    """
    pts_raw = contour.reshape(-1, 2).astype(float)
    n       = len(pts_raw)
    if n < 3:
        return

    # Cumulative arc lengths for smooth parameterisation
    arc = [0.0]
    for i in range(1, n):
        arc.append(arc[-1] + math.hypot(
            pts_raw[i, 0] - pts_raw[i - 1, 0],
            pts_raw[i, 1] - pts_raw[i - 1, 1],
        ))
    total_arc = arc[-1] + math.hypot(
        pts_raw[0, 0] - pts_raw[n - 1, 0],
        pts_raw[0, 1] - pts_raw[n - 1, 1],
    )
    if total_arc < 1.0:
        return

    # Random phase offsets fixed once per call → smooth, non-repeating wave
    phases = [random.uniform(0.0, 2 * math.pi) for _ in range(3)]
    amps   = [wobble_amp, wobble_amp * 0.35, wobble_amp * 0.15]
    freqs  = [wobble_freq, wobble_freq * 2, wobble_freq * 3]

    def _d(s: float) -> float:
        t = 2 * math.pi * s / total_arc
        return sum(a * math.sin(f * t + p) for a, f, p in zip(amps, freqs, phases))

    # Build displaced point array
    disp = []
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        dx = pts_raw[next_i, 0] - pts_raw[prev_i, 0]
        dy = pts_raw[next_i, 1] - pts_raw[prev_i, 1]
        length = math.hypot(dx, dy)
        nx, ny = (-dy / length, dx / length) if length > 0.5 else (0.0, 1.0)
        d = _d(arc[i])
        disp.append((
            int(round(pts_raw[i, 0] + nx * d)),
            int(round(pts_raw[i, 1] + ny * d)),
        ))

    for i in range(n):
        p1 = disp[i]
        p2 = disp[(i + 1) % n]
        t  = max(1, base_thickness + random.choice([-1, 0, 0, 0, 1]))
        cv2.line(canvas, p1, p2, color_bgr, t)


def draw_wool_texture(
    pil_img:  Image.Image,
    mid_mask: np.ndarray,
    color:    tuple,
    density:  float = 0.006,
) -> None:
    """
    Scatters tiny dots and short squiggles across the *mid_mask* region to
    indicate wool texture.  The marks are deliberately sparse so they read
    as a texture hint rather than a tone fill.

    • ~55 % probability → small filled dot (radius 1–2 px)
    • ~45 % probability → 3-point squiggle line (mimics a loose loop stroke)
    """
    draw = ImageDraw.Draw(pil_img)
    ys, xs = np.where(mid_mask > 0)
    if len(ys) == 0:
        return
    n_marks = min(int(len(ys) * density), 280)
    if n_marks == 0:
        return
    idx = np.random.choice(len(ys), size=n_marks, replace=False)
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        if random.random() < 0.55:
            r = random.choice([1, 1, 1, 2])
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        else:
            pts = [
                (x + random.randint(-6, 6), y + random.randint(-3, 3))
                for _ in range(3)
            ]
            draw.line(pts, fill=color, width=1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP D — ANNOTATION  (text block + 4 labeled arrows)
# ─────────────────────────────────────────────────────────────────────────────

def _bezier_quadratic(
    p0: tuple, p1: tuple, p2: tuple, n_pts: int = 18
) -> list:
    """Evaluates n_pts+1 points along the quadratic Bezier P0 → P1 → P2."""
    pts = []
    for i in range(n_pts + 1):
        t = i / n_pts
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((int(round(x)), int(round(y))))
    return pts


def _draw_wobbly_circle(
    draw:   ImageDraw.Draw,
    center: tuple,
    radius: int,
    color:  tuple,
    width:  int   = 2,
    n_pts:  int   = 28,
    wobble: float = 1.8,
) -> None:
    """
    Draws a hand-drawn-style circle as a closed polygon of *n_pts* vertices
    with a smooth low-frequency radial wobble (two sine harmonics with random
    phase), then leaves a single random gap to mimic the pen-lift at the end
    of a hand stroke.
    """
    cx, cy  = center
    phase1  = random.uniform(0, 2 * math.pi)
    phase2  = random.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n_pts):
        t = 2 * math.pi * i / n_pts
        r = (radius
             + wobble * math.sin(3 * t + phase1)
             + wobble * 0.5 * math.sin(5 * t + phase2))
        pts.append((int(round(cx + r * math.cos(t))),
                    int(round(cy + r * math.sin(t)))))
    gap_idx = random.randint(0, n_pts - 1)
    for i in range(n_pts):
        if i == gap_idx:
            continue
        draw.line([pts[i], pts[(i + 1) % n_pts]], fill=color, width=width)


def _draw_arrow(
    draw:      ImageDraw.Draw,
    tail:      tuple,
    tip:       tuple,
    color:     tuple,
    width:     int   = 2,
    head_len:  int   = 14,
    head_half: float = 28.0,
) -> None:
    """
    Draws the arrow shaft as a quadratic Bezier curve — the control point is
    offset perpendicular to the shaft by a random lateral amount, so the line
    arcs gently rather than bending at a sharp midpoint kink.  Each arrowhead
    wing carries independent random length/spread jitter so the head looks
    like two quick marker strokes.
    """
    dx     = tip[0] - tail[0]
    dy     = tip[1] - tail[1]
    length = math.hypot(dx, dy)
    nx, ny = (-dy / length, dx / length) if length > 0.5 else (0.0, 1.0)

    lateral = random.uniform(-8.0, 8.0)
    mid_x   = (tail[0] + tip[0]) / 2.0
    mid_y   = (tail[1] + tip[1]) / 2.0
    ctrl    = (int(round(mid_x + nx * lateral)),
               int(round(mid_y + ny * lateral)))

    shaft = _bezier_quadratic(tail, ctrl, tip, n_pts=16)
    draw.line(shaft, fill=color, width=width)

    near  = shaft[-2] if len(shaft) >= 2 else tail
    angle = math.atan2(tip[1] - near[1], tip[0] - near[0])
    for sign in (+1, -1):
        spread = head_half + random.gauss(0, 3.0)
        wa = angle + math.pi - math.radians(spread) * sign
        hx = int(tip[0] + (head_len + random.randint(-2, 2)) * math.cos(wa))
        hy = int(tip[1] + (head_len + random.randint(-2, 2)) * math.sin(wa))
        draw.line([tip, (hx, hy)], fill=color, width=width)


def draw_annotations(
    draw:         ImageDraw.Draw,
    text_lines:   list,
    features:     dict,
    canvas_wh:    tuple,
    color:        tuple,
    boundary_top: "tuple | None" = None,
) -> None:
    """
    Draws only the "Segmentation Mask" dashed boundary label + arrow.
    """
    W, H   = canvas_wh
    margin = 18
    font_sm = _resolve_font(18)
    HALO = {"stroke_width": 3, "stroke_fill": (255, 255, 255)}

    # ── Segmentation Mask label (top-right) + arrow to boundary top ──────────
    if boundary_top is not None:
        sm_text = "Segmentation Mask"
        tw      = len(sm_text) * 11
        lx      = W - tw - margin
        ly      = 12
        draw.text((lx, ly), sm_text, font=font_sm, fill=color, **HALO)
        _draw_arrow(draw, (lx + tw // 2, ly + 20), boundary_top, color, width=2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP C (replacement) — SIMPLE SCATTERED TEXTURE MARKS
# ─────────────────────────────────────────────────────────────────────────────

def draw_simple_texture_marks(
    canvas:    np.ndarray,
    seg_mask:  np.ndarray,
    color_bgr: tuple,
    n_strokes: int = 32,
    n_dots:    int = 12,
    stroke_len: int = 18,
) -> np.ndarray:
    """
    Scatters a small number of short straight pen strokes and dots across the
    object to indicate physical texture.  Mimics the "few basic, scattered
    dots or simple, straight hatch marks" of a quick beginner sketch.

    No brightness/shadow logic — marks are placed purely at random within the
    segmentation mask so there is no implicit shading or tonal fill.
    """
    ys, xs = np.where(seg_mask > 0)
    if len(ys) == 0:
        return canvas

    rng = np.random.default_rng()

    # Short pen strokes
    idx = rng.choice(len(ys), size=min(n_strokes, len(ys)), replace=False)
    for i in idx:
        cx, cy  = int(xs[i]), int(ys[i])
        angle   = math.radians(45.0 + random.uniform(-25.0, 25.0))
        half    = stroke_len // 2
        x1 = int(cx - half * math.cos(angle))
        y1 = int(cy - half * math.sin(angle))
        x2 = int(cx + half * math.cos(angle))
        y2 = int(cy + half * math.sin(angle))
        cv2.line(canvas, (x1, y1), (x2, y2), color_bgr, 1)

    # Small dots
    idx2 = rng.choice(len(ys), size=min(n_dots, len(ys)), replace=False)
    for i in idx2:
        cx, cy = int(xs[i]), int(ys[i])
        r = random.choice([1, 1, 2])
        cv2.circle(canvas, (cx, cy), r, color_bgr, -1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH-AWARE OCCLUSION EMPHASIS
# ─────────────────────────────────────────────────────────────────────────────

def draw_occlusion_edges(
    canvas:    np.ndarray,
    seg_mask:  np.ndarray,
    depth_path: str,
    color_bgr: tuple,
    thickness: int = 3,
) -> np.ndarray:
    """
    Uses the depth map to find occlusion boundaries — places where a near
    surface passes in front of a far surface.  These depth discontinuity
    edges are drawn thicker to communicate which fold is in front.

    The depth map gradient magnitude is high exactly at occlusion edges
    (sharp depth jump), so we threshold the gradient, mask it to the object
    interior, and draw the result as bold lines.
    """
    if not os.path.exists(depth_path):
        return canvas

    depth = np.load(depth_path).astype(np.float32)
    h, w  = canvas.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h))

    # Zero out background depth so it doesn't create false edges at silhouette
    depth[seg_mask == 0] = 0.0

    # Smooth slightly to suppress noise, then compute gradient magnitude
    depth_smooth = cv2.GaussianBlur(depth, (5, 5), 1.0)
    gx = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # Normalise and threshold — only the sharpest depth jumps (occlusion)
    mx = grad.max()
    if mx < 1e-6:
        return canvas
    grad_u8 = (grad / mx * 255).astype(np.uint8)

    # Adaptive threshold: top 5% of gradient values within the object
    obj_grad = grad_u8[seg_mask > 0]
    if obj_grad.size == 0:
        return canvas
    thresh = float(np.percentile(obj_grad, 95))
    _, occ_mask = cv2.threshold(grad_u8, int(thresh), 255, cv2.THRESH_BINARY)

    # Restrict to object interior (eroded slightly to avoid silhouette)
    eroded = cv2.erode(seg_mask, np.ones((7, 7), np.uint8), iterations=1)
    occ_mask = cv2.bitwise_and(occ_mask, eroded)

    # Dilate the occlusion edges to make them bold
    occ_mask = cv2.dilate(occ_mask, np.ones((thickness, thickness), np.uint8), iterations=1)
    canvas[occ_mask > 0] = color_bgr
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SECTION VOLUME ARCS
# ─────────────────────────────────────────────────────────────────────────────

def draw_cross_section_arcs(
    canvas:     np.ndarray,
    seg_mask:   np.ndarray,
    edges:      np.ndarray,
    color_bgr:  tuple,
    n_arcs:     int   = 3,
    thickness:  int   = 2,
) -> np.ndarray:
    """
    In smooth empty tube-like sections of the silhouette, draws short curved
    arcs crossing the width to suggest cylindrical volume.

    Strategy:
      1. Find the skeleton (medial axis) of the segmentation mask
      2. Sample points along the skeleton that are far from existing edge pixels
      3. At each sample, compute the local silhouette width and perpendicular
         direction, then draw a slight arc across the tube
    """
    h, w = canvas.shape[:2]

    # Approximate the medial axis via distance transform
    dist = cv2.distanceTransform(seg_mask, cv2.DIST_L2, 5)

    # Thin to skeleton: keep only local-maxima ridgeline of the distance field
    dilated_dist = cv2.dilate(dist, np.ones((7, 7), np.float32))
    skeleton_mask = ((dist > 5) & (dist >= dilated_dist - 0.5)).astype(np.uint8) * 255

    # Get skeleton pixel coordinates
    sk_ys, sk_xs = np.where(skeleton_mask > 0)
    if len(sk_ys) < 10:
        return canvas

    # Score each skeleton pixel: prefer points far from existing edges
    # (these are the "empty" tube sections that need volume hints)
    edge_dist = cv2.distanceTransform(
        255 - edges, cv2.DIST_L2, 5
    )
    scores = np.array([float(edge_dist[sk_ys[i], sk_xs[i]]) for i in range(len(sk_ys))])

    # Pick the top-N most edge-distant skeleton points, spaced apart
    order    = np.argsort(scores)[::-1]
    selected = []
    min_gap  = h // 6   # minimum spacing between arcs

    for idx in order:
        pt = (int(sk_xs[idx]), int(sk_ys[idx]))
        too_close = False
        for sel in selected:
            if math.hypot(pt[0] - sel[0], pt[1] - sel[1]) < min_gap:
                too_close = True
                break
        if not too_close:
            selected.append(pt)
        if len(selected) >= n_arcs:
            break

    if not selected:
        return canvas

    # For each selected point, find the local perpendicular direction and width
    # by scanning left/right of the skeleton point until hitting mask boundary
    for cx, cy in selected:
        # Compute local skeleton direction from nearby skeleton pixels
        nearby_mask = (
            (np.abs(sk_xs - cx) < 30) & (np.abs(sk_ys - cy) < 30)
        )
        nearby_x = sk_xs[nearby_mask]
        nearby_y = sk_ys[nearby_mask]
        if len(nearby_x) < 3:
            continue

        # PCA-like: dominant direction of nearby skeleton
        dx = float(nearby_x[-1] - nearby_x[0])
        dy = float(nearby_y[-1] - nearby_y[0])
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        # Tangent along skeleton
        tx, ty = dx / length, dy / length
        # Perpendicular (cross-section direction)
        nx, ny = -ty, tx

        # Walk perpendicular to find mask width
        half_width = 0
        for step in range(1, 200):
            px = int(cx + nx * step)
            py = int(cy + ny * step)
            if px < 0 or px >= w or py < 0 or py >= h or seg_mask[py, px] == 0:
                half_width = step
                break

        if half_width < 8:
            continue

        # Draw a slight arc: quadratic bezier with a small sag
        arc_half = half_width - 4   # slight inset from silhouette
        p0 = (int(cx - nx * arc_half), int(cy - ny * arc_half))
        p2 = (int(cx + nx * arc_half), int(cy + ny * arc_half))
        # Control point: offset along the skeleton tangent for a gentle curve
        sag = arc_half * 0.25 + random.uniform(-2, 2)
        p1 = (int(cx + tx * sag), int(cy + ty * sag))

        pts = _bezier_quadratic(p0, p1, p2, n_pts=14)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color_bgr, thickness)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_sketch(
    render_path:  str,
    obj_name:     str,
    texture_type: str,
    keyword:      str,
) -> np.ndarray:
    """
    Full four-step pipeline.
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
    seg_mask = get_object_mask(img_bgr, alpha=alpha)

    # ── A: Edge detection (interior structural lines) ────────────────────────
    edges = detect_edges(img_bgr, seg_mask=seg_mask, normal_path=normal_path)

    # Thicken interior fold lines so they read as confident pen strokes
    # matching the weight of the rest of the drawing (not wispy Canny pixels).
    edges_thick = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    canvas[edges_thick > 0] = SKETCH_BGR

    # ── A+: Depth-aware occlusion emphasis ────────────────────────────────────
    # Bold lines where a near surface passes in front of a far one.
    canvas = draw_occlusion_edges(canvas, seg_mask, depth_path, SKETCH_BGR,
                                  thickness=3)

    # ── A++: Wobbly outer silhouette from segmentation mask contour ──────────
    outer_cnts, _ = cv2.findContours(
        seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if outer_cnts:
        draw_wobbly_contour(
            canvas,
            max(outer_cnts, key=cv2.contourArea),
            SKETCH_BGR, base_thickness=3, wobble_amp=2.5, wobble_freq=2,
        )

    # ── A+++: Cross-section volume arcs in empty tube sections ───────────────
    canvas = draw_cross_section_arcs(canvas, seg_mask, edges, SKETCH_BGR,
                                     n_arcs=3, thickness=2)

    # ── C: Mid-tone wool texture stippling (dots + squiggles) ────────────────
    gray_img   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    obj_pixels = gray_img[seg_mask > 0]
    if obj_pixels.size > 0:
        lo_pct = float(np.percentile(obj_pixels, 35))
        hi_pct = float(np.percentile(obj_pixels, 70))
        mid_mask = (
            (gray_img >= lo_pct) & (gray_img <= hi_pct) & (seg_mask > 0)
        ).astype(np.uint8) * 255
    else:
        mid_mask = np.zeros_like(seg_mask)

    # Convert to PIL so we can draw vector annotations on the same surface.
    pil  = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    draw_wool_texture(pil, mid_mask, SKETCH_RGB, density=0.004)

    # ── D: Dashed segmentation boundary + annotation ─────────────────────────
    boundary_top = draw_organic_dashed_boundary(
        draw, seg_mask, SKETCH_RGB,
        dilation=40, dash_px=18, gap_px=9, line_width=2,
    )
    draw_annotations(
        draw, [], {},
        canvas_wh=(w, h),
        color=SKETCH_RGB,
        boundary_top=boundary_top,
    )
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Minimal ink-bleed softening — keeps lines crisp but removes aliasing
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
else:
    records = []
    print(f"[WARNING] metadata.jsonl not found at {METADATA_PATH}.")
    print("  Run generate_dataset.py first, then re-run this script.")

for meta in records:
    frame_str   = meta["frame"]
    render_path = os.path.join(
        DATASET_DIR, meta.get("file_name", f"renders/render_{frame_str}.png")
    )
    out_path = os.path.join(CONDITION_DIR, f"conditioning_{frame_str}.png")

    if os.path.exists(out_path):
        print(f"  [{frame_str}] Skipping (already exists)")
        continue

    raw_text     = meta.get("text", "")
    keyword      = meta.get("keyword")      or "texture pattern"
    texture_type = meta.get("texture_type") or "Wool"
    obj_name     = "Wool Scarf" if "wool" in raw_text.lower() else "Cloth"

    try:
        sketch = generate_sketch(render_path, obj_name, texture_type, keyword)
        cv2.imwrite(out_path, sketch)
        print(f"  [{frame_str}] ✓  → {out_path}")
    except Exception as e:
        print(f"  [{frame_str}] [ERROR] {e}")
