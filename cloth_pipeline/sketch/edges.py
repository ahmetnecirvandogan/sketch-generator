"""HED / normal-gradient / bilateral Canny edge detection."""

import os

import cv2
import numpy as np

from cloth_pipeline.sketch.constants import HED_MODEL_DIR, HED_PROTO, HED_WEIGHTS

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


def _simplify_structural_components(
    edges: np.ndarray,
    seg_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Keep only the strongest connected edge components so crowded knot regions
    don't turn into texture. Upper-object detail is capped more aggressively
    than the hanging body, which preserves long drape lines while trimming the
    dense bundle of tiny parallel folds near the knot.
    """
    n_lbl, lbl, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if n_lbl <= 1:
        return edges

    upper_cut = None
    if seg_mask is not None:
        ys, _ = np.where(seg_mask > 0)
        if ys.size > 0:
            y0 = int(ys.min())
            y1 = int(ys.max())
            upper_cut = y0 + 0.39 * max(1, (y1 - y0))

    upper = []
    lower = []
    for lbl_id in range(1, n_lbl):
        x, y, w, h, area = stats[lbl_id]
        if area < 14:
            continue
        long_span = max(int(w), int(h))
        short_span = min(int(w), int(h))
        score = float(area) + 0.65 * float(long_span) - 0.20 * float(short_span)
        if long_span >= 34:
            score += 6.0
        bucket = upper if (upper_cut is not None and centroids[lbl_id][1] < upper_cut) else lower
        bucket.append((score, lbl_id, area, long_span))

    keep: set[int] = set()
    for bucket, cap, long_extra in ((upper, 6, 1), (lower, 9, 2)):
        bucket.sort(reverse=True)
        kept = 0
        kept_long = 0
        for score, lbl_id, area, long_span in bucket:
            is_long = long_span >= 52 and area >= 18
            if kept >= cap and not (is_long and kept_long < long_extra):
                continue
            if area < 20 and long_span < 28:
                continue
            keep.add(int(lbl_id))
            if kept < cap:
                kept += 1
            elif is_long:
                kept_long += 1

    if not keep:
        return edges

    out = np.zeros_like(edges)
    for lbl_id in keep:
        out[lbl == lbl_id] = 255
    return out


def _load_hed():
    global _hed_net
    if _hed_net is not None:
        return _hed_net
    if os.path.exists(HED_PROTO) and os.path.exists(HED_WEIGHTS):
        _hed_net = cv2.dnn.readNetFromCaffe(HED_PROTO, HED_WEIGHTS)
        print("[A] HED neural edge detector loaded.")
    else:
        print(
            f"[A] HED model not found in '{HED_MODEL_DIR}' — using bilateral Canny.\n"
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
         changes direction.  Fine fibre detail has near-zero normal gradient
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

        # Pre-smooth each channel with a mild Gaussian to suppress the
        # high-frequency micro-texture noise that survives even in the normal
        # map (tiny normal perturbations from the yarn fibres).  A ksize=7
        # kernel wipes those out while keeping the broad fold transitions.
        edge_acc = np.zeros((h, w), dtype=np.float32)
        for c in range(min(3, normals.shape[2])):
            ch_smooth = cv2.GaussianBlur(normals[:, :, c], (11, 11), 3.0)
            gx = cv2.Sobel(ch_smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(ch_smooth, cv2.CV_32F, 0, 1, ksize=3)
            edge_acc += gx ** 2 + gy ** 2
        edge_acc = np.sqrt(edge_acc)
        mx       = edge_acc.max()
        edge_u8  = (edge_acc / mx * 255).astype(np.uint8) if mx > 0 else np.zeros((h, w), np.uint8)
        # Keep the dominant folds while allowing a little more structure than
        # v9 so the cloth frame and turning volumes read more clearly.
        edges    = cv2.Canny(edge_u8, 86, 210)
        edges    = _simplify_structural_components(edges, seg_mask=seg_mask)
        print(f"[A] Normal-map edge detection active ({os.path.splitext(normal_path)[1]}, fold geometry only).")

    else:
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # d=15, sigmaColor=80: more aggressive than before to wipe out the
        # fine fibre detail in the RGB render that caused chaotic scratch noise.
        filtered = cv2.bilateralFilter(gray, d=15, sigmaColor=80, sigmaSpace=80)
        v        = float(np.median(filtered[filtered > 0])) if np.any(filtered > 0) else 128.0
        edges    = cv2.Canny(filtered,
                             int(max(0,   0.5 * v)),
                             int(min(255, 1.5 * v)))

    if seg_mask is not None:
        dilated = cv2.dilate(seg_mask, np.ones((5, 5), np.uint8), iterations=1)
        edges   = cv2.bitwise_and(edges, dilated)
    return edges
