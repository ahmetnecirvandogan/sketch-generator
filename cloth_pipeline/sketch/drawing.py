"""Wobbly contours, material-specific mid-tone marks, arrows, and text."""

from __future__ import annotations

import math
import os
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cloth_pipeline.sketch.constants import resolve_font


def draw_occlusion_edges(
    canvas:     np.ndarray,
    seg_mask:   np.ndarray,
    depth_path: str,
    color_bgr:  tuple,
    thickness:  int = 2,
) -> np.ndarray:
    """Bold lines at depth discontinuities — where a near fold passes in front
    of a far one.  Uses top-5% threshold globally, but relaxes to top-15% in
    the lower half of the object so tail-separation boundaries get drawn even
    when the dominant knot area outranks them globally."""
    if not os.path.exists(depth_path):
        return canvas
    h, w = canvas.shape[:2]
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h))
    depth[seg_mask == 0] = 0.0
    depth_s = cv2.GaussianBlur(depth, (5, 5), 1.0)
    gx = cv2.Sobel(depth_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_s, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    mx = grad.max()
    if mx < 1e-6:
        return canvas
    grad_u8 = (grad / mx * 255).astype(np.uint8)
    eroded = cv2.erode(seg_mask, np.ones((7, 7), np.uint8), iterations=1)

    # Split object into upper and lower halves at the median y of object pixels.
    ys_obj, _ = np.where(seg_mask > 0)
    y_mid = int(np.median(ys_obj)) if len(ys_obj) > 0 else h // 2

    upper_region = np.zeros_like(seg_mask)
    upper_region[:y_mid, :] = eroded[:y_mid, :]
    lower_region = np.zeros_like(seg_mask)
    lower_region[y_mid:, :] = eroded[y_mid:, :]

    occ_mask = np.zeros_like(grad_u8)
    # Stricter percentiles + thinner dilate reduce false gray slabs on noisy depth (e.g. 32).
    for region, pct in ((upper_region, 97), (lower_region, 90)):
        obj_grad = grad_u8[region > 0]
        if obj_grad.size == 0:
            continue
        thresh = float(np.percentile(obj_grad, pct))
        _, tmp = cv2.threshold(grad_u8, int(thresh), 255, cv2.THRESH_BINARY)
        occ_mask = cv2.bitwise_or(occ_mask, cv2.bitwise_and(tmp, region))

    dk = max(1, min(thickness, 2))
    occ_mask = cv2.dilate(occ_mask, np.ones((dk, dk), np.uint8), iterations=1)
    canvas[occ_mask > 0] = color_bgr
    return canvas


def draw_depth_layer_boundary(
    canvas:     np.ndarray,
    seg_mask:   np.ndarray,
    depth_path: str,
    color_bgr:  tuple,
    thickness:  int = 2,
) -> np.ndarray:
    """Draws the boundary between the near and far depth layers in the lower
    half of the object — explicitly marking where two overlapping cloth tails
    separate even when the segmentation mask merges them into one blob."""
    if not os.path.exists(depth_path):
        return canvas
    h, w = canvas.shape[:2]
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h))

    ys_obj, _ = np.where(seg_mask > 0)
    if len(ys_obj) == 0:
        return canvas
    y_split = int(np.percentile(ys_obj, 58))

    lower_mask = np.zeros_like(seg_mask)
    lower_mask[y_split:, :] = seg_mask[y_split:, :]
    lower_depths = depth[lower_mask > 0]
    if lower_depths.size < 50:
        return canvas

    depth_s = cv2.GaussianBlur(depth, (5, 5), 1.0)
    gx = cv2.Sobel(depth_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_s, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_vals = grad[lower_mask > 0]
    if grad_vals.size < 50:
        return canvas
    grad_cut = float(np.percentile(grad_vals, 84))
    strong_grad = ((grad >= grad_cut) & (lower_mask > 0)).astype(np.uint8) * 255

    # Split lower region into near/far at the median depth.
    median_d = float(np.median(lower_depths))
    front = ((depth < median_d) & (lower_mask > 0)).astype(np.uint8) * 255
    back  = ((depth >= median_d) & (lower_mask > 0)).astype(np.uint8) * 255

    # Boundary = pixels of back layer that are adjacent to front layer, but
    # only where there is an actual depth discontinuity. This avoids drawing an
    # arbitrary "median split" line through a smoothly changing body section.
    front_dilated = cv2.dilate(front, np.ones((3, 3), np.uint8), iterations=2)
    boundary = cv2.bitwise_and(front_dilated, back)
    boundary = cv2.bitwise_and(boundary, lower_mask)
    boundary = cv2.bitwise_and(
        boundary,
        cv2.dilate(strong_grad, np.ones((3, 3), np.uint8), iterations=1),
    )
    if np.count_nonzero(boundary) < 20:
        return canvas

    ys_l, xs_l = np.where(lower_mask > 0)
    obj_w = int(xs_l.max() - xs_l.min() + 1)
    obj_h = int(ys_l.max() - ys_l.min() + 1)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(boundary, connectivity=8)
    clean = np.zeros_like(boundary)
    for lbl_id in range(1, n_lbl):
        x, y, bw, bh, area = stats[lbl_id]
        if area < 14:
            continue
        if bw > 0.58 * obj_w and bh < max(12, int(0.10 * obj_h)):
            continue
        clean[lbl == lbl_id] = 255
    boundary = clean
    if np.count_nonzero(boundary) < 12:
        return canvas
    boundary = cv2.dilate(boundary, np.ones((thickness, thickness), np.uint8))
    canvas[boundary > 0] = color_bgr
    return canvas


def draw_cross_section_arcs(
    canvas:    np.ndarray,
    seg_mask:  np.ndarray,
    edges:     np.ndarray,
    color_bgr: tuple,
    n_arcs:    int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """Gentle bezier arcs across the tube width in regions with no existing
    edges — communicates cylindrical volume in otherwise empty sections."""
    h, w = canvas.shape[:2]
    arc_mask = np.zeros((h, w), dtype=np.uint8)
    dist = cv2.distanceTransform(seg_mask, cv2.DIST_L2, 5)
    dilated_dist = cv2.dilate(dist, np.ones((7, 7), np.float32))
    skeleton_mask = ((dist > 5) & (dist >= dilated_dist - 0.5)).astype(np.uint8) * 255
    sk_ys, sk_xs = np.where(skeleton_mask > 0)
    if len(sk_ys) < 10:
        return canvas
    ys_obj, _ = np.where(seg_mask > 0)
    if ys_obj.size == 0:
        return canvas
    y_top = int(ys_obj.min())
    y_bottom = int(ys_obj.max())
    top_guard = int(y_top + 0.22 * max(1, (y_bottom - y_top)))
    valid = sk_ys >= top_guard
    sk_ys = sk_ys[valid]
    sk_xs = sk_xs[valid]
    if len(sk_ys) < 10:
        return canvas
    edge_dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    scores = np.array([float(edge_dist[sk_ys[i], sk_xs[i]]) for i in range(len(sk_ys))])
    order = np.argsort(scores)[::-1]
    selected = []
    min_gap = h // 6
    for idx in order:
        if scores[idx] < 10.0:
            break
        pt = (int(sk_xs[idx]), int(sk_ys[idx]))
        if all(math.hypot(pt[0] - s[0], pt[1] - s[1]) >= min_gap for s in selected):
            selected.append(pt)
        if len(selected) >= n_arcs:
            break
    for cx, cy in selected:
        nearby = (np.abs(sk_xs - cx) < 30) & (np.abs(sk_ys - cy) < 30)
        nx_pts, ny_pts = sk_xs[nearby], sk_ys[nearby]
        if len(nx_pts) < 3:
            continue
        dx, dy = float(nx_pts[-1] - nx_pts[0]), float(ny_pts[-1] - ny_pts[0])
        length = math.hypot(dx, dy)
        if length < 1:
            continue
        tx, ty = dx / length, dy / length
        nx_dir, ny_dir = -ty, tx
        half_width = 0
        for step in range(1, 200):
            px, py = int(cx + nx_dir * step), int(cy + ny_dir * step)
            if px < 0 or px >= w or py < 0 or py >= h or seg_mask[py, px] == 0:
                half_width = step
                break
        if half_width < 9:
            continue
        arc_half = half_width - 4
        p0 = (int(cx - nx_dir * arc_half), int(cy - ny_dir * arc_half))
        p2 = (int(cx + nx_dir * arc_half), int(cy + ny_dir * arc_half))
        sag = arc_half * 0.25 + random.uniform(-2, 2)
        p1 = (int(cx + tx * sag), int(cy + ty * sag))
        pts = _bezier_quadratic(p0, p1, p2, n_pts=14)
        for i in range(len(pts) - 1):
            cv2.line(arc_mask, pts[i], pts[i + 1], 255, thickness)
    arc_mask = cv2.bitwise_and(arc_mask, seg_mask)
    canvas[arc_mask > 0] = color_bgr
    return canvas


def _resample_closed_polyline(pts: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced samples along a closed polygon (pts not duplicated at end)."""
    n = pts.shape[0]
    if n < 3:
        return pts.astype(np.float64)
    cum = [0.0]
    for i in range(n):
        j = (i + 1) % n
        cum.append(
            cum[-1]
            + float(np.hypot(pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]))
        )
    total = cum[-1]
    if total < 1.0:
        return pts.astype(np.float64)
    n_new = max(32, int(round(total / step)))
    out = np.empty((n_new, 2), dtype=np.float64)
    for m in range(n_new):
        d = (m / n_new) * total
        k = 0
        while k < n and cum[k + 1] < d:
            k += 1
        k = min(k, n - 1)
        seg_len = cum[k + 1] - cum[k]
        if seg_len < 1e-9:
            t = 0.0
        else:
            t = (d - cum[k]) / seg_len
        i, j = k, (k + 1) % n
        out[m, 0] = pts[i, 0] * (1.0 - t) + pts[j, 0] * t
        out[m, 1] = pts[i, 1] * (1.0 - t) + pts[j, 1] * t
    return out


def _smooth_closed_polyline(xy: np.ndarray, passes: int, window: int) -> np.ndarray:
    """Circular moving average — softens pixel staircase before wobble."""
    n = xy.shape[0]
    if n < window:
        return xy
    half = window // 2
    p = xy.astype(np.float64).copy()
    for _ in range(passes):
        q = np.zeros_like(p)
        for i in range(n):
            sx = sy = 0.0
            for j in range(-half, half + 1):
                sx += p[(i + j) % n, 0]
                sy += p[(i + j) % n, 1]
            q[i, 0] = sx / window
            q[i, 1] = sy / window
        p = q
    return p


def draw_wobbly_contour(
    canvas:         np.ndarray,
    contour:        np.ndarray,
    color_bgr:      tuple,
    base_thickness: float = 1.0,
    wobble_amp:     float = 1.0,
    wobble_freq:    int   = 2,
    sample_step:    float = 2.0,
) -> None:
    """
    One continuous closed stroke: resample, smooth, arc-length wobble, then draw.
    ``base_thickness`` may be fractional in (1, 2): full ink on a 1 px core and
    a (2−1) px halo blended toward white with weight (t−1), approximating t px
    (e.g. 1.5 → half-weight outer ring).  Otherwise ``round(t)`` is passed to
    ``cv2.polylines`` (LINE_AA).
    """
    pts_raw = contour.reshape(-1, 2).astype(np.float64)
    n_orig = pts_raw.shape[0]
    if n_orig < 3:
        return

    base = _resample_closed_polyline(pts_raw, sample_step)
    base = _smooth_closed_polyline(base, passes=2, window=5)

    n = base.shape[0]
    arc = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        arc[i] = arc[i - 1] + float(
            np.hypot(base[i, 0] - base[i - 1, 0], base[i, 1] - base[i - 1, 1])
        )
    arc_close = arc[-1] + float(
        np.hypot(base[0, 0] - base[-1, 0], base[0, 1] - base[-1, 1])
    )
    if arc_close < 1.0:
        return
    total_arc = float(arc_close)

    phases = [random.uniform(0.0, 2 * math.pi) for _ in range(3)]
    amps = [wobble_amp, wobble_amp * 0.35, wobble_amp * 0.15]
    freqs = [wobble_freq, wobble_freq * 2, wobble_freq * 3]

    def _d(s: float) -> float:
        tt = 2 * math.pi * s / total_arc
        return sum(a * math.sin(f * tt + p) for a, f, p in zip(amps, freqs, phases))

    final = np.empty_like(base)
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        dx = base[next_i, 0] - base[prev_i, 0]
        dy = base[next_i, 1] - base[prev_i, 1]
        length = math.hypot(dx, dy)
        nx, ny = (-dy / length, dx / length) if length > 0.5 else (0.0, 1.0)
        d = _d(arc[i])
        final[i, 0] = base[i, 0] + nx * d
        final[i, 1] = base[i, 1] + ny * d

    pts_i32 = np.round(final).astype(np.int32).reshape(-1, 1, 2)
    t = float(base_thickness)
    h, w = canvas.shape[:2]

    if 1.0 < t < 2.0:
        # Integer polylines only: emulate fractional width with core + soft halo.
        ring_w = float(t - 1.0)
        inner_m = np.zeros((h, w), np.uint8)
        outer_m = np.zeros((h, w), np.uint8)
        cv2.polylines(inner_m, [pts_i32], True, 255, 1, cv2.LINE_8)
        cv2.polylines(outer_m, [pts_i32], True, 255, 2, cv2.LINE_8)
        inner = inner_m > 0
        ring = (outer_m > 0) & ~inner
        canvas[inner] = color_bgr
        inv = 1.0 - ring_w
        ink = np.array(color_bgr, dtype=np.float64)
        halo = (ring_w * ink + inv * 255.0).astype(np.uint8)
        canvas[ring] = halo
    else:
        thick = max(1, min(255, int(round(t))))
        cv2.polylines(
            canvas,
            [pts_i32],
            isClosed=True,
            color=color_bgr,
            thickness=thick,
            lineType=cv2.LINE_AA,
        )


# Canonical fabric ids — keep in sync with FABRIC_PRESETS in render_loop.py
_MATERIAL_IDS = frozenset({
    "silk", "satin", "wool", "cotton", "velvet", "linen", "denim",
    "chiffon", "cashmere", "leather",
})


def normalize_material_type(
    material_type: Optional[str],
    material_label: str,
) -> str:
    """
    Map metadata ``material_type`` (preferred) or a human ``material_label``
    to a fabric id for mark vocabulary. Unknown → ``\"default\"``.
    """
    if material_type:
        k = str(material_type).strip().lower()
        if k in _MATERIAL_IDS:
            return k
    t = (material_label or "").strip().lower()
    t = t.replace(" texture", "").replace(" material", "")
    for name in sorted(_MATERIAL_IDS, key=len, reverse=True):
        if name in t:
            return name
    return "default"


def _mid_mask_sample_sites(mid_mask: np.ndarray, density: float, cap: int):
    ys, xs = np.where(mid_mask > 0)
    if len(ys) == 0:
        return None, 0
    n_marks = min(int(len(ys) * density), cap)
    if n_marks == 0:
        return None, 0
    idx = np.random.choice(len(ys), size=n_marks, replace=False)
    return idx, n_marks


def _draw_stipple_default(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    if random.random() < 0.55:
        r = random.choice([1, 1, 1, 2])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    else:
        pts = [
            (x + random.randint(-6, 6), y + random.randint(-3, 3))
            for _ in range(3)
        ]
        draw.line(pts, fill=color, width=1)


def _mark_wool(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Tight curls + fibre dots (crimped yarn)."""
    if random.random() < 0.4:
        r = random.choice([1, 2])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return
    r0 = random.uniform(3.0, 6.0)
    pts = []
    for k in range(5):
        ang = -0.8 + k * 0.55
        pts.append(
            (int(x + r0 * math.cos(ang)), int(y + r0 * math.sin(ang)))
        )
    draw.line(pts, fill=color, width=1)


def _mark_cashmere(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Softer, loftier loops than wool."""
    if random.random() < 0.5:
        r = random.choice([1, 2])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        return
    r0 = random.uniform(5.0, 9.0)
    pts = []
    for k in range(4):
        ang = -0.5 + k * 0.65
        pts.append(
            (int(x + r0 * math.cos(ang)), int(y + r0 * math.sin(ang)))
        )
    draw.line(pts, fill=color, width=1)


def _mark_cotton(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Matte yarn: small crosses + occasional dot."""
    if random.random() < 0.65:
        s = random.randint(2, 4)
        draw.line([(x - s, y - s), (x + s, y + s)], fill=color, width=1)
        draw.line([(x - s, y + s), (x + s, y - s)], fill=color, width=1)
    else:
        r = 1
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _mark_linen(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Slub weave: short parallel hash strokes."""
    ang = random.uniform(-0.5, 0.5)
    ca, sa = math.cos(ang), math.sin(ang)
    L = random.randint(5, 10)
    for off in (-2, 0, 2):
        ox, oy = -sa * off, ca * off
        x0, y0 = int(x + ox - ca * L / 2), int(y + oy - sa * L / 2)
        x1, y1 = int(x + ox + ca * L / 2), int(y + oy + sa * L / 2)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=1)


def _mark_denim(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Twill hint: short diagonal slashes."""
    sign = random.choice([-1.0, 1.0])
    ang = sign * math.pi / 4 + random.uniform(-0.12, 0.12)
    L = random.randint(8, 14)
    ca, sa = math.cos(ang), math.sin(ang)
    x0, y0 = int(x - ca * L / 2), int(y - sa * L / 2)
    x1, y1 = int(x + ca * L / 2), int(y + sa * L / 2)
    draw.line([(x0, y0), (x1, y1)], fill=color, width=1)


def _mark_silk(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Fluid lustre lines (long, smooth)."""
    n = random.randint(5, 8)
    pts = []
    base = random.uniform(0, 2 * math.pi)
    for k in range(n):
        t = k * 0.9
        pts.append(
            (
                int(x + t * 2.2 + 2.0 * math.sin(base + t * 0.7)),
                int(y + t * 1.1 + 1.5 * math.cos(base + t * 0.5)),
            )
        )
    draw.line(pts, fill=color, width=1)


def _mark_satin(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Longer, straighter glide strokes than silk."""
    n = random.randint(6, 10)
    ang = random.uniform(-0.35, 0.35)
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for k in range(n):
        d = k * 2.8 + random.uniform(-0.4, 0.4)
        pts.append((int(x + d * ca), int(y + d * sa)))
    draw.line(pts, fill=color, width=1)


def _mark_chiffon(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Light, airy waves."""
    n = random.randint(4, 7)
    pts = []
    for k in range(n):
        pts.append(
            (
                int(x + k * 2 + random.randint(-2, 2)),
                int(y + 3 * math.sin(k * 0.9) + random.randint(-1, 1)),
            )
        )
    draw.line(pts, fill=color, width=1)


def _mark_velvet(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Pile: short rays from a point."""
    n_ray = random.randint(5, 9)
    r = random.randint(3, 6)
    for _ in range(n_ray):
        ang = random.uniform(0, 2 * math.pi)
        x1 = int(x + r * math.cos(ang))
        y1 = int(y + r * math.sin(ang))
        draw.line([(x, y), (x1, y1)], fill=color, width=1)


def _mark_leather(draw: ImageDraw.Draw, x: int, y: int, color: tuple) -> None:
    """Irregular crease scratches."""
    n = random.randint(3, 5)
    cx, cy = float(x), float(y)
    pts = [(int(cx), int(cy))]
    for _ in range(n):
        cx += random.uniform(-5.0, 5.0)
        cy += random.uniform(-4.0, 4.0)
        pts.append((int(cx), int(cy)))
    draw.line(pts, fill=color, width=1)


_MATERIAL_MARK_CAP = {
    "default": 280,
    "silk": 200,
    "satin": 200,
    "chiffon": 160,
    "leather": 220,
    "velvet": 240,
    "wool": 280,
    "cashmere": 260,
    "cotton": 300,
    "linen": 260,
    "denim": 280,
}


def draw_material_marks(
    pil_img: Image.Image,
    mid_mask: np.ndarray,
    color: tuple,
    density: float,
    *,
    material_type: Optional[str] = None,
    material_label: str = "",
) -> None:
    """
    Mid-tone marks that vary by **BRDF fabric preset** (wool, linen, …).
    These marks encode fabric feel (material handle) in the sketch language.
    """
    key = normalize_material_type(material_type, material_label)
    cap = _MATERIAL_MARK_CAP.get(key, _MATERIAL_MARK_CAP["default"])
    idx, _ = _mid_mask_sample_sites(mid_mask, density, cap)
    if idx is None:
        return

    draw = ImageDraw.Draw(pil_img)
    ys, xs = np.where(mid_mask > 0)

    dispatch = {
        "wool": _mark_wool,
        "cashmere": _mark_cashmere,
        "cotton": _mark_cotton,
        "linen": _mark_linen,
        "denim": _mark_denim,
        "silk": _mark_silk,
        "satin": _mark_satin,
        "chiffon": _mark_chiffon,
        "velvet": _mark_velvet,
        "leather": _mark_leather,
    }
    marker = dispatch.get(key, _draw_stipple_default)

    for i in idx:
        marker(draw, int(xs[i]), int(ys[i]), color)


def draw_fabric_stipple(
    pil_img: Image.Image,
    mid_mask: np.ndarray,
    color: tuple,
    density: float = 0.006,
) -> None:
    """Backward-compatible: same as ``default`` material marks."""
    draw_material_marks(
        pil_img, mid_mask, color, density,
        material_type=None, material_label="",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Highlight / shadow region markers  (ASCII on canvas before padding & scale)
# ─────────────────────────────────────────────────────────────────────────────

# Same fontScale yields identical bbox height for ``*`` and ``#``, but ``*`` is
# narrower and reads smaller; bump it so both markers feel the same size.
_ASTERISK_VISUAL_SCALE = 1.25


def draw_region_markers(
    canvas:           np.ndarray,
    mask:             np.ndarray,
    marker:           str,
    color_bgr:        tuple,
    *,
    step:             int   = 26,
    marker_height_px: float = 14.0,
    thickness:        int   = 1,
) -> np.ndarray:
    """
    Draw a single-character label on a coarse grid over ``mask`` pixels so each
    spatial cell gets at most one marker (avoids solid fills on large regions).

    ``marker_height_px`` is the target cap height in pixels (from unit-scale
    metrics). ``*`` is scaled up slightly so it matches ``#`` visually.
    """
    if mask is None or mask.size == 0:
        return canvas
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return canvas

    buckets: dict[tuple[int, int], tuple[int, int]] = {}
    for y, x in zip(ys, xs):
        key = (x // step, y // step)
        if key not in buckets:
            buckets[key] = (x, y)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, ref_h), _ = cv2.getTextSize(marker, font, 1.0, thickness)
    font_scale = (
        marker_height_px / float(ref_h) if ref_h > 0 else 0.62
    )
    if marker == "*":
        font_scale *= _ASTERISK_VISUAL_SCALE

    for x, y in buckets.values():
        (tw, th), _ = cv2.getTextSize(marker, font, font_scale, thickness)
        org_x = int(x - tw / 2)
        org_y = int(y + th / 2)
        cv2.putText(
            canvas,
            marker,
            (org_x, org_y),
            font,
            font_scale,
            color_bgr,
            thickness,
            cv2.LINE_AA,
        )
    return canvas


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


def measure_top_left_text_pad(
    text_lines: list,
    *,
    margin: int = 18,
    line_step: int = 22,
    gutter: int = 20,
) -> tuple[int, int]:
    """
    Return ``(pad_left, pad_top)`` so a ``(h, w)`` sketch can be placed at
    ``(pad_left, pad_top)`` with no overlap with the text block drawn at
    ``(margin, margin)`` using ``font_sm`` (18) and halo stroke.
    """
    font = resolve_font(18)
    tmp = Image.new("RGB", (8, 8))
    dr = ImageDraw.Draw(tmp)
    max_r = margin
    max_b = margin
    y = margin
    stroke_slack = 10
    for line in text_lines:
        bbox = dr.textbbox((margin, y), line, font=font)
        max_r = max(max_r, bbox[2] + stroke_slack)
        max_b = max(max_b, bbox[3] + stroke_slack // 2)
        y += line_step
    pad_left = int(max_r + gutter)
    pad_top = int(max_b + gutter)
    return pad_left, pad_top


def draw_annotations(
    draw:         ImageDraw.Draw,
    text_lines:   list,
    features:     dict,
    canvas_wh:    tuple,
    color:        tuple,
    boundary_top: "tuple | None" = None,
    sketch_content_top: int = 0,
) -> None:
    """
    Full annotation layout matching the Zahra / fashion-flat sketch grammar:

      • Text block       (top-left, font 18): material, object + colour, keyword.
      • "Segmentation    (top-centre, font 18): arrow pointing to the top of
        Mask"             the dashed rectangular boundary.
      • Highlight / shadow are marked on the garment raster with ``*`` and ``#``
        (see ``draw_region_markers`` in the pipeline), not as separate arrow labels.

    ``sketch_content_top`` — y coordinate where the sketch raster begins; keeps
    captions aligned when the layout reserves a top band.
    """
    W, H    = canvas_wh
    margin  = 18
    font_lg = resolve_font(22)
    font_sm = resolve_font(18)

    # All text uses stroke_width=3 with a white stroke_fill so each label
    # carries its own tight halo.  This keeps every annotation legible even
    # when it crosses a dashed line or a garment fold without erasing any
    # part of the underlying drawing (no opaque backing rectangles).
    HALO = {"stroke_width": 3, "stroke_fill": (255, 255, 255)}

    # y below the text block — used to keep Shadow label clear of the text
    _line_h = 22
    y_below_text = margin + len(text_lines) * _line_h

    # ── Segmentation Mask label (top-right) + arrow to boundary top ──────────
    if boundary_top is not None:
        sm_text = "Segmentation Mask"
        tw      = len(sm_text) * 11
        lx      = W - tw - margin
        ly      = 12
        draw.text((lx, ly), sm_text, font=font_sm, fill=color, **HALO)
        _draw_arrow(draw, (lx + tw // 2, ly + 20), boundary_top, color, width=2)

    # ── Helper: label at right canvas edge, arrow points left to feature ─────
    def _right_label(text: str, feat: tuple, min_y: int, max_y: int) -> None:
        tw = len(text) * 11
        lx = W - tw - margin
        ly = max(min_y, min(max_y, feat[1] - 9))
        draw.text((lx, ly), text, font=font_sm, fill=color, **HALO)
        _draw_arrow(draw, (lx - 6, ly + 10), feat, color, width=2)

    # ── Text block (top-left): material, object/colour, keyword ──────────────
    y = margin
    for line in text_lines:
        draw.text((margin, y), line, font=font_sm, fill=color, **HALO)
        y += _line_h
