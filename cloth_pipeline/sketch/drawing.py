"""Wobbly contours, wool stippling, arrows, and text annotations."""

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cloth_pipeline.sketch.constants import resolve_font

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
    Full annotation layout matching the Zahra / fashion-flat sketch grammar:

      • Text block       (top-left, font 22): material name + colour, texture
                         type, keyword.
      • "Segmentation    (top-centre, font 18): arrow pointing to the top of
        Mask"             the dashed rectangular boundary.
      • "Highlight"      (font 18): small outlined circle at the highlight
                         centroid + label at right canvas edge.
      • "Shadow"         (font 22): label at left canvas edge, arrow to the
                         hatched shadow zone.
    """
    W, H    = canvas_wh
    margin  = 18
    font_lg = resolve_font(22)
    font_sm = resolve_font(18)
    _font_xs = resolve_font(14)

    # All text uses stroke_width=3 with a white stroke_fill so each label
    # carries its own tight halo.  This keeps every annotation legible even
    # when it crosses a dashed line or a garment fold without erasing any
    # part of the underlying drawing (no opaque backing rectangles).
    HALO = {"stroke_width": 3, "stroke_fill": (255, 255, 255)}

    # y below the text block — used to keep Shadow label clear of the text
    y_below_text = margin + len(text_lines) * 20

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

    # ── Highlight: wobbly circle at centroid + label at right edge ───────────
    h_pt = features.get("highlight")
    if h_pt:
        _draw_wobbly_circle(draw, h_pt, radius=18, color=color, width=2)
        _right_label("Highlight", h_pt, min_y=80, max_y=H - 60)

    # ── Shadow: label at left canvas edge, arrow points right ────────────────
    s_pt = features.get("shadow")
    if s_pt:
        sx, sy = s_pt
        lx = margin
        ly = max(y_below_text + 8, min(H - 50, sy - 12))
        draw.text((lx, ly), "Shadow", font=font_lg, fill=color, **HALO)
        _draw_arrow(draw, (lx + 84, ly + 13), (sx, sy), color, width=2)

    # ── 3-line text block (top-left) — disabled ──────────────────────────────
    # y = margin
    # for line in text_lines:
    #     draw.text((margin, y), line, font=_font_xs, fill=color, **HALO)
    #     y += 20

