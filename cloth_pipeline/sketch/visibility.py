"""Camera-facing mask from depth + world-space shading normals (Mitsuba AOVs)."""

from __future__ import annotations

import os

import cv2
import numpy as np


def load_world_normals(path: str, h: int, w: int) -> np.ndarray | None:
    """
    Load (H, W, 3) world normals; ``.npy`` float [-1, 1], ``.png`` as
    ``0.5 + 0.5 * n`` per BGR channel.
    """
    if not path or not os.path.isfile(path):
        return None
    if path.endswith(".png"):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.ndim == 2:
            raw = raw[:, :, np.newaxis]
        n = raw[:, :, :3].astype(np.float32) / 255.0
        normals = n * 2.0 - 1.0
    else:
        normals = np.load(path).astype(np.float32)
    if normals.ndim == 2:
        normals = normals[:, :, np.newaxis]
    if normals.shape[2] < 3:
        return None
    normals = normals[:, :, :3]
    if normals.shape[:2] != (h, w):
        normals = cv2.resize(normals, (w, h), interpolation=cv2.INTER_LINEAR)
    nn = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(nn > 1e-6, normals / (nn + 1e-12), 0.0)
    return normals


def ray_directions_world(
    cam_origin: list | tuple,
    cam_target: list | tuple,
    fov_deg: float,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Unit direction from ``cam_origin`` toward the scene through each pixel
    (perspective ``look_at`` with ``up=[0,1,0]``, same framing as Stage 1).
    """
    O = np.asarray(cam_origin, dtype=np.float64).reshape(3)
    T = np.asarray(cam_target, dtype=np.float64).reshape(3)
    F = T - O
    fn = np.linalg.norm(F)
    if fn < 1e-12:
        return np.zeros((h, w, 3), dtype=np.float32)
    F = F / fn
    up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    R = np.cross(up_w, F)
    rn = np.linalg.norm(R)
    if rn < 1e-8:
        R = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        R = R / rn
    U = np.cross(F, R)
    aspect = w / max(h, 1)
    tan_half = float(np.tan(np.radians(fov_deg) * 0.5))
    uu = np.arange(w, dtype=np.float64)
    vv = np.arange(h, dtype=np.float64)
    u_grid, v_grid = np.meshgrid(uu, vv)
    ndc_x = (u_grid + 0.5) / float(w) * 2.0 - 1.0
    ndc_y = -((v_grid + 0.5) / float(h) * 2.0 - 1.0)
    d = (
        F
        + ndc_x[..., np.newaxis] * (tan_half * aspect) * R
        + ndc_y[..., np.newaxis] * tan_half * U
    )
    dn = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / (dn + 1e-12)
    return d.astype(np.float32)


def camera_facing_mask(
    normals_hw3: np.ndarray,
    ray_dirs_world: np.ndarray,
    depth: np.ndarray,
    *,
    min_ndot: float = 0.04,
    seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-pixel True where the shading normal faces the camera.

    Rays go camera → hit; the vector **toward** the camera is ``-ray_dirs_world``.
    We require ``dot(N, v_toward_cam) > min_ndot`` with ``N`` outward.

    ``depth`` gates invalid / background pixels (finite, positive).
    If ``seg_mask`` is given, orientation is corrected using cloth pixels only
    (median ``ndot`` on the garment); Mitsuba ``sh_normal`` can disagree with
    our ray basis until flipped.
    """
    v_to_cam = -ray_dirs_world
    normals = normals_hw3
    valid = np.isfinite(depth) & (depth > 1e-6)
    if seg_mask is not None:
        cloth = (seg_mask > 0) & valid
        if np.any(cloth):
            ndot_probe = np.sum(normals * v_to_cam, axis=-1)
            med = float(np.median(ndot_probe[cloth]))
            if med < 0.0:
                normals = -normals
    ndot = np.sum(normals * v_to_cam, axis=-1)
    ok = ndot > min_ndot
    ok = ok & valid
    if seg_mask is not None:
        ok = ok & (seg_mask > 0)
    return ok


def visibility_mask_u8(
    normals_path: str,
    depth_path: str,
    cam_origin: list | tuple,
    cam_target: list | tuple,
    fov_deg: float,
    h: int,
    w: int,
    *,
    min_ndot: float = 0.04,
    seg_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Returns ``uint8`` mask 255 = camera-visible cloth for highlight/shadow markers,
    or ``None`` if inputs are missing or invalid.

    ``seg_mask`` (optional) restricts the test to garment pixels and fixes
    one-sided normal orientation from the median sign on cloth.
    """
    if not os.path.isfile(depth_path):
        return None
    normals = load_world_normals(normals_path, h, w)
    if normals is None:
        return None
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    dirs = ray_directions_world(cam_origin, cam_target, fov_deg, w, h)
    facing = camera_facing_mask(
        normals, dirs, depth, min_ndot=min_ndot, seg_mask=seg_mask
    )
    return (facing.astype(np.uint8) * 255)
