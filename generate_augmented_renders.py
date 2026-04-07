"""
generate_augmented_renders.py
------------------------------
Produce augmented variants of existing renders by applying 2-D affine
transforms (rotation + translation) consistently across:
  • render PNG  (BGRA, background stays gray)
  • mask PNG    (grayscale, border → 0)
  • depth .npy  (float32, border → 0)
  • normals .npy (float32 H×W×3, border → 0)

Each existing frame gets NUM_AUGMENTS new variants, saved as sequential
frame IDs continuing from the last existing frame.

Usage:
  python generate_augmented_renders.py [--augments N] [--seed S]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random

import cv2
import numpy as np

from cloth_pipeline.paths import (
    DATASET_DIR,
    DEPTH_DIR,
    MASKS_DIR,
    METADATA_PATH,
    NORMALS_DIR,
    RENDERS_DIR,
)


# Gray background used when Mitsuba composited renders (≈RGB 209, 209, 209 → BGR same)
_GRAY_BG = (209, 209, 209)


def _warp_bgr(img: np.ndarray, M: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_GRAY_BG,
    )


def _warp_gray(img: np.ndarray, M: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _warp_depth(arr: np.ndarray, M: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.warpAffine(
        arr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _warp_normals(arr: np.ndarray, M: np.ndarray, h: int, w: int) -> np.ndarray:
    """Warp each channel separately; border pixels get zero normals."""
    out = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        out[:, :, c] = cv2.warpAffine(
            arr[:, :, c], M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
    return out


def _random_affine(h: int, w: int, *, angle_range: float, tx_range: float,
                   ty_range: float, rng: random.Random) -> np.ndarray:
    """Build a 2×3 affine matrix: rotate around centre then translate."""
    angle = rng.uniform(-angle_range, angle_range)
    tx    = rng.uniform(-tx_range, tx_range)
    ty    = rng.uniform(-ty_range, ty_range)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return M


def augment_frame(
    meta: dict,
    new_frame_str: str,
    *,
    angle_range: float = 25.0,
    tx_range: float = 40.0,
    ty_range: float = 30.0,
    rng: random.Random,
) -> dict | None:
    """
    Apply a random affine to one frame's data, write augmented files,
    and return updated metadata dict (or None on error).
    """
    orig_frame = meta["frame"]

    render_path = os.path.join(DATASET_DIR, meta["file_name"])
    if not os.path.exists(render_path):
        print(f"  [SKIP] render not found: {render_path}")
        return None

    img_raw = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        print(f"  [SKIP] could not read render: {render_path}")
        return None

    h, w = img_raw.shape[:2]
    M = _random_affine(h, w, angle_range=angle_range,
                       tx_range=tx_range, ty_range=ty_range, rng=rng)

    angle_used = float(np.degrees(np.arctan2(-M[0, 1], M[0, 0])))
    tx_used    = float(M[0, 2])
    ty_used    = float(M[1, 2])

    # ── Render ───────────────────────────────────────────────────────────────
    new_render_name = f"renders/render_{new_frame_str}.png"
    new_render_path = os.path.join(RENDERS_DIR, f"render_{new_frame_str}.png")

    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        bgr  = img_raw[:, :, :3]
        alph = img_raw[:, :, 3]
        bgr_w  = _warp_bgr(bgr, M, h, w)
        alph_w = _warp_gray(alph, M, h, w)
        out_render = np.dstack([bgr_w, alph_w])
    else:
        out_render = _warp_bgr(img_raw, M, h, w)

    cv2.imwrite(new_render_path, out_render)

    # ── Mask ─────────────────────────────────────────────────────────────────
    new_mask_name = f"masks/mask_{new_frame_str}.png"
    new_mask_path = os.path.join(MASKS_DIR, f"mask_{new_frame_str}.png")

    mask_rel = meta.get("mask_image")
    mask_src  = (
        os.path.join(DATASET_DIR, mask_rel)
        if mask_rel
        else os.path.join(MASKS_DIR, f"mask_{orig_frame}.png")
    )
    if os.path.exists(mask_src):
        mask_img = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            cv2.imwrite(new_mask_path, _warp_gray(mask_img, M, h, w))

    # ── Depth ─────────────────────────────────────────────────────────────────
    new_depth_name = f"depth/depth_{new_frame_str}.npy"
    new_depth_path = os.path.join(DEPTH_DIR, f"depth_{new_frame_str}.npy")

    depth_src = os.path.join(DATASET_DIR, meta.get("depth_image", f"depth/depth_{orig_frame}.npy"))
    if os.path.exists(depth_src):
        depth_arr = np.load(depth_src).astype(np.float32)
        np.save(new_depth_path, _warp_depth(depth_arr, M, h, w))

    # ── Normals ───────────────────────────────────────────────────────────────
    new_normals_name = f"normals/normals_{new_frame_str}.npy"
    new_normals_path = os.path.join(NORMALS_DIR, f"normals_{new_frame_str}.npy")

    normals_src = os.path.join(DATASET_DIR, meta.get("normals_image", f"normals/normals_{orig_frame}.npy"))
    if os.path.exists(normals_src):
        normals_arr = np.load(normals_src).astype(np.float32)
        if normals_arr.ndim == 3:
            np.save(new_normals_path, _warp_normals(normals_arr, M, h, w))

    # ── Build augmented metadata ──────────────────────────────────────────────
    new_meta = copy.deepcopy(meta)
    new_meta["frame"]             = new_frame_str
    new_meta["file_name"]         = new_render_name
    new_meta["depth_image"]       = new_depth_name
    new_meta["normals_image"]     = new_normals_name
    new_meta["mask_image"]        = new_mask_name
    new_meta["conditioning_image"] = f"conditioning/conditioning_{new_frame_str}.png"
    new_meta["augmented_from"]    = orig_frame
    new_meta["augment_params"]    = {
        "angle_deg": round(angle_used, 2),
        "tx_px":     round(tx_used, 2),
        "ty_px":     round(ty_used, 2),
    }

    return new_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D augmentation of existing renders.")
    parser.add_argument("--augments", type=int, default=2,
                        help="Number of augmented variants per source frame (default: 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42).")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Read existing metadata
    if not os.path.exists(METADATA_PATH):
        print(f"[ERROR] {METADATA_PATH} not found. Run generate_dataset.py first.")
        raise SystemExit(1)

    with open(METADATA_PATH) as f:
        source_records = [json.loads(ln) for ln in f if ln.strip()]

    # Find the highest existing frame number to continue from
    existing_frames = [int(r["frame"]) for r in source_records]
    next_frame_id   = max(existing_frames) + 1

    new_records: list[dict] = []

    for meta in source_records:
        orig = meta["frame"]
        print(f"\n  Source frame {orig} → generating {args.augments} augment(s)")
        for aug_i in range(args.augments):
            new_frame_str = f"{next_frame_id:04d}"
            result = augment_frame(
                meta, new_frame_str,
                angle_range=25.0,
                tx_range=40.0,
                ty_range=30.0,
                rng=rng,
            )
            if result is not None:
                new_records.append(result)
                params = result["augment_params"]
                print(f"    [{new_frame_str}] angle={params['angle_deg']:+.1f}° "
                      f"tx={params['tx_px']:+.0f}px ty={params['ty_px']:+.0f}px")
            next_frame_id += 1

    # Append new records to metadata.jsonl
    with open(METADATA_PATH, "a") as f:
        for rec in new_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✓ Added {len(new_records)} augmented frames to {METADATA_PATH}")
    print("Next: run  python generate_sketches.py --output-dir dataset/conditioning_v18")


if __name__ == "__main__":
    main()
