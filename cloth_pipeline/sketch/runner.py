"""Batch over metadata.jsonl → dataset/<mesh>/view_<idx>/sketch.png."""

from __future__ import annotations

import json
import os

import cv2

from cloth_pipeline.paths import BASE_DIR, METADATA_PATH
from cloth_pipeline.sketch.pipeline import generate_sketch


def _albedo_rel_from_meta(meta: dict) -> str | None:
    rel = meta.get("albedo_map") or meta.get("texture_file")
    return rel if rel else None


def _albedo_tiling_from_meta(meta: dict) -> tuple[float, float] | None:
    raw = meta.get("albedo_tiling") or meta.get("texture_tiling")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (float(raw[0]), float(raw[1]))
    return None


def _pattern_name_from_meta(meta: dict) -> str | None:
    return meta.get("pattern_name") or meta.get("texture_pattern")


def run_from_metadata() -> None:
    if not os.path.exists(METADATA_PATH):
        print(f"[WARNING] metadata.jsonl not found at {METADATA_PATH}.")
        print("  Run generate_dataset.py first, then re-run this script.")
        return

    with open(METADATA_PATH) as f:
        records = [json.loads(ln) for ln in f if ln.strip()]

    for meta in records:
        frame_str   = meta["frame"]
        render_path = os.path.join(
            BASE_DIR, meta.get("file_name", f"renders/render_{frame_str}.png")
        )

        sketch_rel = meta.get("sketch_path")
        if not sketch_rel:
            print(f"  [{frame_str}] [SKIP] sketch_path missing — re-run generate_dataset.py.")
            continue
        out_path = os.path.join(BASE_DIR, sketch_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            print(f"  [{frame_str}] Skipping (already exists)")
            continue

        tex_rel = _albedo_rel_from_meta(meta)
        albedo_map_path = (
            os.path.join(BASE_DIR, tex_rel) if tex_rel else None
        )
        mask_rel = meta.get("mask_image")
        alpha_mask_path = (
            os.path.join(BASE_DIR, mask_rel) if mask_rel else None
        )

        try:
            sketch = generate_sketch(
                render_path,
                alpha_mask_path=alpha_mask_path,
                albedo_map_path=albedo_map_path,
                albedo_tiling=_albedo_tiling_from_meta(meta),
                pattern_name=_pattern_name_from_meta(meta),
            )
            cv2.imwrite(out_path, sketch)
            print(f"  [{frame_str}] ✓  → {out_path}")
        except Exception as e:
            print(f"  [{frame_str}] [ERROR] {e}")
