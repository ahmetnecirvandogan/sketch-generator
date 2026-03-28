"""Batch over metadata.jsonl → conditioning images."""

from __future__ import annotations

import json
import os

import cv2

from cloth_pipeline.paths import (
    CONDITION_DIR,
    DATASET_DIR,
    METADATA_PATH,
    ensure_sketch_stage_dirs,
)
from cloth_pipeline.sketch.pipeline import generate_sketch


def _material_label_from_meta(meta: dict) -> str:
    """Human-readable fabric preset (BRDF category), not albedo pattern."""
    mt = meta.get("material_type")
    if mt:
        s = str(mt).replace("_", " ").strip()
        return s[0].upper() + s[1:] if s else "Fabric"
    legacy = meta.get("texture_type")
    if legacy:
        return (
            str(legacy)
            .replace(" texture", "")
            .replace(" Texture", "")
            .strip()
        )
    return "Wool"


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
    ensure_sketch_stage_dirs()

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

        raw_text = meta.get("text", "")
        keyword  = meta.get("keyword") or "fabric pattern"
        material_label = _material_label_from_meta(meta)
        obj_name = "Wool Scarf" if "wool" in raw_text.lower() else "Cloth"

        tex_rel = _albedo_rel_from_meta(meta)
        albedo_map_path = (
            os.path.join(DATASET_DIR, tex_rel) if tex_rel else None
        )

        try:
            sketch = generate_sketch(
                render_path,
                obj_name,
                material_label,
                keyword,
                albedo_map_path=albedo_map_path,
                albedo_tiling=_albedo_tiling_from_meta(meta),
                pattern_name=_pattern_name_from_meta(meta),
            )
            cv2.imwrite(out_path, sketch)
            print(f"  [{frame_str}] ✓  → {out_path}")
        except Exception as e:
            print(f"  [{frame_str}] [ERROR] {e}")
