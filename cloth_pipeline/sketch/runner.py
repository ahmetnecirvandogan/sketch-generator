"""Batch over metadata.jsonl → conditioning images."""

from __future__ import annotations

import json
import os

import cv2

from cloth_pipeline.paths import CONDITION_DIR, DATASET_DIR, METADATA_PATH, ensure_sketch_stage_dirs
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


def _pattern_name_from_meta(meta: dict) -> str | None:
    return meta.get("pattern_name") or meta.get("texture_pattern")


def run_from_metadata(*, output_dir: str | None = None) -> None:
    ensure_sketch_stage_dirs()
    condition_dir = output_dir or CONDITION_DIR
    os.makedirs(condition_dir, exist_ok=True)

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
        out_path = os.path.join(condition_dir, f"conditioning_{frame_str}.png")

        if os.path.exists(out_path):
            print(f"  [{frame_str}] Skipping (already exists)")
            continue

        raw_text = meta.get("text", "")
        keyword  = meta.get("keyword") or "fabric pattern"
        material_label = _material_label_from_meta(meta)
        obj_name = "Wool Scarf" if "wool" in raw_text.lower() else "Cloth"

        mask_rel = meta.get("mask_image")
        alpha_mask_path = (
            os.path.join(DATASET_DIR, mask_rel) if mask_rel else None
        )

        try:
            co = meta.get("cam_origin")
            ct = meta.get("cam_target")
            fov = float(meta.get("fov_deg", 40.0))
            sketch = generate_sketch(
                render_path,
                obj_name,
                material_label,
                keyword,
                alpha_mask_path=alpha_mask_path,
                pattern_name=_pattern_name_from_meta(meta),
                cam_origin=co if isinstance(co, (list, tuple)) and len(co) >= 3 else None,
                cam_target=ct if isinstance(ct, (list, tuple)) and len(ct) >= 3 else None,
                fov_deg=fov,
            )
            cv2.imwrite(out_path, sketch)
            print(f"  [{frame_str}] ✓  → {out_path}")
        except Exception as e:
            print(f"  [{frame_str}] [ERROR] {e}")
