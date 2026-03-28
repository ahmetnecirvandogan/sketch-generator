"""Batch over metadata.jsonl → conditioning images."""

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
