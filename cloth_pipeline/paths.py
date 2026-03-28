"""Project-root paths used by Stage 1 (dataset) and Stage 2 (sketches)."""

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_PKG_DIR)

MESHES_DIR = os.path.join(BASE_DIR, "cloth_meshes")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RENDERS_DIR = os.path.join(DATASET_DIR, "renders")
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
NORMALS_DIR = os.path.join(DATASET_DIR, "normals")
TEXTURES_DIR = os.path.join(DATASET_DIR, "textures")
CONDITION_DIR = os.path.join(DATASET_DIR, "conditioning")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")


def ensure_dataset_stage_dirs() -> None:
    for d in (RENDERS_DIR, DEPTH_DIR, NORMALS_DIR, MESHES_DIR, TEXTURES_DIR):
        os.makedirs(d, exist_ok=True)


def ensure_sketch_stage_dirs() -> None:
    os.makedirs(CONDITION_DIR, exist_ok=True)
