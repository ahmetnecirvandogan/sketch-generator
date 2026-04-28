"""Project-root paths used by Stage 1 (dataset) and Stage 2 (sketches)."""

import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_PKG_DIR)

MESHES_DIR = os.path.join(BASE_DIR, "cloth_meshes")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RENDERS_DIR = os.path.join(DATASET_DIR, "renders")
DEPTH_DIR = os.path.join(DATASET_DIR, "depth")
NORMALS_DIR = os.path.join(DATASET_DIR, "normals")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
# Procedural albedo / pattern maps (Mitsuba base_color). Folder name is historical.
ALBEDO_MAPS_DIR = os.path.join(DATASET_DIR, "textures")
TEXTURES_DIR = ALBEDO_MAPS_DIR  # deprecated alias — same path as ALBEDO_MAPS_DIR
CONDITION_DIR = os.path.join(DATASET_DIR, "conditioning")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
FRONT_PREVIEW_DIR = os.path.join(DATASET_DIR, "front_previews")

# PBR ground-truth maps, pixel-aligned with renders/sketches.
# Albedo + normal come from the Mitsuba AOV pass; roughness is a post-hoc
# uniform fill from material_props (single BSDF per garment for now).
PBR_DIR = os.path.join(DATASET_DIR, "pbr")
PBR_ALBEDO_DIR = os.path.join(PBR_DIR, "albedo")
PBR_ROUGHNESS_DIR = os.path.join(PBR_DIR, "roughness")
PBR_NORMAL_DIR = os.path.join(PBR_DIR, "normal")


def ensure_dataset_stage_dirs() -> None:
    for d in (
        RENDERS_DIR, DEPTH_DIR, NORMALS_DIR, MASKS_DIR, MESHES_DIR, ALBEDO_MAPS_DIR,
        PBR_ALBEDO_DIR, PBR_ROUGHNESS_DIR, PBR_NORMAL_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def ensure_front_preview_dir() -> None:
    os.makedirs(FRONT_PREVIEW_DIR, exist_ok=True)


def ensure_sketch_stage_dirs() -> None:
    os.makedirs(CONDITION_DIR, exist_ok=True)
