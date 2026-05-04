"""Project-root paths used by Stage 1 (dataset) and Stage 2 (sketches)."""

import os
import re

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_PKG_DIR)

# Pre-#27 single-bucket layout (kept commented for reference)
# MESHES_DIR = os.path.join(BASE_DIR, "cloth_meshes")

# New three-bucket layout (issue #27)
MANUAL_MESHES_DIR = os.path.join(BASE_DIR, "meshes", "manual")
DF3D_MESHES_DIR = os.path.join(BASE_DIR, "meshes", "df3d")
PROCEDURAL_MESHES_DIR = os.path.join(BASE_DIR, "meshes", "procedural")

# Backward-compat alias: MESHES_DIR = manual bucket (replaces old cloth_meshes/).
MESHES_DIR = MANUAL_MESHES_DIR

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")
FRONT_PREVIEW_DIR = os.path.join(DATASET_DIR, "front_previews")

# Training-ready outputs (sketch + PBR maps), grouped per mesh / view.
# Stage 1 writes albedo/normal/roughness here; Stage 2 writes sketch here.
# Layout: dataset/mesh_<sanitized_stem>/view_<idx>/{sketch,albedo,normal,roughness}.png



def sanitize_mesh_name(stem: str) -> str:
    """Lowercase + collapse non-alphanumeric runs to underscores; strip edges.
    Generic enough to use for any path component (mesh, material, pattern)."""
    s = stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def sample_dir_components(
    mesh_path: str,
    material_type: str,
    pattern_name: str,
    view_idx: int,
    frame_id: int,
) -> dict:
    """Named pieces of the per-sample output directory; mirror these into metadata."""
    mesh_stem = os.path.splitext(os.path.basename(mesh_path))[0]
    return {
        "mesh_dir_name": f"mesh_{sanitize_mesh_name(mesh_stem)}",
        "material_pattern_dir_name": (
            f"{sanitize_mesh_name(material_type)}_{sanitize_mesh_name(pattern_name)}"
        ),
        "view_idx": int(view_idx),
        "sample_dir_name": f"sample_{int(frame_id):04d}",
    }


def output_sample_dir(
    mesh_path: str,
    material_type: str,
    pattern_name: str,
    view_idx: int,
    frame_id: int,
) -> str:
    """Absolute path to the per-(mesh, material+pattern, view, frame) directory."""
    parts = sample_dir_components(mesh_path, material_type, pattern_name, view_idx, frame_id)
    return os.path.join(
        DATASET_DIR,
        parts["mesh_dir_name"],
        parts["material_pattern_dir_name"],
        f"view_{parts['view_idx']}",
        parts["sample_dir_name"],
    )


def ensure_dataset_stage_dirs() -> None:
    """Create flat intermediate dirs + the dataset/ root (per-mesh subdirs are
    created on demand at write time)."""
    for d in (MANUAL_MESHES_DIR, DF3D_MESHES_DIR, PROCEDURAL_MESHES_DIR, DATASET_DIR):
        os.makedirs(d, exist_ok=True)


def ensure_front_preview_dir() -> None:
    os.makedirs(FRONT_PREVIEW_DIR, exist_ok=True)
