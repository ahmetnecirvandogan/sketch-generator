"""
generate_dataset.py
-------------------
Stage 1 of the Neural-Contours-Cloth ControlNet dataset pipeline.

Implementation lives in ``cloth_pipeline.dataset`` (paths, procedural albedo maps,
Mitsuba render loop). Run this file from the project root as before:

  python generate_dataset.py
  python generate_dataset.py --front-previews   # one front view per mesh → dataset/front_previews/
  python generate_dataset.py --front-previews --front-preview-only STEM   # one mesh only

Outputs land under ``dataset/``; Stage 2 is ``generate_sketches.py``.
"""

import argparse

from cloth_pipeline.dataset import run_front_mesh_previews, run_generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Mitsuba cloth dataset renders.")
    parser.add_argument(
        "--front-previews",
        action="store_true",
        help="Render one fixed-front preview PNG per OBJ in cloth_meshes/ (dataset/front_previews/).",
    )
    parser.add_argument(
        "--front-preview-only",
        metavar="STEM",
        default=None,
        help="With --front-previews: only render cloth_meshes/STEM.obj (no .obj suffix).",
    )
    parser.add_argument(
        "--materials-per-mesh",
        type=int,
        default=3,
        metavar="N",
        help="Number of (material, pattern) combinations per mesh (default: 3).",
    )
    parser.add_argument(
        "--lightings-per-material",
        type=int,
        default=2,
        metavar="N",
        help="Number of lighting variations per (mesh, material, pattern) (default: 2).",
    )
    args = parser.parse_args()
    if args.front_previews:
        run_front_mesh_previews(only_stem=args.front_preview_only)
    else:
        run_generation(
            materials_per_mesh=args.materials_per_mesh,
            lightings_per_material=args.lightings_per_material,
        )
