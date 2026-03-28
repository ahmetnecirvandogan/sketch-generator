"""
generate_dataset.py
-------------------
Stage 1 of the Neural-Contours-Cloth ControlNet dataset pipeline.

Implementation lives in ``cloth_pipeline.dataset`` (paths, procedural textures,
Mitsuba render loop). Run this file from the project root as before:

  python generate_dataset.py

Outputs land under ``dataset/``; Stage 2 is ``generate_sketches.py``.
"""

from cloth_pipeline.dataset import run_generation

if __name__ == "__main__":
    run_generation()
