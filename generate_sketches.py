"""
generate_sketches.py
--------------------
Stage 2: Mitsuba renders → aligned sketch conditioning images.

The CV pipeline is split under ``cloth_pipeline.sketch`` (edges, segmentation,
shadows, features, drawing, batch runner). Run from project root:

  python generate_sketches.py

See module docstrings there for HED/SAM optional models and env vars.
"""

from cloth_pipeline.sketch import run_from_metadata

if __name__ == "__main__":
    run_from_metadata()
