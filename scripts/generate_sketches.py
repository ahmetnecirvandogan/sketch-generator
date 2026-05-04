"""
generate_sketches.py
--------------------
Stage 2: Mitsuba renders → aligned sketch images written to
``dataset/<mesh>/view_<idx>/sketch.png`` next to the PBR maps from Stage 1.

The CV pipeline is split under ``cloth_pipeline.sketch`` (edges, segmentation,
shadows, features, drawing, batch runner). Run from project root:

  python generate_sketches.py

See module docstrings there for HED/SAM optional models and env vars.
"""

import os
import sys

# scripts/ → repo root on sys.path so `cloth_pipeline` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloth_pipeline.sketch import run_from_metadata

if __name__ == "__main__":
    run_from_metadata()
