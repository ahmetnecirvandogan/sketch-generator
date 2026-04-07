"""
generate_sketches.py
--------------------
Stage 2: Mitsuba renders → aligned sketch conditioning images.

The CV pipeline is split under ``cloth_pipeline.sketch`` (edges, segmentation,
shadows, features, drawing, batch runner). Run from project root:

  python generate_sketches.py

See module docstrings there for HED/SAM optional models and env vars.
"""

import argparse

from cloth_pipeline.sketch import run_from_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conditioning sketches from renders.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for conditioning images (default: dataset/conditioning).",
    )
    args = parser.parse_args()
    run_from_metadata(output_dir=args.output_dir)
