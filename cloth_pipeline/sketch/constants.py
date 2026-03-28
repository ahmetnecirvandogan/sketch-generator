"""Sketch colour, optional model paths, and handwriting font lookup."""

import os

from PIL import ImageFont

from cloth_pipeline.paths import BASE_DIR

SKETCH_RGB = (34, 139, 69)
SKETCH_BGR = (69, 139, 34)

HED_MODEL_DIR = os.environ.get("HED_MODEL_DIR", BASE_DIR)
HED_PROTO = os.path.join(HED_MODEL_DIR, "deploy.prototxt")
HED_WEIGHTS = os.path.join(HED_MODEL_DIR, "hed_pretrained_bsds.caffemodel")

SAM_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT", os.path.join(BASE_DIR, "sam_vit_h_4b8939.pth")
)
SAM_MODEL_TYPE = "vit_h"

_FONT_CANDIDATES = [
    os.path.join(BASE_DIR, "handwriting.ttf"),
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
]


def resolve_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()
