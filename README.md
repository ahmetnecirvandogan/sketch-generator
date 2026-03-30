# Sketch generator

Two-stage pipeline for synthetic cloth data: **Mitsuba 3** renders meshes with random materials and lighting, then a **computer-vision sketch pipeline** turns those renders into aligned conditioning images (edges, segmentation, hatching, labels).

## Requirements

- **Python 3.9+**
- **pip packages:** `numpy`, `opencv-python`, `Pillow`, `mitsuba`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy opencv-python pillow mitsuba
```

Place at least one **`.obj`** cloth mesh in `cloth_meshes/` before running Stage 1.

### Optional (better edges / segmentation)

- **HED:** `deploy.prototxt` and `hed_pretrained_bsds.caffemodel` in the project root, or set `HED_MODEL_DIR` to the folder that contains them. Without these, the pipeline uses bilateral Canny edges.
- **Segment Anything:** `pip install git+https://github.com/facebookresearch/segment-anything.git` plus **PyTorch**, and the ViT-H checkpoint (default path: `sam_vit_h_4b8939.pth` in the project root, or set `SAM_CHECKPOINT`). Without SAM, alpha/threshold segmentation is used when possible.

Optional `handwriting.ttf` in the project root improves label rendering; otherwise system fonts are used (see `cloth_pipeline/sketch/constants.py`).

## Project layout

| Path | Role |
|------|------|
| `cloth_meshes/` | Input `.obj` meshes (you provide) |
| `cloth_pipeline/` | Library code (dataset render loop, sketch pipeline) |
| `dataset/renders/` | Beauty PNGs composited on a light-gray background |
| `dataset/depth/`, `dataset/normals/` | Per-frame `.npy` buffers |
| `dataset/masks/` | Per-frame cloth alpha masks (`mask_XXXX.png`) for sketch segmentation |
| `dataset/textures/` | Procedural texture PNGs |
| `dataset/metadata.jsonl` | One JSON object per frame (lighting, material, paths, …) |
| `dataset/conditioning/` | Output sketch conditioning images (Stage 2) |

## Usage

Run from the **repository root**.

**Stage 1 — synthetic renders**

```bash
python generate_dataset.py
```

Default sample count is set in `cloth_pipeline/dataset/render_loop.py` (`run_generation`). Existing frames are skipped when outputs and metadata already exist (checkpointing).

Stage 1 writes:
- render image with visible light-gray background (easy to inspect; no checkerboard transparency preview)
- depth + normals as `.npy`
- binary cloth mask in `dataset/masks/` used by Stage 2 segmentation

**Stage 2 — sketches from metadata**

```bash
python generate_sketches.py
```

Requires `dataset/metadata.jsonl` and the render paths it references (normally after Stage 1). Existing `conditioning_*.png` files are skipped.

### Sketch options

Texture/pattern strokes are disabled by default to avoid grid artifacts in conditioning sketches.

- Enable with full variable name:
  - `USE_TEXTURE_STROKES=true python generate_sketches.py`
- Or use the short alias:
  - `UTS=1 python generate_sketches.py`

## Mitsuba note

The renderer uses `scalar_rgb`. If `pip install mitsuba` fails on your platform, follow the [Mitsuba 3 documentation](https://mitsuba.readthedocs.io/) for build or variant requirements.