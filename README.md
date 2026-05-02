# Sketch Generator

Three-stage pipeline for synthetic cloth dataset generation: **Blender** generates physically simulated draped cloth meshes, **Mitsuba 3** renders them with randomised materials and lighting, then a **computer-vision sketch pipeline** turns those renders into aligned conditioning images for ControlNet training.

## Pipelines

Forward-looking design document. Pipeline 1 (data generation) is built today; Pipelines 2 (training) and 3 (inference) are not yet implemented.

### Pipeline 1 — Training Data Generation

Runs once, offline. Owner: Neçirvan.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 0 — Mesh Generation (Blender, headless)                           │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
                              output_meshes/*.obj
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — PBR Rendering (Mitsuba 3)                                     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
   render.png   albedo.png   roughness.png   normal.png   mask.png
   texture.png  depth.npy    normals.npy     prompt.txt   metadata.json
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Sketch Extraction                                             │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
                                 sketch.png

   All written into  dataset/<mesh>/<material>_<pattern>/view_<n>/sample_<NNNN>/
```

**Legend**
- `sketch.png` — network input
- `prompt.txt` — text conditioning input
- `albedo.png` + `roughness.png` — Variant A training targets
- `render.png` — Variant B training target
- `normal.png`, `mask.png`, `texture.png`, `depth.npy`, `normals.npy`, `metadata.json` — auxiliary

### Pipeline 2 — Training

Runs once per ablation variant. Owner: Acar. Required ablation per Dr. Montazeri.

```
   VARIANT A — Separated PBR maps (primary)                   VARIANT B — Combined render (ablation)

         sketch.png + prompt.txt                                  sketch.png + prompt.txt
                  │                                                          │
                  ▼                                                          ▼
   ┌─────────────────────────────────────┐                    ┌─────────────────────────────────────┐
   │           Neural Network            │                    │           Neural Network            │
   └─────────┬─────────────────┬─────────┘                    └────────────────┬────────────────────┘
             │                 │                                               │
             ▼                 ▼                                               ▼
          albedo            roughness                                       render

   Loss:  L_albedo + λ · L_roughness  (MSE)                   Loss:  L_render  (MSE)
```

Variants share dataset, loader, and architecture. They differ only in output head and loss target. Compared on a held-out test set via PSNR/SSIM.

**Not yet implemented.**

### Pipeline 3 — Inference

End-user pipeline. Variant choice TBD with Prof. Sezgin.

```
   VARIANT 1 — Gemini + Trellis (external)                    VARIANT 2 — Fully in-house

       user sketch + marks + text prompt                        user sketch + marks + text prompt
                       │                                                        │
              ┌────────┴────────┐                                       ┌───────┴────────┐
              ▼                 ▼                                       ▼                ▼
       ┌────────────┐    ┌─────────────────┐                    ┌────────────┐   ┌─────────────────┐
       │   Gemini   │    │   Trained PBR   │                    │  In-house  │   │   Trained PBR   │
       │    API     │    │      Model      │                    │  sketch →  │   │      Model      │
       └─────┬──────┘    └────────┬────────┘                    │    mesh    │   └────────┬────────┘
             │                    │                             │  (FUTURE)  │            │
             ▼                    ▼                             └─────┬──────┘            ▼
          2D image         albedo + roughness                         │            albedo + roughness
             │                    │                                   ▼                   │
             ▼                    │                                 mesh                  │
       ┌────────────┐             │                                   │                   │
       │  Trellis   │             │                                   └─────────┬─────────┘
       └─────┬──────┘             │                                             ▼
             │                    │                                    ┌────────────────┐
             ▼                    │                                    │   Compositor   │
           mesh                   │                                    └────────┬───────┘
             │                    │                                             │
             └─────────┬──────────┘                                             ▼
                       ▼                                               textured 3D mesh
              ┌────────────────┐
              │   Compositor   │
              └────────┬───────┘
                       │
                       ▼
               textured 3D mesh
```

**Comparison**
- **Variant 1**: external dependency (Gemini, Trellis). Demo-ready now. Narrower research contribution.
- **Variant 2**: fully in-house. Requires sketch → mesh model. Broader research contribution.

**Not yet implemented. Variant choice pending supervisor input.**

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 0 – Mesh Generation (Blender, headless)                   │
│  mesh_generator.py                                               │
│  Drops randomised cloth planes onto collision meshes with         │
│  physics simulation → output_meshes/*.obj                        │
├──────────────────────────────────────────────────────────────────┤
│  Stage 1 – PBR Rendering (Mitsuba 3)                             │
│  generate_dataset.py                                             │
│  Loads meshes from output_meshes/ + cloth_meshes/, applies        │
│  random materials/lighting, renders beauty + depth + normals      │
│  → dataset/<mesh>/<material>/view_0/sample_N/                    │
├──────────────────────────────────────────────────────────────────┤
│  Stage 2 – Sketch Extraction                                     │
│  generate_sketches.py                                            │
│  Processes renders into line-art conditioning images              │
│  → dataset/<mesh>/<material>/view_0/sample_N/sketch.png          │
└──────────────────────────────────────────────────────────────────┘
```

## Requirements

- **Python 3.9+**
- **Blender 3.4+** installed locally (macOS: `/Applications/Blender.app`)
- **pip packages:** `numpy`, `opencv-python`, `Pillow`, `mitsuba`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy opencv-python pillow mitsuba
```

Place at least one **`.obj`** cloth mesh in `cloth_meshes/` before running the pipeline.

### Optional (better edges / segmentation)

- **HED:** `deploy.prototxt` and `hed_pretrained_bsds.caffemodel` in the project root, or set `HED_MODEL_DIR` to the folder that contains them. Without these, the pipeline uses bilateral Canny edges.
- **Segment Anything:** `pip install git+https://github.com/facebookresearch/segment-anything.git` plus **PyTorch**, and the ViT-H checkpoint (default path: `sam_vit_h_4b8939.pth` in the project root, or set `SAM_CHECKPOINT`). Without SAM, alpha/threshold segmentation is used when possible.

Optional `handwriting.ttf` in the project root improves label rendering; otherwise system fonts are used (see `cloth_pipeline/sketch/constants.py`).

## Project Layout

| Path | Role |
|------|------|
| `cloth_meshes/` | Input `.obj` base/collision meshes (you provide) |
| `output_meshes/` | Synthetically generated draped cloth meshes (Stage 0 output) |
| `mesh_generator.py` | Headless Blender script for physics-based cloth generation |
| `run_pipeline.sh` | One-command orchestrator that runs all three stages |
| `cloth_pipeline/` | Library code (dataset render loop, sketch pipeline) |
| `generate_dataset.py` | Stage 1 entry point (Mitsuba rendering) |
| `generate_sketches.py` | Stage 2 entry point (sketch extraction) |
| `dataset/` | All per-sample generated data (renders, depth, normals, sketch, etc) |
| `dataset/metadata.jsonl` | One JSON object per frame (lighting, material, text prompt, paths, …) |

## Usage

### Quick Start — Run Everything

From the **repository root**, run the full pipeline with a single command:

```bash
./run_pipeline.sh
```

This will sequentially execute all three stages:
1. Generate 10 new draped cloth meshes via Blender
2. Render all meshes (new + base) with Mitsuba
3. Extract sketch conditioning images

### Stage 0 — Mesh Generation (Blender)

Generate physically simulated draped cloth meshes:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b -P mesh_generator.py -- --variations 10
```

**CLI flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--variations` | `5` | Number of unique cloth meshes to generate |
| `--subdivisions` | `40` | Mesh detail level (higher = more vertices) |
| `--target_frame` | `100` | Physics simulation length (frames) |
| `--input_dir` | `cloth_meshes` | Folder containing collision base meshes |
| `--output_dir` | `output_meshes` | Folder to save generated `.obj` files |

**Randomisation per mesh:**
- **Base mesh**: Randomly selected from `cloth_meshes/`
- **Cloth shape**: Square, Rectangle, or Scarf (with optional U-bend for worn scarf look)
- **Fabric physics**: Thin Scarf, Silk, Cotton, or Denim presets (mass, stiffness, bending)
- **Drop angle**: Random rotation and tilt before simulation
- **Friction**: Randomised collision friction

### Stage 1 — PBR Rendering (Mitsuba 3)

```bash
python generate_dataset.py
```

Scans **both** `output_meshes/` and `cloth_meshes/` for `.obj` files. New generated meshes are rendered **first** so you can quickly validate them.

Default: 3 materials × 2 lightings = **6 renders per mesh**.

```bash
python generate_dataset.py --materials-per-mesh 5 --lightings-per-material 3
```

Existing frames are skipped when outputs and metadata already exist (checkpointing).

**Outputs per frame:**
- Beauty render with light-gray background
- Depth + normals as `.npy`
- Binary cloth mask
- PBR ground truth maps (albedo, normal, roughness)

### Stage 2 — Sketch Extraction

```bash
python generate_sketches.py
```

Requires `dataset/metadata.jsonl` and the render paths it references (normally after Stage 1). Existing sketches are skipped.

### Sketch Options

Texture/pattern strokes are disabled by default to avoid grid artifacts in conditioning sketches.

- Enable with full variable name:
  - `USE_TEXTURE_STROKES=true python generate_sketches.py`
- Or use the short alias:
  - `UTS=1 python generate_sketches.py`

## Training Data Format

`dataset/metadata.jsonl` is a HuggingFace-compatible JSONL file. Each line contains:
- `text`: Natural language prompt describing the material and pattern (e.g., *"Cloth Scarf, leather material, a photorealistic 3D render of a leather cloth with houndstooth pattern"*)
- `file_name`: Path to the beauty render
- `conditioning_image`: Path to the sketch
- Full material properties, lighting config, and camera parameters

This format is directly loadable by `datasets.load_dataset()` for ControlNet/Stable Diffusion fine-tuning.

## Mitsuba Note

The renderer uses `scalar_rgb`. If `pip install mitsuba` fails on your platform, follow the [Mitsuba 3 documentation](https://mitsuba.readthedocs.io/) for build or variant requirements.