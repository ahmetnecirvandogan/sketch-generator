# Sketch Generator

Three-stage pipeline for synthetic cloth dataset generation: **Blender** generates physically simulated draped cloth meshes, **Mitsuba 3** assembles each sample into a scene (mesh + randomised material + lighting + camera) and renders it, then a **computer-vision sketch pipeline** turns those renders into aligned conditioning images for ControlNet training.

## Pipelines


### Pipeline 1 — Training Data Generation

Runs once, offline.

```
┌──────────────────────────────────────────────────────────┐
│  Stage 0 — Mesh Generation (Blender, headless)           │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
                    output_meshes/*.obj
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — PBR Rendering (Mitsuba 3)                     │
│  scene = mesh + material + lighting + camera             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
       all outputs below come from one render of the scene
                             │
                             ▼
   render.png  albedo.png  roughness.png  normal.png  mask.png
   texture.png  depth.npy  normals.npy
   prompt.txt  metadata.json
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Qwen2-VL-7B  (FROZEN, GPU, DF3D samples only)           │
│  reads render.png, rewrites metadata.json["text"] +      │
│  prompt.txt with a per-sample visual caption             │
│  (color, pattern, fabric, lighting, view).               │
│  Manual + procedural samples are passed through.         │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
   prompt.txt (enriched on DF3D)  metadata.json["text"] (same)
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Sketch Extraction                             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
                        sketch.png

   All written into  dataset/<bucket>/<mesh>/<material>_<pattern>/view_<n>/sample_<NNNN>/
                          ↑ bucket = manual / df3d / procedural (auto-derived from mesh source)
```


**Legend**
- `sketch.png` — network input
- `prompt.txt` — text conditioning input
- `albedo.png` + `roughness.png` — Variant A training targets
- `render.png` — Variant B training target
- `metadata.json`'s `lighting_sh` field — 9 SH coefficients projected from the raw `lights[]` array. Variant A third training target (alongside albedo and roughness). Captures both ambient and directional lighting components.
- `normal.png`, `mask.png`, `texture.png`, `depth.npy`, `normals.npy`, `metadata.json` — auxiliary

### Pipeline 2 — Training

```
   VARIANT A — Separated PBR maps          VARIANT B — Combined render
   (primary)                               (ablation)

          sketch.png + prompt.txt                 sketch.png + prompt.txt
                     │                                       │
                     ▼                                       ▼
   ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
   │       Neural Network             │    │       Neural Network             │
   │                                  │    │                                  │
   │ prompt ──►┌──────────┐           │    │ prompt ──►┌──────────┐           │
   │           │  CLIP    │           │    │           │  CLIP    │           │
   │           │ (FROZEN) │           │    │           │ (FROZEN) │           │
   │           └────┬─────┘           │    │           └────┬─────┘           │
   │                │ (B, 512)        │    │                │ (B, 512)        │
   │                ▼                 │    │                ▼                 │
   │ sketch ──►┌──────────┐           │    │ sketch ──►┌──────────┐           │
   │           │  U-Net   │           │    │           │  U-Net   │           │
   │           │(TRAINABLE)           │    │           │(TRAINABLE)           │
   │           └─┬───┬──┬─┘           │    │           └─────┬────┘           │
   └─────────────┼───┼──┼─────────────┘    └─────────────────┼────────────────┘
                 ▼   ▼  ▼                                     ▼
              albedo roughness lighting                     render
                            (9 floats)

   Loss:  L_albedo + λ₁·L_roughness        Loss:  L_render  (MSE)
                   + λ₂·L_lighting  (MSE)
```

Variants share dataset, loader, and architecture. They differ only in output head and loss target.

Lighting prediction in Variant A captures Dr. Montazeri's point that sketch highlight/shadow marks encode illumination as well as material. Variant B handles this implicitly via its end-to-end render target.

**Not yet implemented.**

### Pipeline 3 — Inference

End-user pipeline.

```
   VARIANT 1 — Gemini + Trellis        VARIANT 2 — Fully in-house
   (external)

    user sketch + marks + prompt        user sketch + marks + prompt
                │                                    │
        ┌───────┴───────┐                   ┌────────┴────────┐
        ▼               ▼                   ▼                 ▼
                ┌─────────────────┐                    ┌─────────────────┐
   ┌──────────┐ │ Trained PBR     │  ┌────────────┐    │ Trained PBR     │
   │  Gemini  │ │   Model         │  │  In-house  │    │   Model         │
   │   API    │ │ ┌────────┐      │  │  sketch →  │    │ ┌────────┐      │
   │ (FROZEN) │ │ │ CLIP   │◄─prompt│  │    mesh    │    │ │ CLIP   │◄─prompt
   └────┬─────┘ │ │(FROZEN)│      │  │  (FUTURE)  │    │ │(FROZEN)│      │
        │       │ └───┬────┘      │  └─────┬──────┘    │ └───┬────┘      │
        │       │     │ (B,512)   │        │           │     │ (B,512)   │
        │       │ ┌───┴─────┐     │        │           │ ┌───┴─────┐     │
        │       │ │  U-Net  │◄──sketch     │           │ │  U-Net  │◄──sketch
        │       │ │(trained │     │        │           │ │(trained │     │
        │       │ │ weights)│     │        │           │ │ weights)│     │
        │       │ └─┬───┬─┬─┘     │        │           │ └─┬───┬─┬─┘     │
        │       └───┼───┼─┼───────┘        │           └───┼───┼─┼───────┘
        ▼           ▼   ▼ ▼                ▼               ▼   ▼ ▼
    2D image    albedo+roughness          mesh        albedo+roughness
                       +lighting           │                   +lighting
        │               │                  │                       │
        ▼               │                  └─────────┬─────────────┘
   ┌──────────┐         │                            ▼
   │ Trellis  │         │                   ┌────────────────┐
   └────┬─────┘         │                   │   Compositor   │
        │               │                   └────────┬───────┘
        ▼               │                            │
      mesh              │                            ▼
        │               │                          Scene
        └───────┬───────┘
                ▼
        ┌────────────────┐
        │   Compositor   │
        └────────┬───────┘
                 │
                 ▼
               Scene

Notes:
- CLIP is the **same frozen weights** as during training (Pipeline 2). The user's
  free-form prompt goes through the same encoder the model learned to read.
- Qwen does NOT run at inference — it was a training-data preprocessing step.
  Once the model is trained, only CLIP + U-Net are needed.
```

**Comparison**
- **Variant 1**: external dependency (Gemini, Trellis).
- **Variant 2**: fully in-house. Requires sketch → mesh model. 

Lighting is predicted as 9 SH coefficients. At render time, the user can use the predicted lighting (reproducing the sketch's intended scene) or override with custom lighting (relighting the textured mesh freely). This preserves artist control over relighting while honoring the illumination cues drawn into the sketch.

**Not yet implemented. Variant choice pending supervisor input.**

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 0 – Mesh Generation (Blender, headless)                   │
│  mesh_generator.py                                               │
│  Drops randomised cloth planes onto collision meshes with         │
│  physics simulation → meshes/procedural/*.obj                    │
├──────────────────────────────────────────────────────────────────┤
│  Stage 1 – PBR Rendering (Mitsuba 3)                             │
│  generate_dataset.py                                             │
│  Builds a scene per sample (mesh + random material + lighting    │
│  + camera) and renders it, writing every AOV in one pass          │
│  → dataset/<bucket>/<mesh>/<material>/view_0/sample_N/          │
├──────────────────────────────────────────────────────────────────┤
│  Stage 2 – Sketch Extraction                                     │
│  generate_sketches.py                                            │
│  Processes renders into line-art conditioning images              │
│  → dataset/<bucket>/<mesh>/<material>/view_0/sample_N/sketch.png │
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

Place at least one **`.obj`** cloth mesh in `meshes/manual/` before running the pipeline.

### Optional — DeepFashion3D V2 dataset (`meshes/df3d/`)

DF3D V2 is a large academic dataset (~3.6 GB, ~590 garment scans) under a research-only license that forbids redistribution. It is therefore **never committed** to this repository. To use it locally, request access at https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D, extract `filtered_registered_mesh.rar`, and symlink the extracted folder into `meshes/df3d/`:

```bash
ln -s /absolute/path/to/your/extracted/filtered_registered_mesh meshes/df3d
```

Stage 1 will pick up DF3D garments automatically (the renderer scans `meshes/df3d/*/*.obj` recursively). The symlink is gitignored.

### Optional (better edges / segmentation)

- **HED:** `deploy.prototxt` and `hed_pretrained_bsds.caffemodel` in the project root, or set `HED_MODEL_DIR` to the folder that contains them. Without these, the pipeline uses bilateral Canny edges.
- **Segment Anything:** `pip install git+https://github.com/facebookresearch/segment-anything.git` plus **PyTorch**, and the ViT-H checkpoint (default path: `sam_vit_h_4b8939.pth` in the project root, or set `SAM_CHECKPOINT`). Without SAM, alpha/threshold segmentation is used when possible.

Optional `handwriting.ttf` in the project root improves label rendering; otherwise system fonts are used (see `cloth_pipeline/sketch/constants.py`).

## Project Layout

| Path | Role |
|------|------|
| `meshes/manual/` | Hand-sourced `.obj` collision/base meshes — original 6 + TurboSquid additions (committed) |
| `meshes/df3d/` | DeepFashion3D V2 garments — symlink to local extracted dataset, never committed |
| `meshes/procedural/` | Stage 0's draped cloth output — regeneratable, gitignored |
| `scripts/mesh_generator.py` | Headless Blender script for physics-based cloth generation |
| `scripts/run_pipeline.sh` | One-command orchestrator that runs all three stages |
| `scripts/check_env.sh` | Sanity-check Python/Blender/Mitsuba/torch/mesh buckets |
| `scripts/smoke_test.sh` | End-to-end smoke test on 3+3+3 meshes (one per bucket) |
| `cloth_pipeline/` | Library code (dataset render loop, sketch pipeline) |
| `scripts/generate_dataset.py` | Stage 1 entry point (Mitsuba rendering) |
| `scripts/generate_sketches.py` | Stage 2 entry point (sketch extraction) |
| `scripts/generate_augmented_renders.py` | Disconnected — affine augmentation reference (not in pipeline) |
| `dataset/manual/` | Renders + sketches + maps for manual-bucket meshes (gitignored except README) |
| `dataset/df3d/` | Renders + sketches + maps for df3d-bucket meshes (gitignored except README) |
| `dataset/procedural/` | Renders + sketches + maps for procedural-bucket meshes (gitignored except README) |
| `dataset/metadata.jsonl` | One JSON object per frame (lighting, material, text prompt, paths, …); gitignored |

## Usage

### Quick Start — Run Everything

From the **repository root**, run the full pipeline with a single command:

```bash
./scripts/run_pipeline.sh
```

This will sequentially execute all three stages:
1. Generate 10 new draped cloth meshes via Blender
2. Render all meshes (new + base) with Mitsuba
3. Extract sketch conditioning images

### Stage 0 — Mesh Generation (Blender)

Generate physically simulated draped cloth meshes:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b -P scripts/mesh_generator.py -- --variations 10
```

**CLI flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--variations` | `5` | Number of unique cloth meshes to generate |
| `--subdivisions` | `40` | Mesh detail level (higher = more vertices) |
| `--target_frame` | `100` | Physics simulation length (frames) |
| `--input_dir` | `meshes/manual` | Folder containing collision base meshes |
| `--output_dir` | `meshes/procedural` | Folder to save generated draped `.obj` files |

**Randomisation per mesh:**
- **Base mesh**: Randomly selected from `meshes/manual/`
- **Cloth shape**: Square, Rectangle, or Scarf (with optional U-bend for worn scarf look)
- **Fabric physics**: Thin Scarf, Silk, Cotton, or Denim presets (mass, stiffness, bending)
- **Drop angle**: Random rotation and tilt before simulation
- **Friction**: Randomised collision friction

### Stage 1 — PBR Rendering (Mitsuba 3)

```bash
python scripts/generate_dataset.py
```

Scans **all three** mesh buckets — `meshes/procedural/`, `meshes/df3d/`, `meshes/manual/` — for `.obj` files. Procedural (Stage 0 output) is rendered **first** so new generations can be checked immediately, then DF3D, then manual.

Pass `--exclude-manual` to skip the `meshes/manual/` bucket — useful once procedural + DF3D coverage is sufficient and you want to drop the original hand-sourced meshes from the dataset.

Default: 3 materials × 2 lightings = **6 renders per mesh**.

```bash
python scripts/generate_dataset.py --materials-per-mesh 5 --lightings-per-material 3
```

Existing frames are skipped when outputs and metadata already exist (checkpointing).

**Outputs per frame:**
- Beauty render with light-gray background
- Depth + normals as `.npy`
- Binary cloth mask
- PBR ground truth maps (albedo, normal, roughness)

### Stage 2 — Sketch Extraction

```bash
python scripts/generate_sketches.py
```

Requires `dataset/metadata.jsonl` and the render paths it references (normally after Stage 1). Existing sketches are skipped.

### Sketch Options

Texture/pattern strokes are disabled by default to avoid grid artifacts in conditioning sketches.

- Enable with full variable name:
  - `USE_TEXTURE_STROKES=true python scripts/generate_sketches.py`
- Or use the short alias:
  - `UTS=1 python scripts/generate_sketches.py`

## Training Data Format

`dataset/metadata.jsonl` is a HuggingFace-compatible JSONL file. Each line contains:
- `text`: Natural language prompt describing the material and pattern (e.g., *"Cloth Scarf, leather material, a photorealistic 3D render of a leather cloth with houndstooth pattern"*)
- `file_name`: Path to the beauty render
- `conditioning_image`: Path to the sketch
- Full material properties, lighting config, and camera parameters

This format is directly loadable by `datasets.load_dataset()` for ControlNet/Stable Diffusion fine-tuning.

## Mitsuba Note

The renderer uses `scalar_rgb`. If `pip install mitsuba` fails on your platform, follow the [Mitsuba 3 documentation](https://mitsuba.readthedocs.io/) for build or variant requirements.
