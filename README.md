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
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1 — PBR Rendering (Mitsuba 3, per sample)                     │
│  scene = mesh + material/texture + lighting + camera                 │
│  DF3D: bundled <id>_tex.png used as albedo (no procedural pattern)   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
                    per sample, all in one folder:
       render.png  albedo.png  roughness.png  normal.png  mask.png
       texture.png  depth.npy  normals.npy  prompt.txt  metadata.json
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
       (DF3D samples only)               (all samples)
                  │                             │
                  ▼                             ▼
   ┌──────────────────────────────┐   ┌─────────────────────────────────┐
   │  Qwen2-VL-7B  (FROZEN, GPU)  │   │  Stage 2 — Sketch Extraction    │
   │                              │   │  (Neçirvan's CV pipeline)       │
   │  input:  render.png          │   │                                 │
   │  output: rich visual caption │   │  input:   render.png + masks +  │
   │          (color, pattern,    │   │           normals + albedo      │
   │           fabric, lighting,  │   │  output:  sketch.png            │
   │           view)              │   │                                 │
   │  rewrites in place:          │   │                                 │
   │    prompt.txt                │   │                                 │
   │    metadata.json["text"]     │   │                                 │
   └──────────────────────────────┘   └─────────────────────────────────┘
                  │                                   │
                  --------------------------------------                     
                                    |
                                    ▼
                                sketch.png 
         

   Both branches run on the same per-sample folder. Qwen and Stage 2 are
   independent — neither depends on the other's output.

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
   VARIANT A — Separated PBR maps (primary)         VARIANT B — Combined render (ablation)

   sketch.png       prompt.txt                      sketch.png       prompt.txt
   (B,3,512,512)    list[str] len=B                 (B,3,512,512)    list[str] len=B
        │                 │                                │                 │
        │                 ▼                                │                 ▼
        │          ┌─────────────┐                         │          ┌─────────────┐
        │          │    CLIP     │                         │          │    CLIP     │
        │          │  (FROZEN)   │                         │          │  (FROZEN)   │
        │          └──────┬──────┘                         │          └──────┬──────┘
        │                 │                                │                 │
        │            text feat (B, 512)                    │            text feat (B, 512)
        │                 │                                │                 │
        ▼                 ▼                                ▼                 ▼
   ┌──────────────────────────────────┐              ┌──────────────────────────────────┐
   │             U-Net                │              │             U-Net                │
   │  takes sketch (image input) +    │              │  takes sketch (image input) +    │
   │  text feat (channel-wise bias    │              │  text feat (channel-wise bias    │
   │  at bottleneck)  →  predicts maps│              │  at bottleneck)  →  predicts img │
   │           (TRAINABLE)            │              │           (TRAINABLE)            │
   └─────┬─────────┬─────────┬────────┘              └─────────────────┬────────────────┘
         │         │         │                                         │
         ▼         ▼         ▼                                         ▼
       albedo   roughness  lighting_sh                              render
     (B,3,H,W) (B,1,H,W)  (B, 9 floats)                          (B,3,H,W)

   Loss:  L_albedo + λ₁·L_roughness + λ₂·L_lighting     Loss:  L_render  (MSE)
                            (all MSE)

   Note: the sketch is an INPUT, never derived from prompt. CLIP only processes
   the prompt. Both flow into the U-Net at different points (sketch as the image
   input, text features added as channel-wise bias at the U-Net bottleneck).
```

Variants share dataset, loader, and architecture. They differ only in output head and loss target.

Lighting prediction in Variant A captures Dr. Montazeri's point that sketch highlight/shadow marks encode illumination as well as material. Variant B handles this implicitly via its end-to-end render target.

**Not yet implemented.**

### Pipeline 3 — Inference

End-user pipeline. Two variants — same Trained PBR Model in both, only the mesh source differs. The user's sketch is an INPUT to both the mesh-source path AND the U-Net inside the PBR Model. CLIP only processes the user's prompt; sketch is never derived from prompt.

#### Variant 1 — Gemini + Trellis (external)

```
                user sketch.png         user prompt
                     │  │                    │
                ┌────┘  └─────┐              │
                │             │              │
                ▼             │              │
            ┌──────────┐      │              │
            │  Gemini  │      │              │
            │ (FROZEN) │      │              │
            └────┬─────┘      │              │
                 ▼            │              │
             2D image         │              │
                 │            │              │
                 ▼            │              │
            ┌──────────┐      │              │
            │ Trellis  │      │              │
            │ (FROZEN) │      │              │
            └────┬─────┘      │              │
                 │            │              ▼
                 ▼            │       ┌─────────────┐
                mesh          │       │    CLIP     │
                 │            │       │  (FROZEN)   │
                 │            │       └──────┬──────┘
                 │            │              │ text feat (B, 512)
                 │            ▼              ▼
                 │       ┌──────────────────────────────┐
                 │       │           U-Net              │
                 │       │  takes sketch + text feat →  │
                 │       │  predicts albedo, roughness, │
                 │       │  lighting_sh                 │
                 │       │      (trained weights;       │
                 │       │       not updated at         │
                 │       │       inference time)        │
                 │       └─────┬─────────┬───────┬──────┘
                 │             ▼         ▼       ▼
                 │          albedo   roughness  lighting_sh
                 │             │         │       │
                 └───────────┐ │         │       │
                             ▼ ▼         ▼       ▼
                          ┌──────────────────────────┐
                          │       Compositor         │
                          │  combines mesh + maps    │
                          │  + (predicted or user)   │
                          │  lighting → final scene  │
                          └────────────┬─────────────┘
                                       ▼
                                     Scene
```

#### Variant 2 — Fully in-house

```
                user sketch.png         user prompt
                     │  │                    │
                ┌────┘  └─────┐              │
                │             │              │
                ▼             │              │
         ┌────────────┐       │              │
         │  In-house  │       │              │
         │  sketch →  │       │              │
         │   mesh     │       │              │
         │  (FUTURE,  │       │              │
         │  TRAINABLE)│       │              │
         └─────┬──────┘       │              │
               │              │              ▼
               ▼              │       ┌─────────────┐
              mesh            │       │    CLIP     │
               │              │       │  (FROZEN)   │
               │              │       └──────┬──────┘
               │              │              │ text feat (B, 512)
               │              ▼              ▼
               │       ┌──────────────────────────────┐
               │       │           U-Net              │
               │       │  takes sketch + text feat →  │
               │       │  predicts albedo, roughness, │
               │       │  lighting_sh                 │
               │       │      (trained weights;       │
               │       │       not updated at         │
               │       │       inference time)        │
               │       └─────┬─────────┬───────┬──────┘
               │             ▼         ▼       ▼
               │          albedo   roughness  lighting_sh
               │             │         │       │
               └───────────┐ │         │       │
                           ▼ ▼         ▼       ▼
                        ┌──────────────────────────┐
                        │       Compositor         │
                        │  combines mesh + maps    │
                        │  + (predicted or user)   │
                        │  lighting → final scene  │
                        └────────────┬─────────────┘
                                     ▼
                                   Scene
```

**Notes that apply to both variants:**
- The **sketch is an input** drawn by the user; it is *never* derived from the prompt. It feeds two paths in parallel: the mesh-source pipeline (Gemini→Trellis or in-house sketch→mesh), AND the U-Net inside the Trained PBR Model.
- **CLIP only processes the prompt.** It produces a 512-dim text feature that gets injected at the U-Net's bottleneck. Same frozen CLIP weights as during training.
- **Qwen does NOT run at inference.** It was a training-data preprocessing step (Pipeline 1). Once the model is trained, only CLIP + U-Net are needed.

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
