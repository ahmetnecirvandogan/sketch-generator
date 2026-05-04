# VALAR setup + first training run

This is the cheatsheet for running the sketch-generator pipeline on the VALAR cluster. Issue #22 (env setup) + #23 (first training run).

> The repo + scripts are already designed to be cluster-safe: paths derive from the script's location, no GUI dependencies, and the smoke tests can be invoked from a SLURM job. This doc just ties everything together so you can paste the commands into a cluster shell.

---

## 1. Clone + conda env (one-time, ~5 min)

```bash
# After SSH'ing into VALAR (login node is fine for setup)
cd $WORK   # or wherever your working volume is
git clone git@github.com:ahmetnecirvandogan/sketch-generator.git
cd sketch-generator

# Create conda env
conda create -n sketchgen python=3.10 -y
conda activate sketchgen

# Install Python deps
pip install numpy opencv-python pillow mitsuba torch torchvision
pip install 'transformers>=4.45,<5'   # CLIP encoder; lazy-imported when --use-clip is set
```

**If VALAR has a `module load` system for CUDA / cuDNN**, load the matching versions for your torch build before `pip install torch`:

```bash
module load cuda/12.1 cudnn/8.9   # whatever your cluster uses
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2. Mesh data setup

The repo committed only the small `meshes/manual/` set (.obj files for the original 6 + TurboSquid 5). DF3D and procedural meshes need separate steps:

```bash
# DF3D V2 (~3.6 GB extracted) — request access from CUHK-SZ GAP LAB if not done.
# Place the extracted dir somewhere on cluster storage:
DF3D_PATH=/path/to/DATASET-DF3D/filtered_registered_mesh
ln -s $DF3D_PATH meshes/df3d/all   # gitignored

# Procedural meshes — generate via Stage 0 on a compute node (Blender required):
sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=stage0
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
module load blender/3.4    # or whatever VALAR has
cd $WORK/sketch-generator
blender -b -P scripts/mesh_generator.py -- --variations 50 --target_frame 60
SBATCH
```

If VALAR doesn't have Blender available as a module, you can:
- Skip Stage 0 entirely (DF3D's ~590 garments are enough for first training)
- Run Stage 0 locally, push the `meshes/procedural/` artifacts to a cluster-shared volume

## 3. Sanity-check (always do this first)

```bash
# Login node, before any compute job:
bash scripts/check_env.sh

# Expected output: all ✓ marks. Warnings on Blender / HED / SAM are OK if you're skipping Stage 0 or sketch quality is acceptable.
```

## 4. Generate the dataset (Stage 1 + Stage 2)

```bash
sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=stage1-2
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
cd $WORK/sketch-generator
conda activate sketchgen

# Stage 1: Mitsuba renders (~10–30 sec per sample, all CPU; bigger dataset = bigger time)
python scripts/generate_dataset.py --materials-per-mesh 3 --lightings-per-material 2

# Stage 2: sketch extraction (fast, ~1 sec per sample)
python scripts/generate_sketches.py

# Preprocess lighting → 9 SH floats per sample (Variant A target)
python pbr_model/preprocess_lighting_sh.py
SBATCH
```

Output lands in `dataset/{manual,df3d,procedural}/<mesh>/<material>_<pattern>/view_<n>/sample_<NNNN>/`.

## 5. Smoke test (validate the trained-model pipeline before kicking off real training)

```bash
# Few-step training to verify the loop runs end-to-end:
python -m pbr_model.train --variant b --batch-size 2 --steps 10 --base-channels 16

# Should see loss decrease over 10 steps + checkpoint saved to checkpoints/scaffold.pt
```

If smoke test passes, you're ready for the real run.

## 6. First real training run (issue #23)

```bash
sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=pbr-train
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1   # or whatever VALAR has
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
cd $WORK/sketch-generator
conda activate sketchgen

# Real training: Variant A, CLIP, bigger model, more steps, GPU
python -m pbr_model.train \
    --variant a \
    --batch-size 8 \
    --steps 5000 \
    --lr 1e-4 \
    --base-channels 32 \
    --use-clip \
    --device cuda \
    --checkpoint checkpoints/run01.pt
SBATCH
```

Tunables (adjust per memory / time budget):
- `--batch-size 8` — drop to 4 if OOM, push to 16 if you have headroom
- `--base-channels 32` — model size; 16 is scaffold, 32 is a good first real run, 64 needs more memory
- `--steps 5000` — first run; if loss is still dropping, do another run loading from this checkpoint
- `--lr 1e-4` — default Adam LR; adjust if loss plateaus or explodes

## 7. Common gotchas

- **`mitsuba` doesn't import on the login node** — Mitsuba may need GLIBC versions that login nodes lack. Always run from a compute node (via `srun` or `sbatch`).
- **First CLIP run downloads 250 MB to `~/.cache/huggingface/`** — make sure your home dir has space, or set `TRANSFORMERS_CACHE=$WORK/hf_cache` before invoking.
- **`torch.load` security warning with newer transformers** — if you see a CVE-2025-32434 error, downgrade to `transformers<4.50` (we use `>=4.45,<5` for compatibility).
- **Blender + headless `__pycache__`** — Blender writes its own `__pycache__` next to `mesh_generator.py`; gitignore covers it.

## 8. What to do if training diverges / fails

1. Re-run the smoke test (`python -m pbr_model.train --variant b --steps 10`). If THAT fails, something broke in the data path — re-check `dataset/metadata.jsonl` and the per-sample folders.
2. Drop `--lr` 10× and try again. Adam can blow up early if the LR is too high relative to model scale.
3. Drop `--base-channels` to 16 if you suspect the model is overshooting on small data.
4. Check `dataset/<bucket>/<mesh>/<mat>_<pat>/view_<n>/sample_<NNNN>/render.png` visually — confirm Stage 1 is producing real images, not corrupted/black outputs.
5. If Variant A's `lighting_sh` loss explodes, increase `--lambda-lighting` from 0.1 to slow it down (it's already small but the SH coefficients can have wide range).

## 9. Open issues that may bite

- Two manual meshes are `.obj.skip` on purpose — `gabardine.obj` (Mitsuba segfault) and `turbosquid_pencil_skirt.obj` (invalid normals). Don't rename them back without re-testing.
- DF3D bundled-texture passthrough is **not yet implemented** (deferred from issue #27) — Stage 1 currently generates random procedural textures over DF3D meshes instead of using the bundled `<id>_tex.png`. For the first real training run this is fine (you'll get textured cloth, just not the original photogrammetry textures). Worth flagging in the paper's methods if relevant.
