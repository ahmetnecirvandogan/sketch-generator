#!/bin/bash
# scripts/smoke_test.sh
# End-to-end smoke test on a 3+3+3 mesh subset (one per bucket).
#
# Expected outputs:
#   9 sketches (one per mesh, in dataset/<mesh>/<mat>_<pat>/view_0/sample_0000/sketch.png)
#   27 PBR maps (albedo + roughness + normal × 9 samples)
#   plus per-sample render.png, mask.png, texture.png, depth.npy, normals.npy, prompt.txt, metadata.json
#   plus dataset/metadata.jsonl with 9 lines + lighting_sh field after preprocessing
#
# Lighting is NOT a map — it's 9 SH coefficients stored inside metadata.json
# (written by pbr_model/preprocess_lighting_sh.py from PR #25).

set -e
cd "$(dirname "$0")/.."

echo "=== Smoke test: 3 manual + 3 df3d + 3 procedural ==="
echo

# Stage 0: generate 3 procedural draped meshes (Blender physics).
# Skip if meshes/procedural already has ≥ 3 .obj files.
N_PROC=$(ls meshes/procedural/*.obj 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_PROC" -lt 3 ]; then
    echo "--- Stage 0: generating 3 procedural meshes ---"
    if [ -n "${BLENDER:-}" ]; then
        BLENDER_BIN="$BLENDER"
    elif command -v blender &>/dev/null; then
        BLENDER_BIN="blender"
    elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
        BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
    else
        echo "[ERROR] Blender not found — set BLENDER or add to PATH"; exit 1
    fi
    "$BLENDER_BIN" -b -P mesh_generator.py -- --variations 3 --target_frame 60 --subdivisions 30
else
    echo "--- Stage 0: skipping (meshes/procedural already has $N_PROC .obj files) ---"
fi
echo

# Stage 1: render 3 from each bucket × 1 material × 1 lighting = 9 samples total.
echo "--- Stage 1: rendering (3 manual + 3 df3d + 3 procedural) × 1 mat × 1 light ---"
python3 generate_dataset.py \
    --max-per-bucket 3 \
    --materials-per-mesh 1 \
    --lightings-per-material 1
echo

# Stage 2: extract sketches from the 9 renders.
echo "--- Stage 2: extracting sketches ---"
python3 generate_sketches.py
echo

# Optional preprocessing: project lighting to SH coefficients (9 floats per sample).
echo "--- Preprocessing: SH lighting projection ---"
python3 pbr_model/preprocess_lighting_sh.py 2>&1 || echo "  (skipped — pbr_model/preprocess_lighting_sh.py not on this branch yet, lands with PR #25)"
echo

# Tally outputs.
echo "=== Output tally ==="
N_SKETCHES=$(find dataset -name 'sketch.png' 2>/dev/null | wc -l | tr -d ' ')
N_ALBEDO=$(find dataset -name 'albedo.png' 2>/dev/null | wc -l | tr -d ' ')
N_ROUGH=$(find dataset -name 'roughness.png' 2>/dev/null | wc -l | tr -d ' ')
N_NORMAL=$(find dataset -name 'normal.png' 2>/dev/null | wc -l | tr -d ' ')
echo "  sketches:   $N_SKETCHES (expected 9)"
echo "  albedo:     $N_ALBEDO    (expected 9)"
echo "  roughness:  $N_ROUGH    (expected 9)"
echo "  normal:     $N_NORMAL    (expected 9)"
echo "  total maps: $((N_ALBEDO + N_ROUGH + N_NORMAL)) (expected 27)"
echo
echo "Done."
