#!/bin/bash
# scripts/smoke_test.sh
# End-to-end smoke test on a 5+5+5 mesh subset (one per bucket).
#
# Expected outputs:
#   15 sketches (one per mesh, in dataset/<bucket>/<mesh>/<mat>_<pat>/view_0/sample_NNNN/sketch.png)
#   45 PBR maps (albedo + roughness + normal × 15 samples)
#   plus per-sample render.png, mask.png, texture.png, depth.npy, normals.npy, prompt.txt, metadata.json
#   plus dataset/metadata.jsonl with 15 lines + lighting_sh field after preprocessing
#
# Lighting is NOT a map — it's 9 SH coefficients stored inside metadata.json
# (written by pbr_model/preprocess_lighting_sh.py from PR #25).

set -e
cd "$(dirname "$0")/.."

N_PER_BUCKET=5

echo "=== Smoke test: ${N_PER_BUCKET} manual + ${N_PER_BUCKET} df3d + ${N_PER_BUCKET} procedural ==="
echo

# Stage 0: generate N procedural draped meshes (Blender physics).
# Skip if meshes/procedural already has ≥ N .obj files.
N_PROC=$(ls meshes/procedural/*.obj 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_PROC" -lt "$N_PER_BUCKET" ]; then
    echo "--- Stage 0: generating $N_PER_BUCKET procedural meshes ---"
    if [ -n "${BLENDER:-}" ]; then
        BLENDER_BIN="$BLENDER"
    elif command -v blender &>/dev/null; then
        BLENDER_BIN="blender"
    elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
        BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
    else
        echo "[ERROR] Blender not found — set BLENDER or add to PATH"; exit 1
    fi
    "$BLENDER_BIN" -b -P scripts/mesh_generator.py -- --variations $N_PER_BUCKET --target_frame 60 --subdivisions 30
else
    echo "--- Stage 0: skipping (meshes/procedural already has $N_PROC .obj files) ---"
fi
echo

# Stage 1: render N from each bucket × 1 material × 1 lighting = 3N samples total.
TOTAL=$((N_PER_BUCKET * 3))
echo "--- Stage 1: rendering (${N_PER_BUCKET} manual + ${N_PER_BUCKET} df3d + ${N_PER_BUCKET} procedural) × 1 mat × 1 light = $TOTAL samples ---"
python3 scripts/generate_dataset.py \
    --max-per-bucket $N_PER_BUCKET \
    --materials-per-mesh 1 \
    --lightings-per-material 1
echo

# Stage 2: extract sketches from the renders.
# Uses the default sketch algorithm (no pattern strokes) — matches what
# Neçirvan ships. To preview pattern strokes, run manually with
# USE_TEXTURE_STROKES=true.
echo "--- Stage 2: extracting sketches ---"
python3 scripts/generate_sketches.py
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
echo "  sketches:   $N_SKETCHES (expected $TOTAL)"
echo "  albedo:     $N_ALBEDO    (expected $TOTAL)"
echo "  roughness:  $N_ROUGH    (expected $TOTAL)"
echo "  normal:     $N_NORMAL    (expected $TOTAL)"
echo "  total maps: $((N_ALBEDO + N_ROUGH + N_NORMAL)) (expected $((TOTAL * 3)))"
echo
echo "=== Per-bucket distribution ==="
for bucket in manual df3d procedural; do
    n=$(find dataset/$bucket -name 'sketch.png' 2>/dev/null | wc -l | tr -d ' ')
    echo "  $bucket: $n sketches"
done
echo
echo "Done."
