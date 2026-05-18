#!/usr/bin/env bash
# End-to-end first training run on VALAR.
#
# Usage:
#   bash scripts/first_training_run.sh           # smoke (50 meshes × 2 lightings)
#   bash scripts/first_training_run.sh --full    # full DF3D (1212 × 2)
#
# Smoke mode produces 100 samples, runs 2 epochs. Takes ~30-45 min.
# Full mode produces ~2,424 samples, runs 10 epochs. Takes ~4-6 hours.
#
# Halts on first error. Prerequisites:
#   - integration/valar-training-ready branch checked out
#   - meshes/df3d/all/ populated (scp from local)
#   - python env with torch + mitsuba + transformers + PIL installed
#   - CUDA GPU strongly recommended

set -euo pipefail

# ────────────────────────────────────────────────────────────────────────────
# mode
# ────────────────────────────────────────────────────────────────────────────
MODE="smoke"
if [[ "${1:-}" == "--full" ]]; then
    MODE="full"
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    head -16 "$0" | tail -14
    exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
    STAGE1_FLAGS=(--max-per-bucket 50 --materials-per-mesh 1 --lightings-per-material 2)
    EPOCHS=2
    BATCH=4
    BASE_CH=16
    CHK_DIR=checkpoints/first-smoke
else
    STAGE1_FLAGS=(--materials-per-mesh 1 --lightings-per-material 2)
    EPOCHS=10
    BATCH=8
    BASE_CH=32
    CHK_DIR=checkpoints/first-real
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  sketch-generator first training run — MODE=$MODE"
echo "═══════════════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────────────
# 0. preflight
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[0/6] preflight checks..."
if [[ ! -d meshes/df3d/all ]]; then
    echo "FATAL: meshes/df3d/all/ missing — scp the DF3D dataset first." >&2
    exit 2
fi
python3 - <<'PY'
import sys
try:
    import torch, mitsuba, transformers
    from PIL import Image  # noqa
except Exception as e:
    sys.exit(f"FATAL: required package missing: {e}")
print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"  mitsuba={mitsuba.__version__}")
print(f"  transformers={transformers.__version__}")
if not torch.cuda.is_available():
    print("  WARNING: no CUDA — training will be very slow")
PY
N_GARMENTS=$(ls meshes/df3d/all | wc -l | tr -d ' ')
echo "  DF3D garments found: $N_GARMENTS"
if [[ "$N_GARMENTS" -lt 100 ]]; then
    echo "  WARNING: very few garments. expected ~1212."
fi

# ────────────────────────────────────────────────────────────────────────────
# 1. validate meshes (issue #42 sub-task 1.2)
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[1/6] validating DF3D meshes..."
python3 scripts/validate_df3d_meshes.py
if [[ -s meshes/df3d/known_bad.txt ]]; then
    echo "  bad meshes flagged (meshes/df3d/known_bad.txt):"
    sed 's/^/    /' meshes/df3d/known_bad.txt
    echo "  NOTE: Stage 1 currently has no --exclude-list flag (#44). Bad meshes"
    echo "        may crash the run; consider moving them out of meshes/df3d/all/"
fi

# ────────────────────────────────────────────────────────────────────────────
# 2. Stage 1 — render dataset (writes per-sample PNGs + metadata.json + metadata.jsonl)
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[2/6] Stage 1: rendering dataset (DF3D only)..."
echo "  flags: ${STAGE1_FLAGS[*]}"
python3 scripts/generate_dataset.py --exclude-manual "${STAGE1_FLAGS[@]}"

# ────────────────────────────────────────────────────────────────────────────
# 3. preprocess_lighting_sh (populates lighting_sh field for Variant A)
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[3/6] preprocessing lighting_sh for Variant A..."
python3 pbr_model/preprocess_lighting_sh.py

# ────────────────────────────────────────────────────────────────────────────
# 4. Stage 2 — sketches
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[4/6] Stage 2: extracting sketches..."
python3 scripts/generate_sketches.py

# ────────────────────────────────────────────────────────────────────────────
# 5. dataset sanity check
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[5/6] dataset sanity check..."
if [[ ! -s dataset/metadata.jsonl ]]; then
    echo "FATAL: dataset/metadata.jsonl is empty after Stage 1." >&2
    exit 3
fi
N_SAMPLES=$(wc -l < dataset/metadata.jsonl | tr -d ' ')
echo "  $N_SAMPLES samples in metadata.jsonl"
python3 -m pbr_model.dataset --variant a --split all --batch-size "$BATCH"

# ────────────────────────────────────────────────────────────────────────────
# 6. train — Variant A (sketch+prompt → albedo + roughness + lighting_sh)
# ────────────────────────────────────────────────────────────────────────────
echo
echo "[6/6] training Variant A..."
mkdir -p "$CHK_DIR"
python3 scripts/train_pbr_model.py \
    --variant a \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --base-channels "$BASE_CH" \
    --save-every 1 \
    --checkpoint-dir "$CHK_DIR" \
    2>&1 | tee "$CHK_DIR/train.log"

# ────────────────────────────────────────────────────────────────────────────
# done
# ────────────────────────────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════"
echo "  DONE — MODE=$MODE"
echo "  samples: $N_SAMPLES"
echo "  checkpoints: $CHK_DIR/"
echo "  log: $CHK_DIR/train.log"
echo "═══════════════════════════════════════════════════════════════"
ls -la "$CHK_DIR"
