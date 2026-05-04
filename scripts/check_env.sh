#!/bin/bash
# scripts/check_env.sh
# Sanity-check the local / cluster environment before running the pipeline.
# Exit 0 = ready. Non-zero = something missing.

set -e
cd "$(dirname "$0")/.."

ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
fail() { echo "  ✗ $*"; exit 1; }

echo "=== Python ==="
python3 -c "import sys; print(f'  python {sys.version.split()[0]}')"
python3 -c "import sys; assert sys.version_info >= (3, 9), 'need Python 3.9+'" || fail "Python 3.9+ required"
ok "Python OK"

echo
echo "=== Required packages ==="
for pkg in numpy cv2 PIL mitsuba; do
    python3 -c "import $pkg" 2>/dev/null && ok "$pkg importable" || fail "$pkg not importable (pip install)"
done
python3 -c "import torch" 2>/dev/null && ok "torch importable (training-side)" || warn "torch not importable (only needed for #19/#20/#21 training scaffolds)"

echo
echo "=== Blender (Stage 0) ==="
if [ -n "${BLENDER:-}" ] && [ -x "$BLENDER" ]; then
    ok "BLENDER env var → $BLENDER"
elif command -v blender &>/dev/null; then
    ok "blender on PATH → $(command -v blender)"
elif [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    ok "Blender at macOS default location"
else
    warn "Blender not found — Stage 0 will fail. Set BLENDER env var or add to PATH."
fi

echo
echo "=== Mesh buckets (issue #27 layout) ==="
[ -d meshes/manual ] && ok "meshes/manual/ exists ($(ls meshes/manual/*.obj 2>/dev/null | wc -l | tr -d ' ') .obj files)" || fail "meshes/manual/ missing"
if [ -L meshes/df3d ]; then
    ok "meshes/df3d → symlink ($(find -L meshes/df3d -maxdepth 2 -name '*.obj' 2>/dev/null | wc -l | tr -d ' ') .obj files reachable)"
elif [ -d meshes/df3d ]; then
    warn "meshes/df3d/ is a real folder, not a symlink — fine but unusual ($(find meshes/df3d -name '*.obj' 2>/dev/null | wc -l | tr -d ' ') .obj files)"
else
    warn "meshes/df3d/ missing — Stage 1 won't see DF3D meshes (run: ln -s /path/to/DATASET-DF3D/filtered_registered_mesh meshes/df3d)"
fi
[ -d meshes/procedural ] && ok "meshes/procedural/ exists ($(ls meshes/procedural/*.obj 2>/dev/null | wc -l | tr -d ' ') .obj files; created by Stage 0)" || warn "meshes/procedural/ missing — Stage 0 will create it on first run"

echo
echo "=== Optional perception models (sketch quality) ==="
[ -f deploy.prototxt ] && [ -f hed_pretrained_bsds.caffemodel ] && ok "HED model files present" || warn "HED model files missing — Stage 2 falls back to Canny edges"
[ -f sam_vit_h_4b8939.pth ] && ok "SAM checkpoint present" || warn "SAM checkpoint missing — Stage 2 falls back to alpha/threshold segmentation"

echo
echo "=== All critical checks passed ==="
