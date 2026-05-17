#!/bin/bash
# scripts/test_pr47.sh
# Smoke-test PR #47 (epoch-based training loop) on VALAR.
#
# Prerequisites:
#   1. feat/training-loop branch is checked out:
#        git fetch origin && git checkout feat/training-loop
#   2. conda env activated:
#        conda activate sketchgen
#   3. dataset/ is populated (Stage 1 + Stage 2 run already)
#   4. GPU available on the node (run inside an sbatch / srun allocation)
#
# Usage from repo root:
#   bash scripts/test_pr47.sh
#
# Exit code 0 = pass. Non-zero = fail.

set -e
cd "$(dirname "$0")/.."

CHECKPOINT_DIR="checkpoints/pr47-test"
LOG_FILE="/tmp/pr47-test-$$.log"

# -------- Preflight --------
[ -f scripts/train_pbr_model.py ] \
    || { echo "FAIL: scripts/train_pbr_model.py missing — did you checkout feat/training-loop?"; exit 1; }

[ -f dataset/metadata.jsonl ] \
    || { echo "FAIL: dataset/metadata.jsonl missing — run Stage 1 + Stage 2 first"; exit 1; }

N_SAMPLES=$(wc -l < dataset/metadata.jsonl | tr -d ' ')
echo "[preflight] dataset has $N_SAMPLES samples"

if [ "$N_SAMPLES" -lt 4 ]; then
    echo "FAIL: dataset has fewer than 4 samples; can't test batch_size=4"
    exit 1
fi

# Clean any prior test artifacts so the checkpoint check is meaningful
rm -rf "$CHECKPOINT_DIR"

# -------- Run --------
echo "[run] 2 epochs, Variant B, batch_size=4, base_channels=16"
echo

python scripts/train_pbr_model.py \
    --variant b \
    --epochs 2 \
    --batch-size 4 \
    --base-channels 16 \
    --save-every 1 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$LOG_FILE"

# -------- Validate --------
echo
echo "=== Validation ==="

FAILED=0

# Device detection
if grep -q 'Device: cuda' "$LOG_FILE"; then
    echo "[device] cuda detected ✓"
else
    echo "[device] cuda NOT detected ✗ — check nvidia-smi + torch CUDA install"
    FAILED=1
fi

# Loss decreased
FIRST_LOSS=$(grep -oE 'Loss: [0-9.]+' "$LOG_FILE" | head -1 | awk '{print $2}')
LAST_LOSS=$(grep -oE 'Loss: [0-9.]+' "$LOG_FILE" | tail -1 | awk '{print $2}')

if [ -z "$FIRST_LOSS" ] || [ -z "$LAST_LOSS" ]; then
    echo "[loss] could not parse loss values from log ✗"
    FAILED=1
else
    echo "[loss] first=$FIRST_LOSS last=$LAST_LOSS"
    if python3 -c "import sys; sys.exit(0 if float('$LAST_LOSS') < float('$FIRST_LOSS') else 1)"; then
        echo "[loss] decreased ✓"
    else
        echo "[loss] did NOT decrease ✗"
        FAILED=1
    fi
fi

# NaN check
if grep -qi 'nan' "$LOG_FILE"; then
    echo "[nan] found NaN in log ✗"
    FAILED=1
else
    echo "[nan] no NaN ✓"
fi

# Checkpoint files
for ck in latest_variant_b.pt epoch_1_variant_b.pt epoch_2_variant_b.pt; do
    if [ -f "$CHECKPOINT_DIR/$ck" ]; then
        echo "[checkpoint] $ck ✓"
    else
        echo "[checkpoint] $ck ✗ MISSING"
        FAILED=1
    fi
done

# Checkpoint loads
if [ -f "$CHECKPOINT_DIR/latest_variant_b.pt" ]; then
    python3 -c "
import torch
ck = torch.load('$CHECKPOINT_DIR/latest_variant_b.pt', weights_only=False)
required = {'epoch', 'model_state_dict', 'optimizer_state_dict', 'variant', 'base_channels', 'avg_loss'}
missing = required - set(ck.keys())
assert not missing, f'checkpoint missing keys: {missing}'
print(f'[checkpoint] loadable | epoch={ck[\"epoch\"]} | avg_loss={ck[\"avg_loss\"]:.4f} | variant={ck[\"variant\"]} ✓')
" || FAILED=1
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo "=== PR #47 smoke test: PASS ==="
    echo
    echo "Next step: comment on PR #47 with these results, then merge."
    exit 0
else
    echo "=== PR #47 smoke test: FAIL ==="
    echo "Log saved at: $LOG_FILE"
    exit 1
fi
