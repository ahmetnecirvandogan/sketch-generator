#!/bin/bash
#
# Smoke run: 50 DF3D meshes × 2 lightings = 100 samples, 2 epochs.
# Use this first to confirm the pipeline works end-to-end.
#
# Submit:   sbatch scripts/first_training_smoke.sbatch
# Watch:    squeue -u $USER
# Logs:     logs/sgen-smoke-<jobid>.out
#
#SBATCH --job-name=sgen-smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sgen-smoke-%j.out
#SBATCH --error=logs/sgen-smoke-%j.err

set -euo pipefail

mkdir -p logs
cd "$HOME/sketch-generator"

# If your env activation lives elsewhere (e.g. module load + venv), put it here.
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

bash scripts/first_training_run.sh
