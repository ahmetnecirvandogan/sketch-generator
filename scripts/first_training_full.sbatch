#!/bin/bash
#
# Full run: 1,212 DF3D meshes × 2 lightings = 2,424 samples, 10 epochs.
# Only submit AFTER smoke run has succeeded.
#
# Submit:   sbatch scripts/first_training_full.sbatch
# Watch:    squeue -u $USER ; tail -f logs/sgen-full-<jobid>.out
# Logs:     logs/sgen-full-<jobid>.out
#
#SBATCH --job-name=sgen-full
#SBATCH --partition=avg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sgen-full-%j.out
#SBATCH --error=logs/sgen-full-%j.err

set -euo pipefail

mkdir -p logs
cd "$HOME/sketch-generator"

# Activate sketchgen conda env (torch + mitsuba + transformers + PIL).
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate sketchgen

bash scripts/first_training_run.sh --full
