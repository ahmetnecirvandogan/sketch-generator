#!/bin/bash
#SBATCH --job-name=render-stage1
#SBATCH --time=02:00:00
#SBATCH --partition=ai
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/render_%j.log

# Ensure log directory exists
mkdir -p logs

# Use the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# Activate Conda
CONDA_PATH=$(which conda)
CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate sketchgen

echo "Starting Stage 1 Rendering..."
# Run the renderer with your test parameters
python scripts/generate_dataset.py --max-per-bucket 1 --exclude-manual

echo "Rendering finished at $(date)"
