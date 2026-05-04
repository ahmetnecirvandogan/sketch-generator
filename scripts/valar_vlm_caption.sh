#!/bin/bash
#SBATCH --job-name=qwen-caption
#SBATCH --time=12:00:00
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/qwen_caption_%j.log

# Ensure log directory exists
mkdir -p logs

# 1. Use the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

# 2. Correct way to activate Conda in a SLURM script
# We find where conda is installed and source its profile
CONDA_PATH=$(which conda)
CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the environment (created in docs/valar.md)
conda activate sketchgen || { echo "Error: Conda environment 'sketchgen' not found. Create it with 'conda create -n sketchgen python=3.10'"; exit 1; }

# 3. Ensure VLM dependencies are present
echo "Checking dependencies..."
pip install -q qwen-vl-utils accelerate

# 4. Run the post-processing script
echo "Starting Qwen2-VL-7B-Instruct captioning job..."
python pbr_model/postprocess_df3d_captions.py

echo "Job finished at $(date)"
