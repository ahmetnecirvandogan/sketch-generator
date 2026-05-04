#!/bin/bash
#SBATCH --job-name=qwen-caption
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/qwen_caption_%j.log

# Ensure log directory exists
mkdir -p logs

# Load environment (assumes setup from docs/valar.md)
cd $WORK/sketch-generator
conda activate sketchgen

# Ensure VLM dependencies are present for the first run
pip install -q qwen-vl-utils accelerate

echo "Starting Qwen2-VL-7B-Instruct captioning job..."
python pbr_model/postprocess_df3d_captions.py

echo "Job finished at $(date)"
