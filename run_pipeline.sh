#!/bin/bash
# Neural Contours Pipeline: Full orchestration script
# 1. Generates synthetically draped cloth meshes
# 2. Renders PBR maps (albedo, normals, depth) for all meshes
# 3. Extracts sketches (contours)

set -e # Exit immediately if a command exits with a non-zero status.

echo "================================================="
echo "   STAGE 0: Mesh Generation (Blender)"
echo "================================================="
# Generate 10 variations by default. Change this number as needed.
/Applications/Blender.app/Contents/MacOS/Blender -b -P mesh_generator.py -- --variations 10 --subdivisions 40
echo "Done generating meshes."

echo ""
echo "================================================="
echo "   STAGE 1: Dataset Rendering (Mitsuba)"
echo "================================================="
# Renders all .obj files found in cloth_meshes/ and output_meshes/
python generate_dataset.py
echo "Done rendering datasets."

echo ""
echo "================================================="
echo "   STAGE 2: Sketch Extraction"
echo "================================================="
# Processes the Mitsuba outputs into line art
python generate_sketches.py
echo "Done extracting sketches."

echo ""
echo "Pipeline completed successfully! Your data is ready in the 'dataset' directory."
