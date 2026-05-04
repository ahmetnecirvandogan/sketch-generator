# dataset/procedural/

Renders + sketches + maps for samples whose source mesh came from `meshes/procedural/` (Stage 0's draped cloth output).

Layout per sample: `<mesh-stem>/<material>_<pattern>/view_<n>/sample_<NNNN>/`. See top-level `dataset/README.md` for the per-sample file list.

Contents (everything except this README) are gitignored — regeneratable by running Stage 0 (`mesh_generator.py`) followed by Stage 1 (`generate_dataset.py`) and Stage 2 (`generate_sketches.py`).
