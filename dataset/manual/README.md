# dataset/manual/

Renders + sketches + maps for samples whose source mesh came from `meshes/manual/` (hand-sourced .obj files: original 6 scarves + TurboSquid additions).

Layout per sample: `<mesh-stem>/<material>_<pattern>/view_<n>/sample_<NNNN>/`. See top-level `dataset/README.md` for the per-sample file list.

Contents (everything except this README) are gitignored — regeneratable by running `python generate_dataset.py` (Stage 1) followed by `python generate_sketches.py` (Stage 2).
