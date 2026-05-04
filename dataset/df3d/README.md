# dataset/df3d/

Renders + sketches + maps for samples whose source mesh came from `meshes/df3d/` (DeepFashion3D V2 garment scans).

Layout per sample: `<mesh-stem>/<material>_<pattern>/view_<n>/sample_<NNNN>/`. See top-level `dataset/README.md` for the per-sample file list.

Contents (everything except this README) are gitignored — regeneratable by running `python generate_dataset.py` (Stage 1) followed by `python generate_sketches.py` (Stage 2). Requires `meshes/df3d/` to be populated (see `meshes/df3d/README.md`).
