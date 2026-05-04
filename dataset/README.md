# dataset/

Stage 1 + Stage 2 outputs: per-sample renders, PBR maps, sketches, and metadata. **Contents are not committed** — they're regeneratable from the meshes in `meshes/`.

## Layout

```
dataset/
├── README.md                              ← committed, this file
├── manual/                                ← committed folder; outputs from meshes/manual/ go here
│   └── mesh_<stem>/<mat>_<pat>/view_<n>/sample_<NNNN>/
├── df3d/                                  ← committed folder; outputs from meshes/df3d/ go here
│   └── mesh_<stem>/<mat>_<pat>/view_<n>/sample_<NNNN>/
├── procedural/                            ← committed folder; outputs from meshes/procedural/ go here
│   └── mesh_<stem>/<mat>_<pat>/view_<n>/sample_<NNNN>/
└── metadata.jsonl                         ← gitignored, one JSON object per sample (HF-compatible)
```

The bucket dimension (`manual` / `df3d` / `procedural`) is inserted automatically by `cloth_pipeline.paths.bucket_for_mesh_path()` based on which mesh source the input came from.

## Per-sample contents

```
sample_<NNNN>/
├── render.png        ← beauty render (Variant B target)
├── albedo.png        ← Variant A target #1
├── roughness.png     ← Variant A target #2
├── normal.png        ← auxiliary surface normals
├── mask.png          ← cloth silhouette
├── texture.png       ← procedural pattern that was applied (or DF3D bundled texture)
├── sketch.png        ← Stage 2 output, network input
├── depth.npy         ← G-buffer (raw float depth)
├── normals.npy       ← G-buffer (raw float normals)
├── prompt.txt        ← text conditioning input
└── metadata.json     ← per-sample JSON (also appears as one line in dataset/metadata.jsonl)
```

After running `pbr_model/preprocess_lighting_sh.py`, each `metadata.json` also gains a `lighting_sh` field (9 floats — Variant A's third target).
