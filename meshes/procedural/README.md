# meshes/procedural/

Stage 0's draped cloth output lives here. **Not committed** — these meshes are regeneratable by running `mesh_generator.py`.

## How to populate this folder

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b -P mesh_generator.py -- --variations N
```

(Or use the wrapper at `scripts/run_pipeline.sh`.) Each variation drops a randomised cloth plane onto a random base mesh from `meshes/manual/`, runs physics, and exports the draped result here as `draped_<base-stem>_<hash>.obj`.

## Layout once populated

```
meshes/procedural/
├── README.md                                ← committed, this file
├── draped_turbosquid_jean_0818e4.obj        ← gitignored, Stage 0 output
├── draped_gabardine_a17b9c.obj
└── ...
```

Stage 1 picks these up via `meshes/procedural/*.obj` glob.

## Why "procedural"?

These meshes are produced by physics simulation, not hand-authored — i.e., procedurally generated. The cloth shape comes from a procedurally-randomised plane + physics drape, and (per issue #15 task 1) the collider it drapes onto can also be procedurally synthesised from primitives + noise.
