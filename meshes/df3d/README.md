# meshes/df3d/

DeepFashion3D V2 garment scans live here. The dataset itself is **not** committed — it's ~7 GB, academic-research-license, redistribution forbidden.

## How to populate this folder

1. Request access at https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D and download `filtered_registered_mesh.rar` from the Google Drive link they email you.
2. Unrar with the password they provide:
   ```bash
   unrar x filtered_registered_mesh.rar
   ```
3. Symlink the extracted folder *inside* this directory (do not replace this directory itself — that would unstage the README):
   ```bash
   ln -s /absolute/path/to/extracted/filtered_registered_mesh meshes/df3d/all
   ```

After that, Stage 1 will find DF3D meshes via the recursive glob (`meshes/df3d/**/*.obj`).

## Layout once populated

```
meshes/df3d/
├── README.md          ← committed, this file
└── all/               ← gitignored symlink to your extraction
    ├── 1-1/model_cleaned.obj + .mtl + tex.png
    ├── 1-2/...
    └── ... (~590 garments)
```

## Categories (from `garment_type_list.txt` shipped with DF3D V2)

~590 garments across 9 categories: long_sleeve_upper, short_sleeve_upper, no_sleeve_upper, long_sleeve_dress, short_sleeve_dress, no_sleeve_dress, long_pants, short_pants, dress.
