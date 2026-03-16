"""
generate_dataset.py
-------------------
Stage 1 of the Neural-Contours-Cloth ControlNet dataset pipeline.

Responsibility: Use Mitsuba 3 to render cloth meshes with randomised
lighting, materials, and geometry. Each frame produces FOUR output files:

  dataset/renders/render_XXXX.png     → beauty render (ControlNet target)
  dataset/depth/depth_XXXX.exr        → linear depth map  (Neural Contours Image Branch)
  dataset/normals/normals_XXXX.exr    → world-space normals (Neural Contours Image Branch)
  dataset/metadata.jsonl              → per-frame camera + transform data (Neural Contours Geometry Branch)

Communicates with Stage 2 (generate_sketches.py) *only* through the
/dataset folder — the two stages run in separate Python environments.
"""

import mitsuba as mi
import numpy as np
import cv2
import os
import json
import random
import glob

mi.set_variant('scalar_rgb')

# ---------------------------------------------------------------------------
# 1. PATH RESOLUTION & DIRECTORY SETUP
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MESHES_DIR   = os.path.join(BASE_DIR,    "cloth_meshes")
DATASET_DIR  = os.path.join(BASE_DIR,    "dataset")
RENDERS_DIR  = os.path.join(DATASET_DIR, "renders")
DEPTH_DIR    = os.path.join(DATASET_DIR, "depth")    # stores .npy float32 arrays
NORMALS_DIR  = os.path.join(DATASET_DIR, "normals")  # stores .npy float32 arrays

for d in (RENDERS_DIR, DEPTH_DIR, NORMALS_DIR, MESHES_DIR):
    os.makedirs(d, exist_ok=True)

# Discover all .obj files in the meshes directory
mesh_files = sorted(glob.glob(os.path.join(MESHES_DIR, "*.obj")))

print("--- PATH DIAGNOSTICS ---")
print(f"Meshes dir  : {MESHES_DIR}")
print(f"  Found     : {len(mesh_files)} .obj file(s)")
print(f"Renders dir : {RENDERS_DIR}")
print(f"Depth dir   : {DEPTH_DIR}")
print(f"Normals dir : {NORMALS_DIR}")
print("------------------------\n")

if not mesh_files:
    print(f"[ERROR] No .obj files found in {MESHES_DIR}.")
    print("Please add your cloth meshes to that folder and run again.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 2. RENDER LOOP
# ---------------------------------------------------------------------------
NUM_SAMPLES = 5   # Set to 1000 for a full training dataset
metadata_path = os.path.join(DATASET_DIR, "metadata.jsonl")

# Load existing metadata so we preserve it when skipping frames
existing_metadata = {}
if os.path.exists(metadata_path):
    with open(metadata_path) as f:
        for ln in f:
            if ln.strip():
                rec = json.loads(ln)
                existing_metadata[rec["frame"]] = rec

metadata_records = []

existing = sum(
    1 for j in range(NUM_SAMPLES)
    if os.path.exists(os.path.join(RENDERS_DIR,  f"render_{j:04d}.png"))
    and os.path.exists(os.path.join(DEPTH_DIR,   f"depth_{j:04d}.npy"))
    and os.path.exists(os.path.join(NORMALS_DIR, f"normals_{j:04d}.npy"))
)
print(f"Checkpoint: {existing}/{NUM_SAMPLES} frames already done, resuming.\n")

for i in range(NUM_SAMPLES):
    frame_str    = f"{i:04d}"
    render_path  = os.path.join(RENDERS_DIR,  f"render_{frame_str}.png")
    depth_path   = os.path.join(DEPTH_DIR,    f"depth_{frame_str}.npy")
    normals_path = os.path.join(NORMALS_DIR,  f"normals_{frame_str}.npy")

    # --- CHECKPOINT: skip frames that are fully complete (and have metadata) ---
    if (os.path.exists(render_path)
            and os.path.exists(depth_path)
            and os.path.exists(normals_path)):
        if frame_str in existing_metadata:
            metadata_records.append(existing_metadata[frame_str])
            print(f"  [{i+1:>4}/{NUM_SAMPLES}] Skipping {frame_str} (already exists)")
            continue
        # No metadata for this frame — must re-render to populate it
        print(f"  [{i+1:>4}/{NUM_SAMPLES}] Re-rendering {frame_str} (missing metadata)")

    # -----------------------------------------------------------------------
    # Randomise Geometry
    # -----------------------------------------------------------------------
    current_mesh_path = random.choice(mesh_files)
    mesh_name = os.path.basename(current_mesh_path).replace(".obj", "")

    # Calculate bounding box for this specific mesh to frame it correctly
    loaded_shape = mi.load_dict({'type': 'obj', 'filename': current_mesh_path})
    bbox    = loaded_shape.bbox()
    center  = bbox.center()
    extents = bbox.extents()
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    # DYNAMIC FRAMING: camera distance derived from perspective geometry so
    # the mesh's largest dimension fills TARGET_FILL of the image height.
    #
    #   visible_height_at_dist = 2 * dist * tan(fov / 2)
    #   we want: max_extent = TARGET_FILL * visible_height_at_dist
    #   → dist = max_extent / (2 * TARGET_FILL * tan(fov / 2))
    #
    # This is purely proportional to the mesh size — no additive constant
    # that would over-recede small meshes like uploads-files-3794072.
    fov          = 40    # fixed field of view (degrees)
    TARGET_FILL  = 0.70  # object should cover ~70 % of the image
    max_extent   = max(float(extents[0]), float(extents[1]), float(extents[2]))
    cam_dist     = max_extent / (2.0 * TARGET_FILL * np.tan(np.radians(fov / 2.0)))
    cam_origin   = [cx, cy, cz + cam_dist]
    cam_target   = [cx, cy, cz]

    # -----------------------------------------------------------------------
    # Randomise Lighting
    # -----------------------------------------------------------------------
    lx = random.uniform(-1.0, 1.0)
    ly = random.uniform(-0.2, 1.0)
    lz = random.uniform(-1.0, -0.1)

    temp_choice = random.choice(['warm', 'neutral', 'cool'])
    if temp_choice == 'warm':    lt = [1.3, 1.0, 0.7]
    elif temp_choice == 'cool':  lt = [0.8, 0.9, 1.3]
    else:                        lt = [1.0, 1.0, 1.0]

    intensity = random.uniform(1.5, 6.0)
    key_irr   = [lt[0] * intensity, lt[1] * intensity, lt[2] * intensity]
    fill_intensity = intensity * random.uniform(0.1, 0.4)
    fill_irr  = [fill_intensity, fill_intensity, fill_intensity]

    # -----------------------------------------------------------------------
    # Randomise Mesh Rotation
    # -----------------------------------------------------------------------
    yaw   = random.uniform(0.0,   360.0)
    pitch = random.uniform(-20.0,  20.0)

    mesh_transform = (
        mi.ScalarTransform4f.translate([cx, cy, cz])
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=yaw)
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=pitch)
        @ mi.ScalarTransform4f.translate([-cx, -cy, -cz])
    )

    # Serialise the 4x4 matrix to a plain Python list for JSON storage.
    # Neural Contours Geometry Branch needs this to reconstruct the exact view.
    mesh_transform_matrix = mesh_transform.matrix.numpy().tolist()

    # -----------------------------------------------------------------------
    # Randomise Material
    # -----------------------------------------------------------------------
    roughness  = random.uniform(0.1, 0.9)
    r, g, b    = (random.uniform(0.1, 0.9) for _ in range(3))
    sheen      = random.uniform(0.0, 1.0)
    sheen_tint = random.uniform(0.0, 1.0)
    anisotropic= random.uniform(0.0, 0.8)
    specular   = random.uniform(0.0, 1.0)

    material_desc = "shiny silk" if roughness < 0.4 else "matte wool"
    texture_type = "Coarse Wool texture" if "wool" in material_desc else "Silk texture"
    keyword = "wool pattern" if "wool" in material_desc else "silk pattern"
    prompt = (
        f"a photorealistic 3D render of a {material_desc} cloth, "
        "physical rendering, detailed fabric folds"
    )

    # -----------------------------------------------------------------------
    # Build Mitsuba Scene
    #
    # AOV integrator exports three channels simultaneously:
    #   • path          → beauty RGB (channels 0-2)
    #   • depth:depth   → linear distance from camera (channel 3)
    #   • normals:sh_normal → shading normals XYZ (channels 4-6)
    #
    # This gives the Neural Contours Image Translation Branch everything it
    # needs for the view-dependent shape representation without extra renders.
    # -----------------------------------------------------------------------
    scene = mi.load_dict({
        'type': 'scene',

        'integrator': {
            'type': 'aov',
            # Format: "<output_name>:<aov_type>"
            # depth    → floating-point distance in scene units
            # sh_normal → shading-space normal vector (X, Y, Z)
            'aovs': 'depth:depth,normals:sh_normal',
            'my_path': {
                'type': 'path',
                'max_depth': 6,
            },
        },

        'sensor': {
            'type': 'perspective',
            'fov': fov,
            'to_world': mi.ScalarTransform4f.look_at(
                origin=cam_origin,
                target=cam_target,
                up=[0, 1, 0],
            ),
            'film': {
                'type': 'hdrfilm',
                'width':  512,
                'height': 512,
                # rgba  = 4 channels (beauty + alpha)
                # + 1 depth  channel
                # + 3 normal channels
                # Total = 8 channels → use 'rgba' base + AOV extras
                'pixel_format': 'rgba',
            },
            'sampler': {
                'type': 'independent',
                'sample_count': 128,
            },
        },

        # Key light
        'key_light': {
            'type': 'directional',
            'direction': [lx, ly, lz],
            'irradiance': {'type': 'rgb', 'value': key_irr},
        },

        # Fill light — opposite direction, dimmer
        'fill_light': {
            'type': 'directional',
            'direction': [-lx, ly, -lz],
            'irradiance': {'type': 'rgb', 'value': fill_irr},
        },

        'cloth_object': {
            'type': 'obj',
            'filename': current_mesh_path,
            'to_world': mesh_transform,
            'bsdf': {
                'type': 'principled',
                'base_color':  {'type': 'rgb', 'value': [r, g, b]},
                'roughness':   roughness,
                'sheen':       sheen,
                'sheen_tint':  sheen_tint,
                'anisotropic': anisotropic,
                'specular':    specular,
            },
        },
    })

    # -----------------------------------------------------------------------
    # Render → multi-channel NumPy tensor (H, W, C)
    #
    # Expected channel layout from the AOV integrator:
    #   [0..2]  RGB beauty (from the 'path' sub-integrator)
    #   [3]     Alpha
    #   [4]     Depth         (depth:depth)
    #   [5..7]  Normals XYZ   (normals:sh_normal)
    # -----------------------------------------------------------------------
    render_np = np.array(mi.render(scene))
    print(f"  [{i+1:>3}/{NUM_SAMPLES}] Raw tensor shape: {render_np.shape}")

    # ---- Save beauty render (8-bit RGBA PNG) ----
    # The alpha channel (channel 3) is 1.0 wherever the cloth exists and 0.0
    # for the pure-black background — independent of shading or shadow darkness.
    # generate_sketches.py reads this alpha to build a perfect object mask so
    # shadow areas are never incorrectly excluded from the silhouette.
    beauty_np    = np.clip(render_np[:, :, :3], 0.0, 1.0)
    alpha_np     = np.clip(render_np[:, :, 3],  0.0, 1.0) if render_np.shape[2] >= 4 else np.ones(render_np.shape[:2], np.float32)
    beauty_uint8 = (beauty_np * 255).astype(np.uint8)
    alpha_uint8  = (alpha_np  * 255).astype(np.uint8)
    bgr_uint8    = cv2.cvtColor(beauty_uint8, cv2.COLOR_RGB2BGR)
    bgra_uint8   = np.dstack([bgr_uint8, alpha_uint8])
    cv2.imwrite(render_path, bgra_uint8)

    # ---- Save depth map (float32 .npy — lossless, no codec needed) ----
    if render_np.shape[2] >= 5:
        depth_channel = render_np[:, :, 4]     # shape (H, W)
    else:
        print(f"  [WARNING] Depth AOV not found for {frame_str}; saving zero depth.")
        depth_channel = np.zeros(render_np.shape[:2], dtype=np.float32)
    np.save(depth_path, depth_channel.astype(np.float32))

    # ---- Save surface normals (float32 .npy, shape (H,W,3), range [-1,1]) ----
    if render_np.shape[2] >= 8:
        normals_np = render_np[:, :, 5:8]      # shape (H, W, 3)
    else:
        print(f"  [WARNING] Normals AOV not found for {frame_str}; saving zero normals.")
        normals_np = np.zeros((*render_np.shape[:2], 3), dtype=np.float32)
    np.save(normals_path, normals_np.astype(np.float32))

    # -----------------------------------------------------------------------
    # Record per-frame metadata
    #
    # The Neural Contours Geometry Branch computes Suggestive Contours and
    # Apparent Ridges by re-projecting the 3D mesh from the exact camera
    # viewpoint. We store everything it needs:
    #   • cam_origin        – where the camera sits in world space
    #   • cam_target        – what the camera looks at
    #   • fov_deg           – field of view used for this frame
    #   • mesh_transform    – 4×4 matrix that was applied to the .obj mesh,
    #                         so the geometry branch can reproduce the pose
    # -----------------------------------------------------------------------
    metadata_records.append({
        "frame":              frame_str,
        "file_name":          f"renders/render_{frame_str}.png",
        "depth_image":        f"depth/depth_{frame_str}.npy",
        "normals_image":      f"normals/normals_{frame_str}.npy",
        "conditioning_image": f"conditioning/conditioning_{frame_str}.png",
        "text":               f"Wool Scarf, {texture_type}, {prompt}",
        "keyword":             keyword,
        "texture_type":        texture_type,
        "mesh_file":          os.path.relpath(current_mesh_path, DATASET_DIR),
        # ── Camera parameters (Neural Contours Geometry Branch) ──
        "cam_origin":         cam_origin,
        "cam_target":         cam_target,
        "fov_deg":            fov,
        # 4×4 row-major matrix as a list-of-lists (JSON serialisable)
        "mesh_transform":     mesh_transform_matrix,
    })

    print(
        f"  [{i+1:>3}/{NUM_SAMPLES}] Saved {frame_str} "
        f"| Mesh: {mesh_name[:20]} | {material_desc}"
    )

# ---------------------------------------------------------------------------
# 3. WRITE / APPEND METADATA
# ---------------------------------------------------------------------------
metadata_path = os.path.join(DATASET_DIR, "metadata.jsonl")
with open(metadata_path, 'w') as f:
    for record in metadata_records:
        f.write(json.dumps(record) + '\n')

print(f"\n✓ Done!  {NUM_SAMPLES} render sets saved.")
print(f"  • beauty PNGs  → {RENDERS_DIR}")
print(f"  • depth .npy   → {DEPTH_DIR}")
print(f"  • normal .npy  → {NORMALS_DIR}")
print(f"  • metadata     → {metadata_path}")
print("\nNext: run  python generate_sketches.py  to run Neural Contours inference.")