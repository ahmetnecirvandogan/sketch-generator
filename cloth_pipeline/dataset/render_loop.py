"""Mitsuba render loop for Stage 1 (dataset generation)."""

import glob
import json
import os
import random

import cv2
import mitsuba as mi
import numpy as np

from cloth_pipeline.dataset.textures import generate_random_albedo_map
from cloth_pipeline.paths import (
    DATASET_DIR,
    DEPTH_DIR,
    MASKS_DIR,
    MESHES_DIR,
    METADATA_PATH,
    NORMALS_DIR,
    RENDERS_DIR,
    ensure_dataset_stage_dirs,
)

mi.set_variant("scalar_rgb")


def run_generation(num_samples: int = 3) -> None:  # bump for full training runs
    ensure_dataset_stage_dirs()

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

    # Load existing metadata so we preserve it when skipping frames
    existing_metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            for ln in f:
                if ln.strip():
                    rec = json.loads(ln)
                    existing_metadata[rec["frame"]] = rec

    metadata_records = []

    existing = sum(
        1 for j in range(num_samples)
        if os.path.exists(os.path.join(RENDERS_DIR,  f"render_{j:04d}.png"))
        and os.path.exists(os.path.join(DEPTH_DIR,   f"depth_{j:04d}.npy"))
        and os.path.exists(os.path.join(NORMALS_DIR, f"normals_{j:04d}.npy"))
        and os.path.exists(os.path.join(MASKS_DIR,   f"mask_{j:04d}.png"))
    )
    print(f"Checkpoint: {existing}/{num_samples} frames already done, resuming.\n")

    for i in range(num_samples):
        frame_str    = f"{i:04d}"
        render_path  = os.path.join(RENDERS_DIR,  f"render_{frame_str}.png")
        depth_path   = os.path.join(DEPTH_DIR,    f"depth_{frame_str}.npy")
        normals_path = os.path.join(NORMALS_DIR,  f"normals_{frame_str}.npy")
        mask_path    = os.path.join(MASKS_DIR,    f"mask_{frame_str}.png")

        # --- CHECKPOINT: skip frames that are fully complete (and have metadata) ---
        if (os.path.exists(render_path)
                and os.path.exists(depth_path)
                and os.path.exists(normals_path)
                and os.path.exists(mask_path)):
            if frame_str in existing_metadata:
                metadata_records.append(existing_metadata[frame_str])
                print(f"  [{i+1:>4}/{num_samples}] Skipping {frame_str} (already exists)")
                continue
            # No metadata for this frame — must re-render to populate it
            print(f"  [{i+1:>4}/{num_samples}] Re-rendering {frame_str} (missing metadata)")

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
        # Randomise Lighting — 1-10 lights with randomised type & direction
        #
        # Light 0 is ALWAYS a directional key light so the cloth is guaranteed
        # to receive a minimum base illumination regardless of how point/spot
        # lights happen to be positioned.  Remaining lights are random types.
        # -----------------------------------------------------------------------
        EXTRA_LIGHT_TYPES = ['directional', 'point', 'spot']
        num_lights = random.randint(1, 10)
        dist_sq = cam_dist ** 2

        lights_meta = []
        lights_dict = {}

        def _random_temperature():
            tc = random.choice(['warm', 'neutral', 'cool'])
            if tc == 'warm':    return tc, [1.3, 1.0, 0.7]
            elif tc == 'cool':  return tc, [0.8, 0.9, 1.3]
            return tc, [1.0, 1.0, 1.0]

        for li in range(num_lights):
            light_type = 'directional' if li == 0 else random.choice(EXTRA_LIGHT_TYPES)
            temp_choice, lt = _random_temperature()

            if light_type == 'directional':
                if li == 0:
                    key_intensity = random.uniform(2.0, 5.0)
                else:
                    key_intensity = random.uniform(0.5, 3.0)
                dx = random.uniform(-1.0, 1.0)
                dy = random.uniform(-0.5, 1.0)
                dz = random.uniform(-1.0, -0.1)
                irr = [lt[0] * key_intensity, lt[1] * key_intensity, lt[2] * key_intensity]
                lights_dict[f'light_{li}'] = {
                    'type': 'directional',
                    'direction': [dx, dy, dz],
                    'irradiance': {'type': 'rgb', 'value': irr},
                }
                lights_meta.append({
                    'type': 'directional', 'direction': [dx, dy, dz],
                    'irradiance': irr, 'temperature': temp_choice,
                })

            elif light_type == 'point':
                px = cx + random.uniform(-max_extent, max_extent)
                py = cy + random.uniform(max_extent * 0.3, max_extent * 2)
                pz = cz + random.uniform(-max_extent, max_extent * 0.5)
                base_power = random.uniform(2.0, 6.0) * dist_sq * 0.5
                power = [lt[0] * base_power, lt[1] * base_power, lt[2] * base_power]
                lights_dict[f'light_{li}'] = {
                    'type': 'point',
                    'position': [px, py, pz],
                    'intensity': {'type': 'rgb', 'value': power},
                }
                lights_meta.append({
                    'type': 'point', 'position': [px, py, pz],
                    'intensity': power, 'temperature': temp_choice,
                })

            else:  # spot — always aimed at the mesh centre
                sx = cx + random.uniform(-max_extent * 1.5, max_extent * 1.5)
                sy = cy + random.uniform(max_extent * 0.5, max_extent * 2.5)
                sz = cz + random.uniform(-max_extent * 1.5, max_extent * 0.5)
                spot_target = [
                    cx + random.uniform(-max_extent * 0.2, max_extent * 0.2),
                    cy + random.uniform(-max_extent * 0.2, max_extent * 0.2),
                    cz + random.uniform(-max_extent * 0.2, max_extent * 0.2),
                ]
                cutoff_angle = random.uniform(20.0, 60.0)
                beam_width = cutoff_angle * random.uniform(0.5, 0.9)
                base_power = random.uniform(2.0, 6.0) * dist_sq * 0.8
                spot_power = [lt[0] * base_power, lt[1] * base_power, lt[2] * base_power]
                lights_dict[f'light_{li}'] = {
                    'type': 'spot',
                    'to_world': mi.ScalarTransform4f.look_at(
                        origin=[sx, sy, sz],
                        target=spot_target,
                        up=[0, 1, 0],
                    ),
                    'cutoff_angle': cutoff_angle,
                    'beam_width': beam_width,
                    'intensity': {'type': 'rgb', 'value': spot_power},
                }
                lights_meta.append({
                    'type': 'spot', 'position': [sx, sy, sz],
                    'target': spot_target,
                    'cutoff_angle': cutoff_angle, 'beam_width': beam_width,
                    'intensity': spot_power, 'temperature': temp_choice,
                })

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
        # Randomise BRDF material preset + procedural albedo pattern
        #
        # Each fabric type defines a physically-coherent range for every
        # principled BSDF parameter.  One preset is picked at random per frame
        # and each parameter is jittered within its range for variety.
        # -----------------------------------------------------------------------
        FABRIC_PRESETS = {
            'silk': {
                'roughness':   (0.08, 0.25), 'specular':    (0.6, 1.0),
                'sheen':       (0.5,  0.9),  'sheen_tint':  (0.3, 0.7),
                'anisotropic': (0.3,  0.7),  'spec_trans':  (0.0, 0.05),
                'clearcoat':   (0.0,  0.1),
            },
            'satin': {
                'roughness':   (0.05, 0.15), 'specular':    (0.7, 1.0),
                'sheen':       (0.3,  0.6),  'sheen_tint':  (0.2, 0.5),
                'anisotropic': (0.4,  0.8),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.05, 0.2),
            },
            'wool': {
                'roughness':   (0.6,  0.95), 'specular':    (0.05, 0.3),
                'sheen':       (0.5,  1.0),  'sheen_tint':  (0.5, 1.0),
                'anisotropic': (0.0,  0.15), 'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.0,  0.0),
            },
            'cotton': {
                'roughness':   (0.5,  0.8),  'specular':    (0.1, 0.35),
                'sheen':       (0.2,  0.5),  'sheen_tint':  (0.3, 0.6),
                'anisotropic': (0.0,  0.1),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.0,  0.0),
            },
            'velvet': {
                'roughness':   (0.55, 0.85), 'specular':    (0.1, 0.3),
                'sheen':       (0.8,  1.0),  'sheen_tint':  (0.7, 1.0),
                'anisotropic': (0.0,  0.1),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.0,  0.05),
            },
            'linen': {
                'roughness':   (0.6,  0.9),  'specular':    (0.05, 0.2),
                'sheen':       (0.1,  0.3),  'sheen_tint':  (0.2, 0.5),
                'anisotropic': (0.05, 0.2),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.0,  0.0),
            },
            'denim': {
                'roughness':   (0.7,  0.95), 'specular':    (0.05, 0.2),
                'sheen':       (0.1,  0.3),  'sheen_tint':  (0.4, 0.8),
                'anisotropic': (0.1,  0.3),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.0,  0.0),
            },
            'chiffon': {
                'roughness':   (0.15, 0.35), 'specular':    (0.3, 0.6),
                'sheen':       (0.1,  0.3),  'sheen_tint':  (0.1, 0.3),
                'anisotropic': (0.0,  0.1),  'spec_trans':  (0.2, 0.5),
                'clearcoat':   (0.0,  0.0),
            },
            'cashmere': {
                'roughness':   (0.5,  0.75), 'specular':    (0.15, 0.4),
                'sheen':       (0.6,  0.9),  'sheen_tint':  (0.5, 0.8),
                'anisotropic': (0.0,  0.1),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.0,  0.0),
            },
            'leather': {
                'roughness':   (0.3,  0.6),  'specular':    (0.3, 0.7),
                'sheen':       (0.0,  0.15), 'sheen_tint':  (0.0, 0.3),
                'anisotropic': (0.0,  0.1),  'spec_trans':  (0.0, 0.0),
                'clearcoat':   (0.15, 0.5),
            },
        }

        material_desc = random.choice(list(FABRIC_PRESETS.keys()))
        preset = FABRIC_PRESETS[material_desc]

        def _sample(key):
            lo, hi = preset[key]
            return random.uniform(lo, hi)

        roughness   = _sample('roughness')
        specular    = _sample('specular')
        sheen       = _sample('sheen')
        sheen_tint  = _sample('sheen_tint')
        anisotropic = _sample('anisotropic')
        spec_trans  = _sample('spec_trans')
        clearcoat   = _sample('clearcoat')

        keyword = f"{material_desc} pattern"

        tex_path, pattern_name, pattern_params = generate_random_albedo_map(frame_str)

        # Analyse mesh UV range to choose a tiling factor that makes the pattern
        # repeat at a visually sensible scale regardless of how the OBJ was UV-unwrapped.
        uv_u_range, uv_v_range = 1.0, 1.0
        try:
            with open(current_mesh_path) as _mf:
                _us, _vs = [], []
                for _line in _mf:
                    if _line.startswith('vt '):
                        _parts = _line.split()
                        _us.append(float(_parts[1]))
                        _vs.append(float(_parts[2]))
                if _us:
                    uv_u_range = max(_us) - min(_us)
                    uv_v_range = max(_vs) - min(_vs)
        except Exception:
            pass

        desired_repeats = random.uniform(3.0, 6.0)
        tile_u = desired_repeats / max(uv_u_range, 0.01)
        tile_v = desired_repeats / max(uv_v_range, 0.01)

        base_color_spec = {
            'type': 'bitmap',
            'filename': tex_path,
            'wrap_mode': 'repeat',
            'to_uv': mi.ScalarTransform3f.scale([tile_u, tile_v]),
        }

        prompt = (
            f"a photorealistic 3D render of a {material_desc} cloth with "
            f"{pattern_name} pattern, physical rendering, detailed fabric folds"
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
        scene_dict = {
            'type': 'scene',

            'integrator': {
                'type': 'aov',
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
                    'pixel_format': 'rgba',
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': 128,
                },
            },

            'cloth_object': {
                'type': 'obj',
                'filename': current_mesh_path,
                'to_world': mesh_transform,
                'bsdf': {
                    'type': 'principled',
                    'base_color':  base_color_spec,
                    'roughness':   roughness,
                    'specular':    specular,
                    'sheen':       sheen,
                    'sheen_tint':  sheen_tint,
                    'anisotropic': anisotropic,
                    'spec_trans':  spec_trans,
                    'clearcoat':   clearcoat,
                },
            },
        }

        scene_dict.update(lights_dict)

        # -----------------------------------------------------------------------
        # Render with brightness guarantee
        #
        # After the first render we check the mean luminance of object pixels
        # (alpha > 0).  If it falls below MIN_BRIGHTNESS the cloth is not
        # visually readable, so we inject a rescue directional light aimed
        # straight at the camera target and re-render.  Up to MAX_RETRIES
        # attempts; each retry doubles the rescue light intensity.
        # -----------------------------------------------------------------------
        MIN_BRIGHTNESS = 0.08   # minimum mean luminance (0-1 float scale)
        MAX_RETRIES    = 3
        rescue_boost   = 0

        for attempt in range(1 + MAX_RETRIES):
            scene = mi.load_dict(scene_dict)
            render_np = np.array(mi.render(scene))

            beauty_tmp = np.clip(render_np[:, :, :3], 0.0, 1.0)
            alpha_tmp  = render_np[:, :, 3] if render_np.shape[2] >= 4 else np.ones(render_np.shape[:2])
            obj_pixels = beauty_tmp[alpha_tmp > 0.5]

            if obj_pixels.size == 0:
                mean_lum = 0.0
            else:
                mean_lum = float(np.mean(obj_pixels[:, 0] * 0.2126
                                         + obj_pixels[:, 1] * 0.7152
                                         + obj_pixels[:, 2] * 0.0722))

            if mean_lum >= MIN_BRIGHTNESS:
                break

            rescue_boost += 1
            rescue_int = 3.0 * (2 ** rescue_boost)
            rescue_key = f'rescue_light_{rescue_boost}'
            scene_dict[rescue_key] = {
                'type': 'directional',
                'direction': [0.0, -0.3, -1.0],
                'irradiance': {'type': 'rgb', 'value': [rescue_int, rescue_int, rescue_int]},
            }
            lights_meta.append({
                'type': 'directional', 'direction': [0.0, -0.3, -1.0],
                'irradiance': [rescue_int, rescue_int, rescue_int],
                'temperature': 'neutral', 'rescue': True,
            })
            print(f"  [{i+1:>3}/{num_samples}] Brightness {mean_lum:.3f} < {MIN_BRIGHTNESS} — "
                  f"adding rescue light (attempt {rescue_boost}, intensity {rescue_int:.1f})")

        print(f"  [{i+1:>3}/{num_samples}] Raw tensor shape: {render_np.shape} | brightness: {mean_lum:.3f}")

        # ---- Save beauty render (8-bit BGRA PNG with visible background) ----
        # Composite the cloth over a solid light-gray backdrop so renders look
        # presentation-ready out of the box (no transparent checkerboard / black).
        #
        # Save cloth coverage to a separate mask file for Stage 2 segmentation.
        beauty_np    = np.clip(render_np[:, :, :3], 0.0, 1.0)
        alpha_np     = np.clip(render_np[:, :, 3],  0.0, 1.0) if render_np.shape[2] >= 4 else np.ones(render_np.shape[:2], np.float32)
        bg_rgb       = np.array([0.82, 0.82, 0.82], dtype=np.float32)  # ~RGB(209,209,209)
        comp_np      = beauty_np * alpha_np[..., None] + bg_rgb * (1.0 - alpha_np[..., None])
        comp_uint8   = (comp_np * 255).astype(np.uint8)
        bgr_uint8    = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2BGR)
        alpha_uint8  = np.full(alpha_np.shape, 255, dtype=np.uint8)
        bgra_uint8   = np.dstack([bgr_uint8, alpha_uint8])
        cv2.imwrite(render_path, bgra_uint8)
        cv2.imwrite(mask_path, (alpha_np * 255).astype(np.uint8))

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
            "mask_image":         f"masks/mask_{frame_str}.png",
            "conditioning_image": f"conditioning/conditioning_{frame_str}.png",
            "text":               f"Cloth Scarf, {material_desc} material, {prompt}",
            "keyword":             keyword,
            "mesh_file":          os.path.relpath(current_mesh_path, DATASET_DIR),
            # ── Camera parameters (Neural Contours Geometry Branch) ──
            "cam_origin":         cam_origin,
            "cam_target":         cam_target,
            "fov_deg":            fov,
            # 4×4 row-major matrix as a list-of-lists (JSON serialisable)
            "mesh_transform":     mesh_transform_matrix,
            # ── Lighting configuration ──
            "num_lights":         num_lights,
            "lights":             lights_meta,
            # ── Material properties ──
            "material_type":      material_desc,
            "material_props": {
                "roughness":   round(roughness,   4),
                "specular":    round(specular,     4),
                "sheen":       round(sheen,        4),
                "sheen_tint":  round(sheen_tint,   4),
                "anisotropic": round(anisotropic,  4),
                "spec_trans":  round(spec_trans,   4),
                "clearcoat":   round(clearcoat,    4),
            },
            # ── Procedural albedo map (surface pattern, not BRDF material) ──
            "pattern_name":       pattern_name,
            "pattern_params":     pattern_params,
            "albedo_map":         f"textures/texture_{frame_str}.png",
            "albedo_tiling":      [tile_u, tile_v],
        })

        light_types_str = ', '.join(lm['type'] for lm in lights_meta)
        print(
            f"  [{i+1:>3}/{num_samples}] Saved {frame_str} "
            f"| Mesh: {mesh_name[:20]} | {material_desc:9s} "
            f"| {num_lights} light(s): [{light_types_str}] "
            f"| pattern: {pattern_name}"
        )

    # ---------------------------------------------------------------------------
    # Write metadata.jsonl
    # ---------------------------------------------------------------------------

    with open(METADATA_PATH, 'w') as f:
        for record in metadata_records:
            f.write(json.dumps(record) + '\n')

    print(f"\n✓ Done!  {num_samples} render sets saved.")
    print(f"  • beauty PNGs  → {RENDERS_DIR}")
    print(f"  • depth .npy   → {DEPTH_DIR}")
    print(f"  • normal .npy  → {NORMALS_DIR}")
    print(f"  • metadata     → {METADATA_PATH}")
    print("\nNext: run  python generate_sketches.py  to run Neural Contours inference.")

