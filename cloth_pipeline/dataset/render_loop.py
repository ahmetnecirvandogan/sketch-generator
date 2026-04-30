"""Mitsuba render loop for Stage 1 (dataset generation)."""

from __future__ import annotations

import glob
import json
import os
import random
import sys

import cv2
import mitsuba as mi
import numpy as np

from cloth_pipeline.dataset.textures import generate_random_albedo_map
from cloth_pipeline.paths import (
    BASE_DIR,
    DATASET_DIR,
    DEPTH_DIR,
    FRONT_PREVIEW_DIR,
    MASKS_DIR,
    MESHES_DIR,
    METADATA_PATH,
    NORMALS_DIR,
    OUTPUTS_DIR,
    RENDERS_DIR,
    ensure_dataset_stage_dirs,
    ensure_front_preview_dir,
    output_sample_dir,
    sample_dir_components,
)

mi.set_variant("scalar_rgb")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Default film size / samples are tuned for laptops where Mitsuba often prints
# ``jitc_llvm_init(): LLVM API initialization failed`` — JIT falls back to a
# very slow path; 768² @ 512 spp can look “hung” for tens of minutes on frame 0.
# Box rfilter stays sharp; raise quality with env: NECH_FILM_W=768 NECH_SAMPLES=384
FILM_WIDTH = max(64, _env_int("NECH_FILM_W", 512))
FILM_HEIGHT = max(64, _env_int("NECH_FILM_H", 512))
PATH_SAMPLE_COUNT = max(8, _env_int("NECH_SAMPLES", 256))

# Optional per-mesh camera tweaks (OBJ stem without extension). The default front
# view uses a small +Y lift; very flat / looped pieces can read as a “top-down
# into the ring” shot — lower `front_y_lift` (or 0) for a more head-on view.
MESH_FRONT_CAMERA_OVERRIDES: dict[str, dict[str, float]] = {
    # Flat loop reads as “looking down the ring” with default lift; pure +Z is more head-on.
    # Softer key in previews only — full dataset still uses random key strength per frame.
    "10152_WomensScarf_v01_L3": {"front_y_lift": 0.0, "preview_key_scale": 0.55},
}


def _object_mask_from_depth(depth: np.ndarray) -> np.ndarray:
    """True where the primary ray hit the mesh (not environment / miss)."""
    d = np.asarray(depth, dtype=np.float64)
    return (
        np.isfinite(d)
        & (d > 1e-5)
        & (d < 1e8)
    ).astype(np.float32)


def _front_y_lift_for_mesh(mesh_stem: str) -> float:
    base = float(os.environ.get("NECH_FRONT_ELEV", "0.22"))
    ov = MESH_FRONT_CAMERA_OVERRIDES.get(mesh_stem, {})
    if "front_y_lift" in ov:
        return float(ov["front_y_lift"])
    return base


def _front_camera_offset_dir(y_lift: float) -> np.ndarray:
    """Unit direction in the +Z hemisphere with given world +Y lift."""
    d = np.array([0.0, y_lift, 1.0], dtype=np.float64)
    d /= max(1e-12, float(np.linalg.norm(d)))
    return d


def front_camera_look_at(
    cx: float,
    cy: float,
    cz: float,
    cam_dist: float,
    *,
    mesh_stem: str = "",
) -> tuple[list[float], list[float]]:
    """Fixed front view: camera offset from bbox center; optional per-mesh y lift."""
    y_lift = _front_y_lift_for_mesh(mesh_stem)
    d = _front_camera_offset_dir(y_lift)
    ox = cx + float(d[0]) * cam_dist
    oy = cy + float(d[1]) * cam_dist
    oz = cz + float(d[2]) * cam_dist
    return [ox, oy, oz], [cx, cy, cz]


def _suppress_mc_sparkles(rgb: np.ndarray) -> np.ndarray:
    """
    Reduce path-tracer fireflies (“glitter”): pixels much brighter than a local
    5×5 median luminance are scaled down. Wide specular lobes stay intact.
    """
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    lum = np.max(rgb, axis=2)
    lum_u8 = (lum * 255.0).astype(np.uint8)
    med = cv2.medianBlur(lum_u8, 5).astype(np.float32) / 255.0
    allowed = np.clip(med * 4.0 + 0.04, 0.06, 1.0)
    scale = np.where(lum > allowed, allowed / np.maximum(lum, 1e-6), 1.0).astype(np.float32)
    return np.clip(rgb * scale[..., None], 0.0, 1.0)


def run_generation(
    materials_per_mesh: int = 3,
    lightings_per_material: int = 2,
) -> None:
    """Three-level deterministic loop: mesh × (material+pattern) × lighting.

    For each mesh, ``materials_per_mesh`` (material, pattern) combinations are
    sampled. For each, ``lightings_per_material`` lighting setups are rendered.
    Mesh, material and pattern are *fixed* across the lighting siblings, so
    their albedo/normal/roughness GTs are identical and only the sketches
    (which encode highlight/shadow positions) differ — the disentanglement
    signal we want for training.

    Total samples = N_meshes × materials_per_mesh × lightings_per_material.
    """
    ensure_dataset_stage_dirs()

    OUTPUT_MESHES_DIR = os.path.join(BASE_DIR, "output_meshes")
    # Render output_meshes first so new generations can be checked immediately
    mesh_files = (
        sorted(glob.glob(os.path.join(OUTPUT_MESHES_DIR, "*.obj"))) +
        sorted(glob.glob(os.path.join(MESHES_DIR, "*.obj")))
    )
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

    samples_per_mesh = materials_per_mesh * lightings_per_material
    num_samples = len(mesh_files) * samples_per_mesh
    print(
        f"Generating {len(mesh_files)} meshes × {materials_per_mesh} materials "
        f"× {lightings_per_material} lightings = {num_samples} samples\n"
    )

    # (mesh_idx, material_idx) → cached material + pattern + texture, so all
    # `lightings_per_material` siblings of the same (mesh, material) combo
    # share bit-identical PBR inputs. Reset per run; in-memory only.
    material_pattern_cache: dict[tuple[int, int], dict] = {}

    # Load existing metadata so we preserve it when skipping frames
    existing_metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            for ln in f:
                if ln.strip():
                    rec = json.loads(ln)
                    existing_metadata[rec["frame"]] = rec

    # Pre-populate the material+pattern cache from existing metadata so that a
    # mid-run restart (e.g. after a Mitsuba segfault) reuses the SAME material
    # and texture for the second lighting sibling instead of re-rolling. Without
    # this, the (mesh, material+pattern) → lighting-pair invariant is violated.
    for rec in existing_metadata.values():
        if "mesh_idx" not in rec or "material_idx" not in rec:
            continue
        cache_key = (int(rec["mesh_idx"]), int(rec["material_idx"]))
        if cache_key in material_pattern_cache:
            continue
        tex_rel = rec.get("albedo_map")
        tex_path = os.path.join(DATASET_DIR, tex_rel) if tex_rel else None
        if not tex_path or not os.path.exists(tex_path):
            continue
        props = rec.get("material_props") or {}
        tiling = rec.get("albedo_tiling") or [1.0, 1.0]
        material_pattern_cache[cache_key] = {
            "material_desc":   rec.get("material_type"),
            "roughness":       float(props.get("roughness", 0.5)),
            "specular":        float(props.get("specular", 0.5)),
            "sheen":           float(props.get("sheen", 0.5)),
            "sheen_tint":      float(props.get("sheen_tint", 0.5)),
            "anisotropic":     float(props.get("anisotropic", 0.0)),
            "spec_trans":      float(props.get("spec_trans", 0.0)),
            "clearcoat":       float(props.get("clearcoat", 0.0)),
            "tex_path":        tex_path,
            "pattern_name":    rec.get("pattern_name", "solid"),
            "pattern_params":  rec.get("pattern_params", {}),
            "tile_u":          float(tiling[0]),
            "tile_v":          float(tiling[1]),
        }

    metadata_records = []

    existing = sum(
        1 for j in range(num_samples)
        if os.path.exists(os.path.join(RENDERS_DIR,  f"render_{j:04d}.png"))
        and os.path.exists(os.path.join(DEPTH_DIR,   f"depth_{j:04d}.npy"))
        and os.path.exists(os.path.join(NORMALS_DIR, f"normals_{j:04d}.npy"))
        and os.path.exists(os.path.join(MASKS_DIR,   f"mask_{j:04d}.png"))
    )
    print(f"Checkpoint: {existing}/{num_samples} frames already done, resuming.\n")
    print(
        f"Film {FILM_WIDTH}×{FILM_HEIGHT}, {PATH_SAMPLE_COUNT} samples/pixel. "
        f"Override: NECH_FILM_W, NECH_FILM_H, NECH_SAMPLES.",
        flush=True,
    )
    print(
        "If LLVM init failed above, the first frame can sit silent a long time "
        "while Mitsuba compiles; lower NECH_SAMPLES or fix the Mitsuba/LLVM install.\n",
        flush=True,
    )

    for i in range(num_samples):
        frame_str    = f"{i:04d}"
        render_path  = os.path.join(RENDERS_DIR,  f"render_{frame_str}.png")
        depth_path   = os.path.join(DEPTH_DIR,    f"depth_{frame_str}.npy")
        normals_path = os.path.join(NORMALS_DIR,  f"normals_{frame_str}.npy")
        mask_path    = os.path.join(MASKS_DIR,    f"mask_{frame_str}.png")

        # --- CHECKPOINT: skip when intermediates + recorded PBR sample dir
        # all exist. The mesh+material+pattern combination chosen for a frame
        # is random, so we read the recorded paths straight from metadata. ---
        if (frame_str in existing_metadata
                and os.path.exists(render_path)
                and os.path.exists(depth_path)
                and os.path.exists(normals_path)
                and os.path.exists(mask_path)):
            rec = existing_metadata[frame_str]
            rec_paths = [rec.get("pbr_albedo"), rec.get("pbr_normal"), rec.get("pbr_roughness")]
            if all(rec_paths) and all(
                os.path.exists(os.path.join(BASE_DIR, p)) for p in rec_paths
            ):
                metadata_records.append(rec)
                print(f"  [{i+1:>4}/{num_samples}] Skipping {frame_str} (already exists)")
                continue
            print(f"  [{i+1:>4}/{num_samples}] Re-rendering {frame_str} (PBR outputs missing)")

        print(
            f"  [{i+1:>4}/{num_samples}] Frame {frame_str}: setup (mesh, lights, materials)…",
            flush=True,
        )

        # -----------------------------------------------------------------------
        # Three-level index: outer mesh / middle material / inner lighting.
        # Optional env override pins one frame to a specific mesh stem.
        # -----------------------------------------------------------------------
        mesh_idx = i // samples_per_mesh
        within_mesh = i % samples_per_mesh
        material_idx = within_mesh // lightings_per_material
        lighting_idx = within_mesh % lightings_per_material

        only_frame_env = os.environ.get("NECH_ONLY_FRAME", "").strip()
        force_stem = os.environ.get("NECH_FORCE_MESH_STEM", "").strip()
        pin_idx: int | None = None
        if only_frame_env != "":
            try:
                pin_idx = int(only_frame_env)
            except ValueError:
                pin_idx = None
        if pin_idx is not None and force_stem != "" and i == pin_idx:
            forced_path = os.path.join(MESHES_DIR, f"{force_stem}.obj")
            if not os.path.isfile(forced_path):
                raise FileNotFoundError(
                    f"NECH_FORCE_MESH_STEM: mesh not found: {forced_path}"
                )
            current_mesh_path = forced_path
        else:
            current_mesh_path = mesh_files[mesh_idx]
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
        # Fixed front camera (+Z hemisphere, slight +Y lift); mesh is not spun.
        cam_origin, cam_target = front_camera_look_at(
            cx, cy, cz, cam_dist, mesh_stem=mesh_name
        )

        # -----------------------------------------------------------------------
        # Lighting — exactly 1 constant environment + 1 directional key
        #
        # The environment emitter gives uniform fill so the object stays readable.
        # The directional uses a random unit direction each frame for varied
        # highlights and shadows. No point/spot lights.
        # -----------------------------------------------------------------------
        num_lights = 2

        # Environment must stay achromatic: tinted constant env is what the camera
        # sees on rays that miss the cloth, and film alpha stays 1 there — so warm/cool
        # tints were turning the whole backdrop brown/blue instead of composited gray.
        # Optional mild tint on the directional only (highlights), not on fill.
        def _key_temperature_tint():
            tc = random.choice(['warm', 'neutral', 'cool'])
            if tc == 'warm':
                return tc, [1.08, 1.0, 0.92]
            if tc == 'cool':
                return tc, [0.92, 0.96, 1.06]
            return tc, [1.0, 1.0, 1.0]

        temp_key, tint_key = _key_temperature_tint()

        # Uniform random direction on the sphere (Mitsuba: direction the emitter radiates along).
        _dir = np.random.randn(3).astype(np.float64)
        _dir /= max(1e-12, float(np.linalg.norm(_dir)))
        dx, dy, dz = float(_dir[0]), float(_dir[1]), float(_dir[2])

        # Stronger neutral fill so fabrics (especially dark/rough) stay readable.
        env_scale = random.uniform(0.45, 0.75)
        base_env_rgb = [env_scale, env_scale, env_scale]

        key_irr = random.uniform(4.5, 9.0)
        base_key_rgb = [tint_key[j] * key_irr for j in range(3)]

        lights_meta = [
            {
                'type': 'constant',
                'radiance': list(base_env_rgb),
                'temperature': 'neutral',
            },
            {
                'type': 'directional',
                'direction': [dx, dy, dz],
                'irradiance': list(base_key_rgb),
                'temperature': temp_key,
            },
        ]

        lights_dict = {
            'environment': {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': list(base_env_rgb)},
            },
            'key_light': {
                'type': 'directional',
                'direction': [dx, dy, dz],
                'irradiance': {'type': 'rgb', 'value': list(base_key_rgb)},
            },
        }

        # -----------------------------------------------------------------------
        # Canonical mesh pose (no random yaw/pitch) — fixed front camera defines the view.
        # -----------------------------------------------------------------------
        mesh_transform = mi.ScalarTransform4f.translate([0.0, 0.0, 0.0])
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

        # Material + pattern is cached per (mesh_idx, material_idx) so all
        # lighting siblings within a (mesh, material) group reuse the same
        # BRDF parameters AND the same procedural texture file → byte-identical
        # albedo/normal/roughness PBR ground truth.
        cache_key = (mesh_idx, material_idx)
        if cache_key not in material_pattern_cache:
            material_desc = random.choice(list(FABRIC_PRESETS.keys()))
            preset = FABRIC_PRESETS[material_desc]

            def _sample(key, _p=preset):
                lo, hi = _p[key]
                return random.uniform(lo, hi)

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

            material_pattern_cache[cache_key] = {
                'material_desc':   material_desc,
                'roughness':       _sample('roughness'),
                'specular':        _sample('specular'),
                'sheen':           _sample('sheen'),
                'sheen_tint':      _sample('sheen_tint'),
                'anisotropic':     _sample('anisotropic'),
                'spec_trans':      _sample('spec_trans'),
                'clearcoat':       _sample('clearcoat'),
                'tex_path':        tex_path,
                'pattern_name':    pattern_name,
                'pattern_params':  pattern_params,
                'tile_u':          desired_repeats / max(uv_u_range, 0.01),
                'tile_v':          desired_repeats / max(uv_v_range, 0.01),
            }

        mp = material_pattern_cache[cache_key]
        material_desc   = mp['material_desc']
        roughness       = mp['roughness']
        specular        = mp['specular']
        sheen           = mp['sheen']
        sheen_tint      = mp['sheen_tint']
        anisotropic     = mp['anisotropic']
        spec_trans      = mp['spec_trans']
        clearcoat       = mp['clearcoat']
        tex_path        = mp['tex_path']
        pattern_name    = mp['pattern_name']
        pattern_params  = mp['pattern_params']
        tile_u          = mp['tile_u']
        tile_v          = mp['tile_v']
        keyword         = f"{material_desc} pattern"
        prompt = (
            f"a photorealistic 3D render of a {material_desc} cloth with "
            f"{pattern_name} pattern, physical rendering, detailed fabric folds"
        )

        base_color_spec = {
            'type': 'bitmap',
            'filename': tex_path,
            'wrap_mode': 'repeat',
            'to_uv': mi.ScalarTransform3f.scale([tile_u, tile_v]),
        }

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
                # Append-only AOV order: depth(4), normals(5-7), albedo(8-10).
                # Existing channel slicing below depends on these positions.
                'aovs': 'depth:depth,normals:sh_normal,albedo:albedo',
                'my_path': {
                    'type': 'path',
                    # Low depth avoids long specular chains (fireflies / “glitter”) while
                    # still allowing one extra bounce vs pure direct (helps folds & thin
                    # fabrics). Deeper than ~4 tends to sparkle on principled+clearcoat.
                    'max_depth': 3,
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
                    'width': FILM_WIDTH,
                    'height': FILM_HEIGHT,
                    'pixel_format': 'rgba',
                    'rfilter': {'type': 'box'},
                },
                'sampler': {
                    'type': 'independent',
                    'sample_count': PATH_SAMPLE_COUNT,
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
        # If mean object luminance is below MIN_BRIGHTNESS, scale both the
        # environment and key light together (still only two emitters).
        # -----------------------------------------------------------------------
        MIN_BRIGHTNESS = 0.08   # minimum mean luminance (0-1 float scale)
        MAX_RETRIES    = 3
        brightness_scale = 1.0

        for attempt in range(1 + MAX_RETRIES):
            scene_dict['environment']['radiance']['value'] = [
                base_env_rgb[j] * brightness_scale for j in range(3)
            ]
            scene_dict['key_light']['irradiance']['value'] = [
                base_key_rgb[j] * brightness_scale for j in range(3)
            ]
            print(
                f"  [{i+1:>4}/{num_samples}] Ray tracing (attempt {attempt + 1}) — "
                f"{FILM_WIDTH}×{FILM_HEIGHT} @ {PATH_SAMPLE_COUNT} spp…",
                flush=True,
            )
            scene = mi.load_dict(scene_dict)
            render_np = np.array(mi.render(scene))
            sys.stdout.flush()

            beauty_tmp = np.clip(render_np[:, :, :3], 0.0, 1.0)
            if render_np.shape[2] >= 5:
                obj_mask = _object_mask_from_depth(render_np[:, :, 4]) > 0.5
            else:
                alpha_tmp = render_np[:, :, 3] if render_np.shape[2] >= 4 else np.ones(render_np.shape[:2])
                obj_mask = alpha_tmp > 0.5
            obj_pixels = beauty_tmp[obj_mask]

            if obj_pixels.size == 0:
                mean_lum = 0.0
            else:
                mean_lum = float(np.mean(obj_pixels[:, 0] * 0.2126
                                         + obj_pixels[:, 1] * 0.7152
                                         + obj_pixels[:, 2] * 0.0722))

            if mean_lum >= MIN_BRIGHTNESS:
                break

            brightness_scale *= 2.0
            print(f"  [{i+1:>3}/{num_samples}] Brightness {mean_lum:.3f} < {MIN_BRIGHTNESS} — "
                  f"scaling env+key ×{brightness_scale:.1f} (attempt {attempt + 1})")

        lights_meta[0]['radiance'] = [base_env_rgb[j] * brightness_scale for j in range(3)]
        lights_meta[1]['irradiance'] = [base_key_rgb[j] * brightness_scale for j in range(3)]

        print(f"  [{i+1:>3}/{num_samples}] Raw tensor shape: {render_np.shape} | brightness: {mean_lum:.3f}")

        # ---- Save beauty render (8-bit BGRA PNG with visible background) ----
        # Composite the cloth over a solid light-gray backdrop so renders look
        # presentation-ready out of the box (no transparent checkerboard / black).
        #
        # Use depth to decide cloth vs background: film alpha is often 1 for env hits,
        # which previously let tinted environment colour replace the gray backdrop.
        # Save the same mask for Stage 2 segmentation.
        beauty_np = _suppress_mc_sparkles(np.clip(render_np[:, :, :3], 0.0, 1.0))
        if render_np.shape[2] >= 5:
            mesh_alpha = _object_mask_from_depth(render_np[:, :, 4])
        else:
            mesh_alpha = np.clip(
                render_np[:, :, 3], 0.0, 1.0
            ) if render_np.shape[2] >= 4 else np.ones(render_np.shape[:2], np.float32)
        bg_rgb = np.array([0.82, 0.82, 0.82], dtype=np.float32)  # ~RGB(209,209,209)
        comp_np = beauty_np * mesh_alpha[..., None] + bg_rgb * (1.0 - mesh_alpha[..., None])
        comp_uint8 = (comp_np * 255).astype(np.uint8)
        bgr_uint8 = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2BGR)
        alpha_uint8 = np.full(mesh_alpha.shape, 255, dtype=np.uint8)
        bgra_uint8 = np.dstack([bgr_uint8, alpha_uint8])
        cv2.imwrite(render_path, bgra_uint8)
        cv2.imwrite(mask_path, (mesh_alpha * 255).astype(np.uint8))

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
        # PBR ground-truth maps → outputs/mesh_<stem>/<material>_<pattern>/view_<idx>/sample_<frame>/
        #   • albedo.png    → Mitsuba aov:albedo channels 8-10 (linear RGB), masked
        #   • normal.png    → re-encode normals_np from [-1,1] → [0,255] RGB, masked
        #   • roughness.png → uniform fill from BSDF scalar × mask (Design C, post-hoc)
        # The per-sample frame folder makes every frame collision-free, so two
        # frames picking the same (mesh, material, pattern, view) live side by
        # side as different lighting variations.
        # -----------------------------------------------------------------------
        view_idx = 0  # multi-view rendering not implemented yet
        frame_id = i
        sample_parts = sample_dir_components(
            current_mesh_path, material_desc, pattern_name, view_idx, frame_id
        )
        sample_out_dir = output_sample_dir(
            current_mesh_path, material_desc, pattern_name, view_idx, frame_id
        )
        os.makedirs(sample_out_dir, exist_ok=True)
        pbr_albedo_path    = os.path.join(sample_out_dir, "albedo.png")
        pbr_normal_path    = os.path.join(sample_out_dir, "normal.png")
        pbr_roughness_path = os.path.join(sample_out_dir, "roughness.png")

        if render_np.shape[2] >= 11:
            albedo_np = np.clip(render_np[:, :, 8:11], 0.0, 1.0)
        else:
            print(f"  [WARNING] Albedo AOV not found for {frame_str}; saving zero albedo.")
            albedo_np = np.zeros((*render_np.shape[:2], 3), dtype=np.float32)
        albedo_masked = albedo_np * mesh_alpha[..., None]
        albedo_uint8 = (albedo_masked * 255).astype(np.uint8)
        cv2.imwrite(pbr_albedo_path, cv2.cvtColor(albedo_uint8, cv2.COLOR_RGB2BGR))

        normal_encoded = ((normals_np + 1.0) * 0.5).clip(0.0, 1.0)
        normal_uint8 = (normal_encoded * 255 * mesh_alpha[..., None]).astype(np.uint8)
        cv2.imwrite(pbr_normal_path, cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2BGR))

        roughness_uint8 = np.clip(roughness * mesh_alpha * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(pbr_roughness_path, roughness_uint8)

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
            "camera_mode":        "fixed_front",
            "front_camera_y_lift": _front_y_lift_for_mesh(mesh_name),
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
            # ── Mesh / material+pattern / view / sample hierarchy ──
            "frame_id":                  frame_id,
            "mesh_idx":                  mesh_idx,
            "material_idx":              material_idx,
            "lighting_idx":              lighting_idx,
            "mesh_dir_name":             sample_parts["mesh_dir_name"],
            "material_pattern_dir_name": sample_parts["material_pattern_dir_name"],
            "view_idx":                  view_idx,
            "sample_dir_name":           sample_parts["sample_dir_name"],
            "sample_output_dir":         os.path.relpath(sample_out_dir, BASE_DIR),
            "pbr_albedo":                os.path.relpath(pbr_albedo_path,    BASE_DIR),
            "pbr_roughness":             os.path.relpath(pbr_roughness_path, BASE_DIR),
            "pbr_normal":                os.path.relpath(pbr_normal_path,    BASE_DIR),
            "sketch_path":               os.path.relpath(
                os.path.join(sample_out_dir, "sketch.png"), BASE_DIR
            ),
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
    print(f"  • beauty PNGs   → {RENDERS_DIR}")
    print(f"  • depth .npy    → {DEPTH_DIR}")
    print(f"  • normal .npy   → {NORMALS_DIR}")
    print(f"  • PBR maps      → {OUTPUTS_DIR}/mesh_<stem>/<mat>_<pat>/view_0/sample_NNNN/")
    print(f"  • metadata      → {METADATA_PATH}")
    print("\nNext: run  python generate_sketches.py  to write sketch.png next to PBR maps.")


def run_front_mesh_previews(*, only_stem: str | None = None) -> None:
    """
    One neutral render per mesh under dataset/front_previews/ using the same fixed
    front camera as the main dataset (identity mesh pose). For quick visual check
    of framing before full runs.

    If ``only_stem`` is set (OBJ basename without ``.obj``), only that mesh is rendered.
    """
    ensure_front_preview_dir()
    OUTPUT_MESHES_DIR = os.path.join(BASE_DIR, "output_meshes")
    # Render output_meshes first so new generations can be checked immediately
    mesh_files = (
        sorted(glob.glob(os.path.join(OUTPUT_MESHES_DIR, "*.obj"))) +
        sorted(glob.glob(os.path.join(MESHES_DIR, "*.obj")))
    )
    if not mesh_files:
        print(f"[ERROR] No .obj files found in {MESHES_DIR}.")
        raise SystemExit(1)
    if only_stem:
        mesh_files = [
            p
            for p in mesh_files
            if os.path.basename(p).replace(".obj", "") == only_stem
        ]
        if not mesh_files:
            print(f"[ERROR] No mesh matching stem {only_stem!r} in {MESHES_DIR}.")
            raise SystemExit(1)

    prev_w = max(64, _env_int("NECH_PREVIEW_W", 512))
    prev_h = max(64, _env_int("NECH_PREVIEW_H", 512))
    prev_spp = max(16, _env_int("NECH_PREVIEW_SAMPLES", 96))
    fov = 40.0
    target_fill = 0.70

    print(f"Front previews: {len(mesh_files)} mesh(es) → {FRONT_PREVIEW_DIR}")
    print(f"  size {prev_w}×{prev_h}, {prev_spp} spp (NECH_PREVIEW_* env to override)\n", flush=True)

    for mesh_path in mesh_files:
        stem = os.path.basename(mesh_path).replace(".obj", "")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
        out_png = os.path.join(FRONT_PREVIEW_DIR, f"{safe}_front.png")

        loaded_shape = mi.load_dict({"type": "obj", "filename": mesh_path})
        bbox = loaded_shape.bbox()
        center = bbox.center()
        extents = bbox.extents()
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        max_extent = max(float(extents[0]), float(extents[1]), float(extents[2]))
        cam_dist = max_extent / (2.0 * target_fill * np.tan(np.radians(fov / 2.0)))
        cam_origin, cam_target = front_camera_look_at(
            cx, cy, cz, cam_dist, mesh_stem=stem
        )

        env_rgb = [0.55, 0.55, 0.55]
        kd = np.array([-0.28, -0.45, -0.85], dtype=np.float64)
        kd /= max(1e-12, float(np.linalg.norm(kd)))
        key_rgb = [6.2, 6.1, 6.0]
        pk = float(MESH_FRONT_CAMERA_OVERRIDES.get(stem, {}).get("preview_key_scale", 1.0))
        key_rgb = [pk * x for x in key_rgb]

        scene_dict = {
            "type": "scene",
            "integrator": {
                "type": "aov",
                "aovs": "depth:depth,normals:sh_normal",
                "my_path": {"type": "path", "max_depth": 3},
            },
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=cam_origin,
                    target=cam_target,
                    up=[0, 1, 0],
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": prev_w,
                    "height": prev_h,
                    "pixel_format": "rgba",
                    "rfilter": {"type": "box"},
                },
                "sampler": {"type": "independent", "sample_count": prev_spp},
            },
            "cloth_object": {
                "type": "obj",
                "filename": mesh_path,
                "to_world": mi.ScalarTransform4f.translate([0.0, 0.0, 0.0]),
                "bsdf": {
                    "type": "principled",
                    "base_color": {"type": "rgb", "value": [0.62, 0.6, 0.58]},
                    "roughness": 0.62,
                    "specular": 0.35,
                    "sheen": 0.25,
                    "sheen_tint": 0.5,
                    "anisotropic": 0.0,
                    "spec_trans": 0.0,
                    "clearcoat": 0.0,
                },
            },
            "environment": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": env_rgb},
            },
            "key_light": {
                "type": "directional",
                "direction": [float(kd[0]), float(kd[1]), float(kd[2])],
                "irradiance": {"type": "rgb", "value": key_rgb},
            },
        }

        print(f"  … {safe} ({prev_w}×{prev_h} @ {prev_spp} spp)", flush=True)
        scene = mi.load_dict(scene_dict)
        render_np = np.array(mi.render(scene))
        beauty_np = _suppress_mc_sparkles(np.clip(render_np[:, :, :3], 0.0, 1.0))
        if render_np.shape[2] >= 5:
            mesh_alpha = _object_mask_from_depth(render_np[:, :, 4])
        else:
            mesh_alpha = (
                np.clip(render_np[:, :, 3], 0.0, 1.0)
                if render_np.shape[2] >= 4
                else np.ones(render_np.shape[:2], np.float32)
            )
        bg_rgb = np.array([0.82, 0.82, 0.82], dtype=np.float32)
        comp_np = beauty_np * mesh_alpha[..., None] + bg_rgb * (1.0 - mesh_alpha[..., None])
        comp_uint8 = (comp_np * 255).astype(np.uint8)
        bgr_uint8 = cv2.cvtColor(comp_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_png, bgr_uint8)
        print(f"     ✓ {out_png}", flush=True)

    print(f"\n✓ Done. Open {FRONT_PREVIEW_DIR} to review each mesh.")

