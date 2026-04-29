import bpy
import os
import sys
import argparse
import random
import time

def get_args():
    # Since it's run via `blender -b -P script.py -- [args]`, we need to parse sys.argv after `--`
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    
    parser = argparse.ArgumentParser(description="Generate draped cloth meshes.")
    parser.add_argument("--variations", type=int, default=1, help="Number of variations to generate")
    parser.add_argument("--input_dir", type=str, default="cloth_meshes", help="Directory with base collision meshes")
    parser.add_argument("--output_dir", type=str, default="output_meshes", help="Directory to save generated .obj files")
    parser.add_argument("--subdivisions", type=int, default=50, help="Subdivisions for cloth plane")
    parser.add_argument("--target_frame", type=int, default=100, help="Frame to freeze the simulation at")
    return parser.parse_args(argv)

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_simulation(input_dir, base_mesh_file, output_dir, subdivisions, target_frame, seed):
    random.seed(seed)
    
    base_mesh_path = os.path.abspath(os.path.join(input_dir, base_mesh_file))
    
    # 2. Import base mesh
    try:
        bpy.ops.wm.obj_import(filepath=base_mesh_path)
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=base_mesh_path)
    
    if not bpy.context.selected_objects:
        print("Error: Could not import object or no objects selected after import.")
        return False
        
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = "CollisionMesh"
    
    # Ensure it's active
    bpy.context.view_layer.objects.active = base_obj
    
    # Scale adjustment if needed (assuming base meshes are normalized, but just in case)
    # We leave scaling as is, assuming cloth_meshes/ are properly scaled.
    
    # Apply Collision modifier
    bpy.ops.object.modifier_add(type='COLLISION')
    
    # 3. Create Cloth Plane
    # Spawn it above the collision mesh. The exact height might depend on the base mesh scale.
    # Let's get the max Z of the bounding box of the base mesh.
    bbox_corners = [base_obj.matrix_world @ mathutils.Vector(corner) for corner in base_obj.bound_box]
    max_z = max([v.z for v in bbox_corners])
    
    spawn_z = max_z + 2.0 # 2.0 units above the highest point to prevent instant intersection
    # Ensure spawn_z is at least 2.5
    spawn_z = max(spawn_z, 2.5)
    
    # Also calculate a reasonable size for the cloth plane based on X/Y bounding box
    max_x = max([v.x for v in bbox_corners])
    min_x = min([v.x for v in bbox_corners])
    max_y = max([v.y for v in bbox_corners])
    min_y = min([v.y for v in bbox_corners])
    
    # Randomize plane size multiplier for variety
    size_mult = random.uniform(1.2, 1.8)
    plane_size = max(max_x - min_x, max_y - min_y) * size_mult
    plane_size = max(plane_size, 2.0) # At least size 2.0
    
    bpy.ops.mesh.primitive_plane_add(size=plane_size, enter_editmode=False, align='WORLD', location=(0, 0, spawn_z))
    cloth_obj = bpy.context.active_object
    cloth_obj.name = "DrapedCloth"
    
    # Subdivide the plane
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=subdivisions)
    bpy.ops.object.editmode_toggle()
    
    # Randomize location/rotation significantly for variation
    cloth_obj.location.x += random.uniform(-0.2 * plane_size, 0.2 * plane_size)
    cloth_obj.location.y += random.uniform(-0.2 * plane_size, 0.2 * plane_size)
    cloth_obj.rotation_euler.z += random.uniform(0, 3.14159 * 2) # Full 360 rotation possible
    
    # Apply Cloth Modifier
    cloth_mod = cloth_obj.modifiers.new(name="Cloth", type='CLOTH')
    
    # Randomize Cloth Physics Properties for different fold styles
    cloth_settings = cloth_mod.settings
    cloth_settings.mass = random.uniform(0.1, 0.5)
    cloth_settings.tension_stiffness = random.uniform(5, 25)
    cloth_settings.compression_stiffness = random.uniform(5, 25)
    cloth_settings.shear_stiffness = random.uniform(1, 10)
    cloth_settings.bending_stiffness = random.uniform(0.1, 2.0)
    
    collision_settings = cloth_mod.collision_settings
    collision_settings.use_collision = True
    collision_settings.use_self_collision = True
    
    # Smooth shading
    bpy.ops.object.shade_smooth()
    
    # Add Subdivision Surface Modifier
    subsurf_mod = cloth_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf_mod.levels = 1
    subsurf_mod.render_levels = 1
    
    # 4. Run Simulation
    print(f"Baking simulation up to frame {target_frame}...")
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = target_frame
    
    # Set point cache frame end
    cloth_mod.point_cache.frame_end = target_frame
    
    # Bake the physics
    bpy.ops.ptcache.bake_all(bake=True)
    
    # Set frame to the end to freeze
    bpy.context.scene.frame_set(target_frame)
    bpy.context.view_layer.update()
        
    # 5. Apply Modifiers to Freeze Geometry
    bpy.context.view_layer.objects.active = cloth_obj
    for mod in cloth_obj.modifiers:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        
    # 6. Delete Base Mesh
    bpy.data.objects.remove(base_obj, do_unlink=True)
    
    # 7. Export .obj
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = int(time.time() * 1000)
    export_filename = f"draped_{timestamp}_{base_mesh_file}"
    export_path = os.path.abspath(os.path.join(output_dir, export_filename))
    
    # Select only the cloth
    bpy.ops.object.select_all(action='DESELECT')
    cloth_obj.select_set(True)
    
    try:
        bpy.ops.wm.obj_export(
            filepath=export_path,
            export_selected_objects=True,
            export_materials=False
        )
    except AttributeError:
        bpy.ops.export_scene.obj(
            filepath=export_path,
            use_selection=True,
            use_materials=False
        )
        
    print(f"Successfully generated: {export_path}")
    return True

def main():
    args = get_args()
    
    import mathutils # Need to make sure mathutils is available inside the function or globally
    global mathutils
    import mathutils
    
    # Select ONE random base mesh for all variations
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
        
    obj_files = [f for f in os.listdir(input_dir) if f.endswith(".obj")]
    if not obj_files:
        print(f"Error: No .obj files found in '{input_dir}'.")
        return
        
    base_mesh_file = random.choice(obj_files)
    print(f"Selected base mesh for this run: {base_mesh_file}")

    print(f"Starting generation of {args.variations} variations...")
    for i in range(args.variations):
        print(f"\n--- Generating variation {i+1}/{args.variations} ---")
        clear_scene()
        success = setup_simulation(
            args.input_dir, 
            base_mesh_file,
            args.output_dir, 
            args.subdivisions, 
            args.target_frame,
            seed=time.time() + i
        )
        if not success:
            print("Failed to generate variation.")
            break
            
if __name__ == "__main__":
    main()
