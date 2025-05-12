import numpy as np
import igl
import meshplot
import copy 
import polyscope as ps
import trimesh
import argparse
import process_floors

ps.init()

def register_mesh(v, f, n, c, title="Mesh"):
    # Register the mesh with Polyscope
    mesh = ps.register_surface_mesh(title, v, f)
    mesh.add_color_quantity("colors", c, defined_on='vertices', enabled=True)
    return mesh

def load_mesh_with_attributes(filename):
    mesh = trimesh.load(filename)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    normals = np.array(mesh.vertex_normals)
    colors = None
    if hasattr(mesh.visual, 'vertex_colors'):
        # Convert RGBA to RGB and normalize to 0-1 range
        colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0
    return vertices, faces, normals, colors

def get_max_connected_component(surface_f): 
    # gets the largest connected component of the mesh, which removes artifacts
    components = igl.facet_components(surface_f)
    max_freq_component = np.argmax(np.bincount(components))
    max_component = surface_f[components == max_freq_component]
    return max_component

if __name__ == "__main__":
    # GET MESH
    try:
        parser = argparse.ArgumentParser(description="Process a mesh file.")
        parser.add_argument("--f", type=str, help="Path to the mesh file")
        parser.add_argument("--save", nargs='?', const="processed_mesh.obj", type=str, help="Path to save the processed mesh. If no path is provided, defaults to 'processed_mesh.obj'")
        args = parser.parse_args()
        if not args.f:
            raise ValueError("Mesh file path (--f) is required.")
        filename = args.f
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    print("Loading mesh from:", filename)
    v, f, n, c = load_mesh_with_attributes(filename)
    register_mesh(v, f, n, c)
    
    # CLEAR ARTIFACTS
    print("Removing floating artifacts...")
    f = get_max_connected_component(f)
    register_mesh(v, f, n, c, "Removed Floating Artifacts from Raw Mesh")
    
    # REMOVE DIPPING FLOORS 
    print("Removing dipping floors...")
    v, f = process_floors.process_floors(v, f)
    register_mesh(v, f, n, c, "Removed Dipping Floors after Removing Floating Artifacts")
    
    # HOLE FILLING
    print("Filling holes...")
    f = process_floors.fill_mesh_holes(v, f)
    register_mesh(v, f, n, c, "Filled Holes After Removing Dipping Floors")
    
    # SAVE MESH
    if args.save:
        igl.write_triangle_mesh("mesh_output.ply", v, f)
        print(f"Mesh saved to mesh_output.ply")
    
    ps.show()