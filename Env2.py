import numpy as np
import noise  # Perlin noise library
import blueprints as blue
import matplotlib.pyplot as plt
import blueprints.assets as assets  # For MeshAsset

def island(X, Y, frequency=1.0, amplitude=1.0, gradient=True):
    """
    Generates a 2D heightmap using Perlin noise with a radial gradient.
    """
    height_map = np.zeros((Y, X))
    center_x, center_y = X / 2, Y / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(Y):
        for x in range(X):
            nx, ny = x / X - 0.5, y / Y - 0.5
            noise_value = noise.pnoise2(nx * frequency, ny * frequency, octaves=1) * amplitude
            if gradient:
                dist_x, dist_y = x - center_x, y - center_y
                distance = np.sqrt(dist_x**2 + dist_y**2)
                radial_factor = (max_radius - distance) / max_radius
                noise_value *= radial_factor
            height_map[y, x] = noise_value
    return height_map

def position_object_on_terrain(obj, terrain_heightmap, x, z, vertical_offset=2.0):
    """
    Positions an object on the terrain by sampling the heightmap.
    """
    if 0 <= x < terrain_heightmap.shape[1] and 0 <= z < terrain_heightmap.shape[0]:
        terrain_height = terrain_heightmap[int(z), int(x)]
        # Convert height to float explicitly (if not already)
        obj.position = [x, float(terrain_height + vertical_offset), z]
        print("Object positioned at:", obj.position)
    else:
        raise ValueError("Coordinates are out of terrain bounds.")

def heightmap_to_mesh(heightmap, scale=(1.0, 1.0, 1.0)):
    """
    Converts the 2D heightmap into vertices and faces.
    Each grid cell becomes two triangles.
    """
    rows, cols = heightmap.shape
    sx, sy, sz = scale

    # Create vertices (each grid point becomes a vertex)
    vertices = []
    for z in range(rows):
        for x in range(cols):
            vx = x * sx
            vy = heightmap[z, x] * sy
            vz = z * sz
            vertices.append([vx, vy, vz])
    vertices = np.array(vertices, dtype=np.float32)

    # Create faces (two triangles per grid cell)
    faces = []
    for z in range(rows - 1):
        for x in range(cols - 1):
            i0 = z * cols + x
            i1 = (z + 1) * cols + x
            i2 = z * cols + (x + 1)
            i3 = (z + 1) * cols + (x + 1)
            faces.append([i0, i1, i2])
            faces.append([i2, i1, i3])
    faces = np.array(faces, dtype=np.int32)
    
    return vertices, faces

def write_obj(filename, vertices, faces):
    """
    Writes vertices and faces to an OBJ file.
    Ensures that face indices are written as integers (OBJ files use 1-indexing).
    """
    with open(filename, 'w') as f:
        # Write vertices (they can be floats)
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        # Write faces: cast each index to int and add 1 (because OBJ is 1-indexed)
        for face in faces:
            # Ensure indices are properly rounded to integer values
            face_int = [int(round(i)) for i in face]
            f.write("f {:d} {:d} {:d}\n".format(face_int[0] + 1, face_int[1] + 1, face_int[2] + 1))

def main():
    # Generate a heightmap for the terrain.
    X, Y = 200, 200
    terrain_heightmap = island(X, Y, frequency=0.1, amplitude=1.0, gradient=True)
    
    # Create the Blueprints world.
    world = blue.World()

    # Convert the heightmap into a mesh (with scaling in X/Z and Y).
    mesh_verts, mesh_faces = heightmap_to_mesh(terrain_heightmap, scale=(4.0, 1.0, 4.0))
    
    # Convert the numpy arrays to lists.
    verts_list = mesh_verts.tolist()
    faces_list = mesh_faces.tolist()
    
    # Write the mesh data to an OBJ file. This file will be used by the caching system.
    dummy_filename = "generated_mesh.obj"
    write_obj(dummy_filename, verts_list, faces_list)
    
    # Create a MeshAsset with the required keys.
    # Note: the key for vertices is intentionally misspelled as "vertcies" per Blueprints' API.
    mesh_asset = assets.MeshAsset(
        vertcies=verts_list,
        faces=faces_list,
        file=dummy_filename
    )
    
    # Create a terrain mesh using the MeshAsset.
    terrain_mesh = blue.geoms.Mesh(
        asset=mesh_asset,
        color="white"
    )
    
    # Attach the terrain mesh to the world.
    world.attach(terrain_mesh)
    
    # Load, scale, and position additional objects.
    apple = blue.geoms.Mesh(file="Chestnut.obj", color="orange").scaled(0.1)
    acorn = blue.geoms.Mesh(file="Acorn.obj", color="yellow").scaled(0.1)
    stone = blue.geoms.Mesh(file="Stone.obj", color="teal").scaled(0.1)
    
    position_object_on_terrain(apple, terrain_heightmap, x=100, z=100, vertical_offset=5.0)
    position_object_on_terrain(acorn, terrain_heightmap, x=100, z=100, vertical_offset=5.0)
    position_object_on_terrain(stone, terrain_heightmap, x=90, z=110, vertical_offset=5.0)
    
    world.attach(apple)
    world.attach(acorn)
    world.attach(stone)
    
    # Launch the view.
    world.view()

if __name__ == "__main__":
    main()
