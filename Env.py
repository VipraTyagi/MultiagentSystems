import numpy as np
import noise  # Using the noise library which implements Perlin noise
import blueprints as blue
import matplotlib.pyplot as plt

# Function to generate the terrain with a larger scale
def island(X, Y, frequency=1.0, amplitude=1.0, gradient=True):
    height_map = np.zeros((Y, X))  # Create an empty height map
    
    center_x, center_y = X / 2, Y / 2  # Calculate center
    max_radius = np.sqrt(center_x**2 + center_y**2)  # Calculate max radius
    
    for y in range(Y):
        for x in range(X):
            nx, ny = x / X - 0.5, y / Y - 0.5  # Normalized coordinates
            noise_value = noise.pnoise2(nx * frequency, ny * frequency, octaves=1) * amplitude
            if gradient:
                dist_x, dist_y = x - center_x, y - center_y
                distance = np.sqrt(dist_x**2 + dist_y**2)
                radial_factor = (max_radius - distance) / max_radius
                noise_value *= radial_factor
            height_map[y, x] = noise_value  # Set height map value
    
    return height_map

# Function to position the apple on terrain with a larger vertical offset
def position_object_on_terrain(obj, terrain_heightmap, x, z, vertical_offset=2.0):
    if 0 <= x < terrain_heightmap.shape[1] and 0 <= z < terrain_heightmap.shape[0]:
        terrain_height = terrain_heightmap[int(z), int(x)]  # Get terrain height at (x, z)
        obj.position = [x, terrain_height + vertical_offset, z]  # Position the object above terrain
        print(f"Apple positioned at: {obj.position}")  # Debugging info
    else:
        raise ValueError("Coordinates are out of terrain bounds.")

# Main function to create the environment
def main():
    X, Y = 200, 200  # Increased the size of the terrain for a larger area
    terrain_heightmap = island(X, Y, frequency=0.1, amplitude=1.0, gradient=True)
    
    # Create the world
    world = blue.World()
    
    # Add the terrain with a larger size
    hfield = blue.geoms.HField(terrain=terrain_heightmap, color="white")
    world.attach(hfield)
    
    # Increase the size of the terrain by scaling the hfield
    hfield.scale = [4.0, 1.0, 4.0]  # Scale the terrain in X and Z direction to make it larger
    
    # Add the apple and position it on the terrain
    apple = blue.geoms.Mesh(file="Chestnut.obj", color="orange")
    apple.pos = [0.0, 0.0, 6.5]  # Scale the apple to a visible size
    print(f"Apple scale set to: {apple.scale}")
    apple=apple.scaled(0.1)
    
    acorn = blue.geoms.Mesh(file="Acorn.obj", color="yellow")
    acorn.pos = [-3.0, -3.0, 3.2]  # Scale the apple to a visible size
    print(f"Acorn scale set to: {acorn.scale}")
    acorn=acorn.scaled(0.1)
    
    stone = blue.geoms.Mesh(file="Stone.obj", color="teal")
    stone.pos = [4.0, 4.0, 10]  # Scale the apple to a visible size
    stone=stone.scaled(0.1)
    
    # Position the apple above the terrain with a proper vertical offset
    position_object_on_terrain(apple, terrain_heightmap, x=100, z=100, vertical_offset=5.0)
    position_object_on_terrain(acorn, terrain_heightmap, x=100, z=100, vertical_offset=5.0)
    position_object_on_terrain(acorn, terrain_heightmap, x=100, z=100, vertical_offset=5.0) # Increased vertical offset
    world.attach(apple)
    world.attach(acorn)
    world.attach(stone)

    # View the environment
    world.view()

if __name__ == "__main__":
    main()
