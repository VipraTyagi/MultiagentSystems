import numpy as np
import noise  # Using the noise library which implements Perlin noise
import blueprints as blue
import matplotlib.pyplot as plt



def island(X, Y, frequency=2.0, amplitude=2.0, gradient=True):
    # Create an empty numpy array for the height map
    height_map = np.zeros((Y, X))
    
    # Center of the map for gradient calculation
    center_x, center_y = X / 2, Y / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    # Generate height map
    for y in range(Y):
        for x in range(X):
            nx, ny = x / X - 0.5, y / Y - 0.5
            # Generate noise value at this point
            noise_value = noise.pnoise2(nx * frequency, ny * frequency, octaves=1) * amplitude
            # Apply radial gradient if enabled
            if gradient:
                dist_x, dist_y = x - center_x, y - center_y
                distance = np.sqrt(dist_x**2 + dist_y**2)
                radial_factor = (max_radius - distance) / max_radius
                noise_value *= radial_factor
            height_map[y, x] = noise_value
    
    return height_map

# def plot_height_map(height_map):
#     plt.figure(figsize=(10, 7))
#     plt.imshow(height_map, cmap='terrain')
#     plt.colorbar()
#     plt.show()

# Example usage

def main():
    X, Y = 400, 400  # Dimensions of the height map
    height_map = island(X, Y, frequency=0.2, amplitude=2.0, gradient=True) 
    
    # Initialize the world
    world = blue.World()
    
    # Create the height field (terrain)
    hfield = blue.geoms.HField(terrain=height_map, color="white")
    world.attach(hfield)
    
    # Integrate the apple (Chestnut.obj) file
    apple = blue.geoms.Mesh(file="Acorn.obj", color="orange")
    
  # Adjust the apple's scale
    apple.position = [0.5, 0.2, 0.5]  # Middle of the terrain, slightly above

    
    # Map apple's position to the terrain
    apple_x = X // 2  # Center X coordinate
    apple_y = Y // 2  # Center Y coordinate
    terrain_height = height_map[apple_y, apple_x]  # Terrain height at the apple's position
    
    # Set apple position above the terrain
    apple.position = [apple_x / X, terrain_height + 0.3, apple_y / Y]
     # Debugging: Print the apple's position
    print(f"Apple position: {apple.position}")
    
    # Add slight rotation for realism (if supported)
    if hasattr(apple, 'rotation'):
        apple.rotation = (0, 45, 0)  # Rotate 45 degrees on the Y-axis
    
    # Attach apple to the world
    world.attach(apple)
    
    # Visualize the world
    world.view()

if __name__ == "__main__":
    main()