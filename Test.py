import numpy as np
import matplotlib.pyplot as plt
import blueprints as blue
import numpy as np
from math import trunc, ceil
from itertools import product



def smoothstep(x):
	y = 3 * x**2 - 2 * x**3
	y[x <= 0] = 0.
	y[x >= 1] = 1.
	return y



def padding(x, axis):
	axis = tuple(slice(None if i == -1 else 1, -1 if i == -1 else None) for i in axis)
	return x[*axis,...]



def resize(img, shape):
	x_shape = img.shape
	y_shape = shape
	for x, y in zip(x_shape, y_shape):
		X = np.arange(x)[None,...]
		Y = np.arange(y)[...,None]
		mini = np.minimum((1 + X) * y / x, (1 + Y))
		maxi = np.maximum(     X  * y / x,      Y )
		M = np.maximum(0, mini - maxi)
		img = np.einsum('ij,j...->i...', M, img)
		img = np.moveaxis(img, 0, -1)
	return img



def perlin(resolution, frequency, torus=False):
	ALPHA = 'abcdefghijklmnopqrstuvwx'
	min_width = int(ceil(min(resolution) / frequency))
	grid_size = tuple(int(ceil(x / min_width)) for x in resolution)
	cell_size = tuple(int(ceil(x / y)) for x, y in zip(resolution, grid_size))
	ndim      = len(grid_size)
	cell      = np.stack(np.meshgrid(*(np.linspace(-1, 1, n) for n in cell_size)), axis=-1)
	corners   = list(map(np.array, product(*((-1, 1) for _ in range(ndim)))))
	offsets   = np.stack(list(map(lambda x: cell - x, corners)), axis=-1)
	distances = np.sqrt(np.sum(offsets**2, axis=-2)) / 2
	factors   = 1 - smoothstep(distances)
	grads     = np.random.uniform(low=-1, high=1, size=(*(x + 1 for x in grid_size), ndim))
	grads     = grads / (1e-8 + np.sqrt(np.sum(grads**2, axis=-1)[...,None]))
	if torus:
		for i, _ in enumerate(grid_size):
			grads[*(slice(None) for _ in range(i)), -1, ...] = grads[*(slice(None) for _ in range(i)),0,...]
	rule     = f'{ALPHA[:ndim].upper()}y,{ALPHA[:ndim]}yz->{ALPHA[:ndim].upper()}{ALPHA[:ndim]}z'
	products = np.einsum(rule, grads, offsets)
	correlations = np.stack([padding(products[...,i], corner) for i, corner in enumerate(corners)], axis=-1)
	rule     = f'{ALPHA[:ndim].upper()}{ALPHA[:ndim]}y,{ALPHA[:ndim]}y->{ALPHA[:ndim].upper()}{ALPHA[:ndim]}'
	heights  = np.einsum(rule, correlations, factors)
	heights  = np.moveaxis(heights, 0, 1)
	for _ in range(ndim):
		heights = np.concatenate(np.rollaxis(heights, axis=0), axis=ndim-1)
	heights  = np.moveaxis(heights, 0, 1)
	heights  = resize(heights, resolution)
	return heights

def island(X, Y, frequency=1.0, amplitude=1.0, gradient=True, base=1.8, number=10, offset=1, water=6):
    height_map = np.zeros((Y, X))
    center_x, center_y = X / 2, Y / 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    
    # Layered Perlin noise
    for i in range(offset, number + offset):
        perlin_noise = perlin((Y, X), base ** i, torus=True)
        for y in range(Y):
            for x in range(X):
                nx, ny = x / X - 0.5, y / Y - 0.5
                noise_value = perlin_noise[y, x] * amplitude
                if gradient:
                    dist_x, dist_y = x - center_x, y - center_y
                    distance = np.sqrt(dist_x**2 + dist_y**2)
                    radial_factor = (max_radius - distance) / max_radius
                    noise_value *= radial_factor
                height_map[y, x] += noise_value / (base ** i)
    
    # Apply water level transformation
    height_map = water ** height_map
    return height_map

def generate_terrain(resolution, base, number, offset, water, use_island=False):
    if use_island:
        return island(resolution[1], resolution[0], frequency=0.1, amplitude=1.0, gradient=True)
    else:
        np.random.seed(42)
        img = np.zeros(resolution)
        for i in range(offset, number + offset):
            N = perlin(resolution, base ** i, torus=True)
            img += N / (base ** i)
        img = water ** img
        return img

def visualize_with_blueprints(height_map):
    world = blue.World()
    hfield = blue.geoms.HField(terrain=height_map, color="orange")
    world.attach(hfield)
    world.view()

def save_terrain_image(height_map, filename='test.png'):
    plt.imsave(filename, height_map, cmap='terrain')

def create_terrain_mesh(width=1, height=1, subdivision_width=40, subdivision_height=40):
    num_vertices = (subdivision_width + 1) * (subdivision_height + 1)
    vertices = np.zeros((num_vertices, 3))
    st = np.zeros((num_vertices, 2))
    inv_subdivision_width = 1.0 / subdivision_width
    inv_subdivision_height = 1.0 / subdivision_height

    for j in range(subdivision_height + 1):
        for i in range(subdivision_width + 1):
            index = j * (subdivision_width + 1) + i
            vertices[index] = [width * (i * inv_subdivision_width - 0.5), 0, height * (j * inv_subdivision_height - 0.5)]
            st[index] = [i * inv_subdivision_width, j * inv_subdivision_height]

    # Generate Perlin noise map
    noise_map = perlin((subdivision_height + 1, subdivision_width + 1), frequency=1.0)

    # Displace vertices using noise map
    for i in range(num_vertices):
        x = min(int(st[i, 0] * subdivision_width), subdivision_width)
        y = min(int(st[i, 1] * subdivision_height), subdivision_height)
        vertices[i, 1] = 2 * noise_map[y, x] - 1  # Displace along y-axis
    return vertices


def visualize_terrain_mesh(vertices, subdivision_width, subdivision_height):
    # Convert vertices to height map for visualization
    height_map = vertices[:, 1].reshape((subdivision_height + 1, subdivision_width + 1))
    visualize_with_blueprints(height_map)

if __name__ == "__main__":
    
    
    resolution = (216, 512)
    base = 1.8
    number = 5
    offset = 1
    water = 10
    np.random.seed(42)

    # Generate terrain mesh using Perlin noise
    vertices = create_terrain_mesh(subdivision_width=40, subdivision_height=40)

    # Visualize the terrain mesh
    visualize_terrain_mesh(vertices, subdivision_width=40, subdivision_height=40)

    # Optionally, save the terrain as an image
    height_map = vertices[:, 1].reshape((41, 41))  # Adjust shape based on subdivisions
    save_terrain_image(height_map)