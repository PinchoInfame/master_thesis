import numpy as np

import numpy as np

def generate_obstacles(num_obs, grid_size, max_radius, min_radius=1.0, max_attempts=1000):
    """
    Generate non-overlapping circular obstacles in a 2D grid.

    :param num_obs: Number of obstacles to generate.
    :param grid_size: Size of the grid (obstacles will be in [0, grid_size] x [0, grid_size]).
    :param max_radius: Maximum radius of obstacles.
    :param min_radius: Minimum radius of obstacles.
    :param max_attempts: Max attempts before giving up for each obstacle.
    :return: NumPy array of shape (num_obs, 3) with (x, y, radius).
    """
    obstacles = []
    attempts = 0

    while len(obstacles) < num_obs and attempts < max_attempts:
        # Random center and radius
        x = np.random.uniform(0, grid_size)
        y = np.random.uniform(0, grid_size)
        r = np.random.uniform(min_radius, max_radius)

        new_obs = np.array([x, y, r])

        # Check overlap
        overlap = False
        for existing in obstacles:
            dx = x - existing[0]
            dy = y - existing[1]
            dist = np.hypot(dx, dy)
            if dist < r + existing[2]:  # sum of radii
                overlap = True
                break

        if not overlap:
            obstacles.append(new_obs)
        attempts += 1

    if len(obstacles) < num_obs:
        raise ValueError(f"Could only place {len(obstacles)} non-overlapping obstacles after {max_attempts} attempts.")

    return np.array(obstacles)
