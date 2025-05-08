import numpy as np

class ComputeRobustness:
    def __init__(self):
        pass
    
    def min_distance_to_obstacles(self, trajectories, obstacles, number_of_robots):
        min_distance = float('inf')

        for i in range(number_of_robots):
            traj = trajectories[i*4:i*4+2, :].T  # Extract x, y for each robot
            positions = np.array([[p[0], p[1]] for p in traj])  # Extract (x, y)
            for obs in obstacles:
                x_obs = obs[0]
                y_obs = obs[1]
                radius = obs[2]
                obs_center = np.array([x_obs, y_obs])
                distances = np.linalg.norm(positions - obs_center, axis=1) - radius
                min_distance = min(min_distance, distances.min())
        return min_distance