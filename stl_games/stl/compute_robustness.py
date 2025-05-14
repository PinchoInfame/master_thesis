import numpy as np

class ComputeRobustness:
    """
    Class to compute the robustness of a given trajectory to non-linear STL specs.
    """
    def __init__(self):
        pass
    
    def min_distance_to_obstacles(self, trajectories: np.ndarray, obstacles: list[tuple[float, float, float]], number_of_agents: int) -> float:
        """
        Compute the minimum distance from the robot trajectories to the obstacles.

        :param trajectories: State trajectory of the robots (array of shape (nx*number_of_robots, number_of_steps)).
        :param obstacles: List of obstacles in the format [(x_centre, y_centre, radius), ...].
        :param number_of_robots: Number of robots.

        :return: Minimum distance to the obstacles.
        """
        min_distance = float('inf')

        for i in range(number_of_agents):
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
    
    def min_distance_agents(self, trajectories: np.ndarray, number_of_agents: int) -> float:
        """
        Compute the minimum distance between the agents.

        :param trajectories: State trajectory of the agents (array of shape (nx*number_of_agents, number_of_steps)).
        :param number_of_agents: Number of agents.

        :return: Minimum distance between the robots.
        """
        min_distance = float('inf')

        for i in range(number_of_agents):
            traj_i = trajectories[i*4:i*4+2, :].T
            for j in range(i+1, number_of_agents):
                traj_j = trajectories[j*4:j*4+2, :].T
                positions_i = np.array([[p[0], p[1]] for p in traj_i])
                positions_j = np.array([[p[0], p[1]] for p in traj_j])
                distances = np.linalg.norm(positions_i - positions_j, axis=1)
                min_distance = min(min_distance, distances.min())
        return min_distance
