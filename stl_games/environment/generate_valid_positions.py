import numpy as np
import random

class GenerateValidPositions_4States:
    """Class to generate valid start and goal positions for agents with 4 states in a 2D continuous space."""
    def __init__(self):
        self.start_positions = []
        self.goal_positions = []
    def generate_valid_start_positions(self, grid_size: int, num_positions: int, obstacles: list[tuple[float, float, float]], min_dist: float=0, min_dist_obs: float=0) -> np.ndarray:
        """
        Generate valid start positions ensuring they are not inside obstacles and are not too close to each other.
        
            :param grid_size (int): Size of the grid (assumed square).
            :param num_positions (int): Number of start positions to generate.
            :param obstacles (list): List of obstacles in the format [(x_centre, y_centre, radius), ...].
            :param min_dist (float): Minimum distance between start positions.
            :param min_dist_obs (float): Minimum distance from obstacles.
            :return start_positions (np.ndarray): Array of valid start positions in the format [(x, y, vx, vy), ...].
        """
        while len(self.start_positions) < num_positions:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            x = float(x)
            y = float(y)
            if self.is_valid_position((x, y), self.start_positions, obstacles, min_dist, min_dist_obs):
                self.start_positions.append((x, y, 0.0, 0.0))
        self.start_positions = np.array(self.start_positions)
        return self.start_positions
    
    def generate_valid_goal_positions(self, grid_size: int, num_positions: int, number_of_robots: int, number_of_goals: list[int], obstacles: list[tuple[float, float, float]], min_dist: float=0, min_dist_obs: float=0):
        """
        Generate goal positions ensuring they are not inside obstacles and are not too close to each other.
            :param grid_size (int): Size of the grid (assumed square).
            :param num_positions (int): Total number of goal positions to generate.
            :param number_of_robots (int): Number of robots.
            :param number_of_goals (list): List of number of goals for each robot.
            :param obstacles (list): List of obstacles in the format [(x_centre, y_centre, radius), ...].
            :param min_dist (float): Minimum distance between goal positions.
            :param min_dist_obs (float): Minimum distance from obstacles.
            :return goal_positions (np.ndarray): Array of valid goal positions in the format [(x, y, vx, vy), ...].
            :return goal_list (list): List of goal positions for each agent in the format [[[xᵢⱼ, yᵢⱼ, vxᵢⱼ, vyᵢⱼ] for j in goals_i] for i in robots].
        """
        while len(self.goal_positions) < num_positions:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            x = float(x)
            y = float(y)
            if self.is_valid_position((x, y), self.goal_positions, obstacles, min_dist, min_dist_obs):
                self.goal_positions.append((x, y, 0.0, 0.0))
        xG = []
        goal_positions = self.goal_positions
        for i in range(number_of_robots):
            xGi = []
            for j in range(number_of_goals[i]):
                xGi.append(goal_positions[j])
            goal_positions = goal_positions[number_of_goals[i]:]
            xG.append(xGi)
        self.goal_list = xG
        self.goal_positions = np.array(self.goal_positions)
        return self.goal_positions, self.goal_list
    def is_valid_position(self, pos, existing_positions, obstacles, min_dist, min_dist_obs):
        """Check if a position is valid (not in obstacle, not too close to existing positions)."""
        x, y = pos
        for (xobs, yobs, radius) in obstacles:
            if np.linalg.norm(np.array([x, y]) - np.array([xobs, yobs])) < radius + min_dist_obs:
                return False # Position is inside an obstacle
        for px, py, vx, vy in existing_positions:
            if np.linalg.norm(np.array([x, y]) - np.array([px, py])) < min_dist:
                return False  # Too close to another position
        return True
        