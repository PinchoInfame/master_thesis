import numpy as np
import random

class GenerateValidPositions:
    def __init__(self):
        self.start_positions = []
        self.goal_positions = []
    def generate_valid_start_positions(self, grid_size, num_positions, obstacles, min_dist=0, min_dist_obs=0):
        """Generate valid start or goal positions ensuring constraints."""
        while len(self.start_positions) < num_positions:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            x = float(x)
            y = float(y)
            if self.is_valid_position((x, y), self.start_positions, obstacles, min_dist, min_dist_obs):
                self.start_positions.append((x, y, 0.0, 0.0))
        self.start_positions = np.array(self.start_positions)
    def generate_valid_goal_positions(self, grid_size, num_positions, number_of_robots, number_of_goals, obstacles, min_dist=0, min_dist_obs=0):
        """Generate goal positions ensuring they are not inside obstacles and are not too close to each other."""
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
        self.xG = xG
        self.goal_positions = np.array(self.goal_positions)
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
        