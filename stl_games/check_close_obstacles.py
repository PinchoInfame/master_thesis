import numpy as np
class CheckCloseObstacles:
    def __init__(self, number_of_robots, obstacles, threshold):
        self.number_of_robots = number_of_robots
        self.obstacles = obstacles
        self.threshold = threshold
        self.close_obstacles = None
    def __call__(self, trajectory):
        close_obstacles_list = set()
        for i in range(self.number_of_robots):
            x, y = trajectory[4 * i], trajectory[4 * i + 1]  # Extract position of robot i

            for j, (x_min, x_max, y_min, y_max) in enumerate(self.obstacles):
                distance = self.min_distance_to_rectangle(x, y, x_min, x_max, y_min, y_max)
                if np.any(distance < self.threshold):
                    close_obstacles_list.add(j)  # Store obstacle index

        self.close_obstacles = [self.obstacles[i] for i in close_obstacles_list]
    
    def min_distance_to_rectangle(self, x, y, x_min, x_max, y_min, y_max):
        x = np.atleast_1d(x)  # Ensure x is an array (even if scalar)
        y = np.atleast_1d(y)  # Ensure y is an array (even if scalar)
        # Check if the point is inside the rectangle
        inside = np.logical_and.reduce((x_min <= x, x <= x_max, y_min <= y, y <= y_max))
        if np.any(inside):
            return 0.0
        # Compute minimum distance to the edges
        dx = np.where((x_min <= x) & (x <= x_max), 0, np.minimum(abs(x - x_min), abs(x - x_max)))
        dy = np.where((y_min <= y) & (y <= y_max), 0, np.minimum(abs(y - y_min), abs(y - y_max)))
        return np.sqrt(dx**2 + dy**2)