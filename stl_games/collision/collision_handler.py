import itertools
import numpy as np

class CollisionHandler:
    def __init__(self):
        self.trajectories_to_be_modified=[]
        self.input_to_be_modified = []
        self.collision_times = []
        self.collision_detected = False
        self.collision_indices = []
    def __call__(self, x, u, number_of_robots, safe_dist, delta_t):
        self.trajectories_to_be_modified=[]
        self.input_to_be_modified = []
        self.collision_times = []
        self.collision_detected = False
        self.collision_indices = []
        collision_detection = CollisionDetection()
        collision_detection(x, number_of_robots, safe_dist)
        if collision_detection.collision_detected:
            print("Collision detected")
            self.collision_detected = True
            collisions = collision_detection.collision_times
            for (ind1, ind2, t) in collisions:
                if t>=2:
                    self.trajectories_to_be_modified.append(np.concatenate((x[(ind1*4):(ind1*4+4), (t-delta_t):], x[(ind2*4):(ind2*4+4), (t-delta_t):]), axis=0))
                    self.input_to_be_modified.append(np.concatenate((u[(ind1*2):(ind1*2+2), t-delta_t:], u[(ind2*2):(ind2*2+2), t-delta_t:]), axis=0))
                    self.collision_times.append(t)
                    self.collision_indices.append((ind1, ind2))
            #self.trajectories_to_be_modified = np.array(self.trajectories_to_be_modified)
            #self.input_to_be_modified = np.array(self.input_to_be_modified)
            #self.collision_times = np.array(self.collision_times)
        else:
            print("No collision detected")
            self.collision_detected = False
            self.trajectories_to_be_modified = []
            self.input_to_be_modified = []

class CollisionDetection:
    def __init__(self):
        self.collision_points_x_plot = []
        self.collision_points_y_plot = []
        self.collision_times = []
        self.collision_detected = False
    def __call__(self, x, number_of_robots, safe_dist):
        robot_positions = []
        for i in range(number_of_robots):
            robot_x=(x[i*4])
            robot_y=(x[(i*4)+1])
            robot_positions.append((robot_x, robot_y))
        for robot1_index, robot2_index in itertools.combinations(range(number_of_robots), 2):
            robot1_x, robot1_y = robot_positions[robot1_index]
            robot2_x, robot2_y = robot_positions[robot2_index]
            #collision_check = np.sqrt((robot1_x - robot2_x)**2 + (robot1_y - robot2_y)**2) < safe_dist
            collision_check = (np.abs(robot1_x-robot2_x)<safe_dist) & (np.abs(robot1_y-robot2_y)<safe_dist)
            if (np.any(collision_check)):
                self.collision_detected = True
                self.collision_times.append((robot1_index, robot2_index, np.where(collision_check==True)[0][0]))
            self.collision_points_x_plot.append((robot1_x[collision_check] + robot2_x[collision_check])/2)
            self.collision_points_y_plot.append((robot1_y[collision_check] + robot2_y[collision_check])/2)
        self.collision_times = np.array(self.collision_times)

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
            