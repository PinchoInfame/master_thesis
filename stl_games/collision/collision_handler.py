import itertools
import numpy as np

class CollisionHandler:
    def __init__(self):
        pass
    def handle_collision(self, x, u, number_of_robots, safe_dist, delta_t=2):
        trajectories_to_be_modified=[]
        input_to_be_modified = []
        steps_of_collision = []
        collision_detected = False
        collision_indices = []
        collision_detected, collisions, _, _ = self.detect_collision(x, number_of_robots, safe_dist)
        if collision_detected:
            print("Collision detected")
            for (ind1, ind2, t) in collisions:
                if t>=delta_t:
                    trajectories_to_be_modified.append(np.concatenate((x[(ind1*4):(ind1*4+4), (t-delta_t):], x[(ind2*4):(ind2*4+4), (t-delta_t):]), axis=0))
                    input_to_be_modified.append(np.concatenate((u[(ind1*2):(ind1*2+2), t-delta_t:], u[(ind2*2):(ind2*2+2), t-delta_t:]), axis=0))
                    steps_of_collision.append(t)
                    collision_indices.append((ind1, ind2))
                else:
                    trajectories_to_be_modified.append(np.concatenate((x[(ind1*4):(ind1*4+4), :], x[(ind2*4):(ind2*4+4), :]), axis=0))
                    input_to_be_modified.append(np.concatenate((u[(ind1*2):(ind1*2+2), :], u[(ind2*2):(ind2*2+2), :]), axis=0))
                    steps_of_collision.append(t)
                    collision_indices.append((ind1, ind2))
        else:
            print("No collision detected")
            collision_detected = False
            trajectories_to_be_modified = []
            input_to_be_modified = []
        return collision_detected, trajectories_to_be_modified, input_to_be_modified, steps_of_collision, collision_indices
    
    def detect_collision(self, x, number_of_robots, safe_dist):
        robot_positions = []
        steps_of_collision = []
        collision_detected = False
        collision_points_x_plot = []
        collision_points_y_plot = []
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
                collision_detected = True
                steps_of_collision.append((robot1_index, robot2_index, np.where(collision_check==True)[0][0]))
            collision_points_x_plot.append((robot1_x[collision_check] + robot2_x[collision_check])/2)
            collision_points_y_plot.append((robot1_y[collision_check] + robot2_y[collision_check])/2)
        steps_of_collision = np.array(steps_of_collision)
        return collision_detected, steps_of_collision, collision_points_x_plot, collision_points_y_plot

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
            