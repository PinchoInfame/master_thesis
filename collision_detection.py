import itertools
import numpy as np
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
            