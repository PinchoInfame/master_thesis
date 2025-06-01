import itertools
import numpy as np

def detect_collision(x, number_of_robots, safe_dist):
    collision_indices = []
    collision_detected = False
    for robot1_index, robot2_index in itertools.combinations(range(number_of_robots), 2):
        robot1_x = x[robot1_index * 4]
        robot1_y = x[robot1_index * 4 + 1]
        robot2_x = x[robot2_index * 4]
        robot2_y = x[robot2_index * 4 + 1]
        distances = np.sqrt((robot1_x - robot2_x)**2 + (robot1_y - robot2_y)**2)
        collision_check = distances < safe_dist
        if np.any(collision_check):
            collision_detected = True
            collision_indices.append((robot1_index, robot2_index))
    return collision_detected, collision_indices
            