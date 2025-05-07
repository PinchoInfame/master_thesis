import numpy as np

class Computex0:
    def __init__(self):
        pass
    def __call__(self, start_positions, number_of_robots):
        self.x0 = []
        for i in range(number_of_robots):
            px0 = start_positions[i, 0]
            py0 = start_positions[i, 1]
            vx0 = 0
            vy0 = 0
            self.x0.append(np.array([px0, py0, vx0, vy0]))
        self.x0 = np.array(self.x0)
        self.x0 = self.x0.flatten()