import numpy as np
class AugmentTrajectoryPoints:
    def __init__(self):
        self.augmented_trajectory = []
    def __call__(self, trajectory, num_points):
        for i in range(np.size(trajectory, axis=1)-1):
            p1 = trajectory[:,i]
            p2 = trajectory[:,i+1]
            self.augmented_trajectory.append(p1)
            for j in range(1, num_points+1):
                alpha = j / (num_points + 1)
                interpolated_point = (1-alpha) * p1 + alpha * p2
                self.augmented_trajectory.append(interpolated_point)
        self.augmented_trajectory.append(trajectory[:,-1])
        self.augmented_trajectory = np.array(self.augmented_trajectory).T