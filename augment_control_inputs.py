import numpy as np
class AugmentControlInput:
    def __init__(self):
        self.augmented_input = None

    def __call__(self, control_input, num_points):
        #divided_vector = control_input / (num_points+1)
        self.augmented_input = np.repeat(control_input[:,:-1], num_points+1, axis=1)
        last_col = control_input[:, -1][:, np.newaxis]  # Reshape to keep dimensions
        self.augmented_input = np.concatenate((self.augmented_input, last_col), axis=1)
        self.augmented_input = np.array(self.augmented_input)
