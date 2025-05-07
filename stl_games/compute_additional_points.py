from stl_games.augment_trajectory_points import AugmentTrajectoryPoints
from stl_games.augment_control_inputs import AugmentControlInput

class ComputeAdditionalPoints:
    def __init__(self):
        self.augmented_trajectory = None
        self.augmented_input = None
        self.additional_points = None
    '''
    def __call__(self, horizon_mpc, number_original_points, trajectory_to_be_modified, input_to_be_modified):
        self.additional_points = (horizon_mpc+1-number_original_points)/(number_original_points-1)
        if self.additional_points % 1:
            raise ValueError("Number of points in the trajectory is not compatible with the horizon")
        else:
            self.additional_points = int(self.additional_points)
        augment_trajectory_points = AugmentTrajectoryPoints()
        augment_trajectory_points(trajectory_to_be_modified, self.additional_points)
        self.augmented_trajectory = augment_trajectory_points.augmented_trajectory
        augment_control_input = AugmentControlInput()
        augment_control_input(input_to_be_modified, self.additional_points)
        self.augmented_input = augment_control_input.augmented_input
    '''
    def __call__(self, trajectory_to_be_modified, input_to_be_modified, additional_points):
        augment_trajectory_points = AugmentTrajectoryPoints()
        augment_trajectory_points(trajectory_to_be_modified, additional_points)
        self.augmented_trajectory = augment_trajectory_points.augmented_trajectory
        augment_control_input = AugmentControlInput()
        augment_control_input(input_to_be_modified, additional_points)
        self.augmented_input = augment_control_input.augmented_input