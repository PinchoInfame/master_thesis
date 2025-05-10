import numpy as np

class ComputeTrajectories:
    """
    Class to compute the trajectory of the robots based on the system dynamics and control inputs.
    """
    def __init__(self):
        self.y = None
        self.augmented_y = None
        self.augmented_u = None
        self.x = None

    def compute_y_concatenate(self, x: np.ndarray, u: np.ndarray, number_of_robots: int):
        """
        Compute the output as a concatenation of the state trajectory and control inputs.

        :param x: State trajectory of the robots (array of shape (nx*number_of_robots, number_of_steps)).
        :param u: Control input trajectory (array of shape (nu*number_of_robots, number_of_steps)).
        :param number_of_robots: Number of robots.

        :return: y: Output trajectory (array of shape ((nu+nx)*number_of_robots, number_of_steps)).
        """
        if np.size(u, axis=1) == np.size(x, axis=1):
            pass
        elif np.size(u, axis=1) == np.size(x, axis=1)-1:
            u = np.hstack((u, np.zeros((u.shape[0],1))))
        else:
            raise ValueError("The size of the control input does not match the size of the state trajectory")
        y = np.concatenate([x[0:4,:],u[0:2,:]])
        for i in range(1,number_of_robots):
            y = np.concatenate([y, x[(i*4):(i*4+4),:], u[(i*2):(i*2+2),:]])
        #if np.size(u, axis=1) == np.size(x, axis=1):
        #    y = y[:,:-1]
        self.y = y
        return self.y

    def compute_augmented_y(self, x, x0, u, combined_system, additional_points, dt, number_of_robots):
        # collision times contains indices of the robots which collide and time at which they collide
        A = combined_system.A
        B = combined_system.B
        if dt != 1:
            for i in range(number_of_robots):
                A[i*4,(i*4)+2]=dt
                A[(i*4)+1,(i*4)+3]=dt
                B[i*4,i*2]= (dt**2)/2
                B[(i*4)+1,(i*2)+1]= (dt**2)/2
                B[(i*4)+2,i*2]=dt
                B[(i*4)+3,(i*2)+1]=dt
        if np.size(u, axis=1) == np.size(x, axis=1):
            u = u[:,:-1]
        augment_u = AugmentControlInput()
        augment_u(u, additional_points)
        augmented_u = augment_u.augmented_input
        num_steps = np.size(augmented_u, axis=1)
        augmented_x = np.zeros((number_of_robots*4, num_steps + 1))
        augmented_x[:, 0] = x0
        for t in range(num_steps):
            u_t = augmented_u[:, t]
            augmented_x[:, t+1] = A @ augmented_x[:,t] + B @ u_t
        augmented_u = np.hstack((augmented_u, np.zeros((number_of_robots*2,1))))
        augmented_y = np.concatenate([augmented_x[0:4,:],augmented_u[0:2,:]])
        for i in range(1,number_of_robots):
            augmented_y = np.concatenate([augmented_y, augmented_x[(i*4):(i*4+4),:], augmented_u[(i*2):(i*2+2),:]])
        self.augmented_y = augmented_y
        self.augmented_u = augmented_u
    def compute_x(self, system_dynamics, number_of_robots, dt, u, x0):
        # create a function to substitute the trajectory with the one computed by the mpc when a collision occurs
        A = np.zeros((number_of_robots*4, number_of_robots*4))
        B = np.zeros((number_of_robots*4, number_of_robots*2))
        for i in range(number_of_robots):
            A[i*4, i*4]=1
            A[i*4+1, i*4+1]=1
            A[i*4+2, i*4+2]=1
            A[i*4+3, i*4+3]=1
            A[i*4,(i*4)+2]=dt
            A[(i*4)+1,(i*4)+3]=dt
            B[i*4,i*2]= (dt**2)/2
            B[(i*4)+1,(i*2)+1]= (dt**2)/2
            B[(i*4)+2,i*2]=dt
            B[(i*4)+3,(i*2)+1]=dt
        num_steps = np.size(u, axis=1)
        x = np.zeros((number_of_robots*4, num_steps + 1))
        x[:, 0] = x0
        for t in range(num_steps):
            u_t = u[:, t]
            x[:, t+1] = A @ x[:,t] + B @ u_t
        self.x = x

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

class AugmentControlInput:
    def __init__(self):
        self.augmented_input = None

    def __call__(self, control_input, num_points):
        #divided_vector = control_input / (num_points+1)
        self.augmented_input = np.repeat(control_input[:,:-1], num_points+1, axis=1)
        last_col = control_input[:, -1][:, np.newaxis]  # Reshape to keep dimensions
        self.augmented_input = np.concatenate((self.augmented_input, last_col), axis=1)
        self.augmented_input = np.array(self.augmented_input)

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