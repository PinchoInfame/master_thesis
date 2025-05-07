import numpy as np
from stl_games.augment_control_inputs import AugmentControlInput

class ComputeTrajectories:
    def __init__(self):
        self.y = None
        self.augmented_y = None
        self.augmented_u = None
        self.x = None

    def compute_y(self, x, u, number_of_robots):
        y = np.concatenate([x[0:4,:],u[0:2,:]])
        for i in range(1,number_of_robots):
            y = np.concatenate([y, x[(i*4):(i*4+4),:], u[(i*2):(i*2+2),:]])
        #if np.size(u, axis=1) == np.size(x, axis=1):
        #    y = y[:,:-1]
        self.y = y

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