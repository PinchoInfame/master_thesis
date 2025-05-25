import numpy as np
def continuous_time_dynamics(t, x, u, number_of_robots):
    """
    Continuous time dynamics for the system.
    """
    Ac = np.array([[0, 0, 1, 0],  
                [0, 0, 0, 1],  
                [0, 0, 0, 0],  
                [0, 0, 0, 0]])
    Bc = np.array([[0, 0],  
                    [0, 0],  
                    [1, 0],  
                    [0, 1]])
    Cc = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    Dc = np.array([[0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]])
    A_full = np.kron(np.eye(number_of_robots), Ac)
    B_full = np.kron(np.eye(number_of_robots), Bc)
    C_full = np.kron(np.eye(number_of_robots), Cc)
    D_full = np.kron(np.eye(number_of_robots), Dc)
    return A_full @ x + B_full @ u