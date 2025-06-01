import numpy as np
import cvxpy as cp

class MPCPlayer1:
    def __init__(self, horizon, dt, N, d_safe=0.5):
        self.horizon = horizon
        self.dt = dt
        self.d_safe = d_safe
        self.N = N
        self.create_optimization_variables()

    def create_optimization_variables(self):
        self.parameter1_coll = cp.Parameter((2))
        self.parameter2_coll = cp.Parameter()
        self.slack_coll = cp.Variable(nonneg=True)
        self.x1 = cp.Variable((4, self.N+1))  # States: x, y, vx, vy
        self.u1 = cp.Variable((2, self.N))    # Controls: ax, ay
        self.x0_1 = cp.Parameter(4)
        self.x2_fixed = cp.Parameter((4, self.N+1))  # Fixed trajectory of player 2
        self.x1_ref = cp.Parameter((4, self.N+1))

    def define_dynamics(self, state: np.ndarray, input: np.ndarray) -> np.ndarray:
        '''
        Compute the next state of the multi-agent system using discretized double integrator dynamics.

        :param state: Current state vector of all agents, flattened as a 1D array
        :param input: Control input vector for all agents, flattened as a 1D array
        :return: Flattened next state vector after applying control input.
        '''
        dt = self.dt
        Ad = np.array([[1, 0, dt, 0],  
                       [0, 1, 0, dt],  
                       [0, 0, 1, 0],  
                       [0, 0, 0, 1]])
        Bd = np.array([[(dt**2)/2, 0],  
                       [0, (dt**2)/2],  
                       [dt, 0],  
                       [0, dt]])  
        return Ad @ state + Bd @ input
        

    def build(self):
          # Horizon steps
        self.Q = np.diag([1, 1, 0.1, 0.1])  # Penalize position strongly, velocity lightly
        self.R = 0.1 * np.eye(2)
        self.slack_weight_coll = 100.0

        cost = 0
        constraints = [self.x1[:, 0] == self.x0_1]

        cbf_constraint_obs = self.parameter1_coll @ (self.u1[:, 0]) + self.parameter2_coll
        constraints.append(cbf_constraint_obs >= -self.slack_coll)
        for k in range(self.horizon-1):
            # Dynamics constraint
            x_next = self.define_dynamics(self.x1[:, k], self.u1[:, k])
            constraints.append(self.x1[:, k + 1] == x_next)
            
            # Cost: tracking + control effort
            cost += cp.quad_form(self.x1[:, k] - self.x1_ref[:, k], self.Q)
            cost += cp.quad_form(self.u1[:, k], self.R)            

        # Terminal cost
        cost += cp.quad_form(self.x1[:, self.horizon] - self.x1_ref[:, self.horizon], self.Q)
        cost += self.slack_weight_coll * self.slack_coll

        # Solve the QP
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x1_ref, x2_fixed, x0_1):
        self.x0_1.value = x0_1
        self.x1_ref.value = x1_ref
        self.x2_fixed.value = x2_fixed

        alpha_coll = 0.5
        epsilon = 1e-6
        x0_2 = x2_fixed[:, 0]
        dist = x0_1[0:2] - x0_2[0:2]
        norm_dist = np.linalg.norm(dist) + epsilon
        db_dx = dist / norm_dist
        b = norm_dist - self.d_safe
        Ap = np.array([ [1, 0, self.dt, 0],
                        [0, 1, 0, self.dt]])
        Bp = np.array([ [(self.dt**2)/2, 0],
                        [0, (self.dt**2)/2]])
        self.parameter1_coll.value = db_dx @ (Bp / self.dt)
        self.parameter2_coll.value = db_dx @ ((Ap @ x0_1 - x0_1[0:2]) / self.dt) + alpha_coll * b

        try:
            self.problem.solve(solver=cp.GUROBI)
            if self.problem.status in ["optimal", "optimal_inaccurate"]:
                u_opt = self.u1.value[:,0]
                return u_opt, self.x1.value, self.u1.value
            else:
                print("Problem is not optimal. Status:", self.problem.status)
                return None, None, None
        except cp.SolverError as e:
            print('Solver failed:', e)
            return None, None, None
            
    def solve_receding_horizon(self, x1_ref, x2_fixed, x0_1, max_iterations=10):
        x1_opt = [x0_1]
        u1_opt = []
        x_current = x0_1
        for t in range(max_iterations):
            u_opt, x_prev, u_prev = self.solve(x1_ref, x2_fixed, x_current)
            if u_opt is None:
                print("Solver failed")
                exit()
            x1_ref = np.hstack([x1_ref[:, 1:], x1_ref[:, -1:]])
            x2_fixed = np.hstack([x2_fixed[:, 1:], x2_fixed[:, -1:]])
            x_current = self.define_dynamics(x_current, u_opt)
            x1_opt.append(x_current)
            u1_opt.append(u_opt)
            # Simulate the system dynamics for the next time step
        x1_opt = np.array(x1_opt).T
        u1_opt = np.array(u1_opt).T
        return x1_opt, u1_opt
