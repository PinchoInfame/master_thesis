import numpy as np
import cvxpy as cp

class MPCPlayer1:
    def __init__(self, horizon, dt, N, d_safe=0.5, goal_list=None, goal_size=4.0, step_to_reach_goal=None, osbtacles=None):
        self.horizon = horizon
        self.dt = dt
        self.d_safe = d_safe
        self.N = N
        self.goal_list = goal_list if goal_list is not None else []
        self.goal_area_size = goal_size
        self.step_to_reach_goal = step_to_reach_goal if step_to_reach_goal is not None else horizon
        self.obs_list = osbtacles if osbtacles is not None else []
        self.number_of_obs = len(self.obs_list)
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
        self.gamma_goal1 = cp.Parameter((len(self.goal_list), self.horizon + 1)) 
        self.gamma_goal2 = cp.Parameter((len(self.goal_list), self.horizon + 1))
        self.gamma_goal3 = cp.Parameter((len(self.goal_list), self.horizon + 1))
        self.gamma_goal4 = cp.Parameter((len(self.goal_list), self.horizon + 1))
        self.slack_cbf1 = cp.Variable((len(self.goal_list), self.horizon), nonneg=True)
        self.slack_cbf2 = cp.Variable((len(self.goal_list), self.horizon), nonneg=True) 
        self.slack_cbf3 = cp.Variable((len(self.goal_list), self.horizon), nonneg=True) 
        self.slack_cbf4 = cp.Variable((len(self.goal_list), self.horizon), nonneg=True) 
        self.slack_terminal1 = cp.Variable((len(self.goal_list)), nonneg=True) 
        self.slack_terminal2 = cp.Variable((len(self.goal_list)), nonneg=True)
        self.slack_terminal3 = cp.Variable((len(self.goal_list)), nonneg=True) 
        self.slack_terminal4 = cp.Variable((len(self.goal_list)), nonneg=True)
        self.parameter1_obs = cp.Parameter((2*self.number_of_obs))
        self.parameter2_obs = cp.Parameter((self.number_of_obs))
        self.slack_obs = cp.Variable((len(self.obs_list)), nonneg=True) 

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
        

    def build(self, x0_1):
          # Horizon steps
        self.Q = np.diag([1, 1, 0.1, 0.1])  # Penalize position strongly, velocity lightly
        self.R = 0.1 * np.eye(2)
        self.slack_weight_coll = 100.0
        self.slack_weight_goal = 200.0
        self.slack_weight_obs = 300.0
        self.slack_cost_goal = 0.0
        self.slack_cost_obs = 0.0
        self.alpha_obs = 0.8
        self.alpha_goal = 1.0
        self.alpha_coll = 0.5

        constraints = []

        gamma_goal1 = np.zeros((len(self.goal_list), self.horizon+1)) 
        gamma_goal2 = np.zeros((len(self.goal_list), self.horizon+1)) 
        gamma_goal3 = np.zeros((len(self.goal_list), self.horizon+1)) 
        gamma_goal4 = np.zeros((len(self.goal_list), self.horizon+1)) 

        time_interval = (0,self.step_to_reach_goal)
        switch = self.step_to_reach_goal+1
        for j in range(len(self.goal_list)):
            h0_goal1 = self.goal_area_size - (x0_1[0]-self.goal_list[j][0])
            h0_goal2 = self.goal_area_size - (x0_1[1]-self.goal_list[j][1])
            h0_goal3 = self.goal_area_size - (-x0_1[0]+self.goal_list[j][0])
            h0_goal4 = self.goal_area_size - (-x0_1[1]+self.goal_list[j][1])
            gamma0_goal1, tau_goal1 = self.define_gamma_params(time_interval, 'eventually', h0_goal1)
            gamma0_goal2, tau_goal2 = self.define_gamma_params(time_interval, 'eventually', h0_goal2)
            gamma0_goal3, tau_goal3 = self.define_gamma_params(time_interval, 'eventually', h0_goal3)
            gamma0_goal4, tau_goal4 = self.define_gamma_params(time_interval, 'eventually', h0_goal4)
            gamma_goal1[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1, tau_goal1, switch)
            gamma_goal2[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2, tau_goal2, switch)
            gamma_goal3[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal3, tau_goal3, switch)
            gamma_goal4[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal4, tau_goal4, switch)
        self.gamma_goal1.value = gamma_goal1
        self.gamma_goal2.value = gamma_goal2
        self.gamma_goal3.value = gamma_goal3
        self.gamma_goal4.value = gamma_goal4

        M = 1e6  # Large constant
        z = cp.Variable((len(self.goal_list)), boolean=True)
        if len(self.goal_list)>1:
            constraints.append(cp.sum(z) >= 1)

        
        # Obstacle avoidance just at the first step of each iteration
        for j, obs in enumerate(self.obs_list):
            cbf_constraint = self.parameter1_obs[(j*2):(j*2)+2] @ (self.u1[0:2, 0]) + self.parameter2_obs[j]
            constraints.append(cbf_constraint >= -self.slack_obs[j])

        cost = 0
        constraints = [self.x1[:, 0] == self.x0_1]

        cbf_constraint_obs = self.parameter1_coll @ (self.u1[:, 0]) + self.parameter2_coll
        constraints.append(cbf_constraint_obs >= -self.slack_coll)
        for k in range(self.horizon-1):
            # Dynamics constraint
            x_next = self.define_dynamics(self.x1[:, k], self.u1[:, k])
            constraints.append(self.x1[:, k + 1] == x_next)

            # Goal reaching with cbf: b(x, t) = h(x) + gamma(t) >= 0
            for j in range(len(self.goal_list)):
                dist_to_goal1 = self.x1[0,k] - self.goal_list[j][0]
                dist_to_goal2 = self.x1[1,k] - self.goal_list[j][1]
                dist_to_goal3 = -self.x1[0,k] + self.goal_list[j][0]
                dist_to_goal4 = -self.x1[1,k] + self.goal_list[j][1]
                h1 = self.goal_area_size - dist_to_goal1
                h2 = self.goal_area_size - dist_to_goal2
                h3 = self.goal_area_size - dist_to_goal3
                h4 = self.goal_area_size - dist_to_goal4
                b1 =  h1 + self.gamma_goal1[j][k]
                b2 =  h2 + self.gamma_goal2[j][k]
                b3 =  h3 + self.gamma_goal3[j][k]
                b4 =  h4 + self.gamma_goal4[j][k]
                dist_to_goal1_next = self.x1[0,k+1] - self.goal_list[j][0]
                dist_to_goal2_next = self.x1[1,k+1] - self.goal_list[j][1]
                dist_to_goal3_next = -self.x1[0,k+1] + self.goal_list[j][0]
                dist_to_goal4_next = -self.x1[1,k+1] + self.goal_list[j][1]
                h1_next = self.goal_area_size - dist_to_goal1_next
                h2_next = self.goal_area_size - dist_to_goal2_next
                h3_next = self.goal_area_size - dist_to_goal3_next
                h4_next = self.goal_area_size - dist_to_goal4_next
                b1_next =  h1_next + self.gamma_goal1[j][k+1]
                b2_next =  h2_next + self.gamma_goal2[j][k+1]
                b3_next =  h3_next + self.gamma_goal3[j][k+1]
                b4_next =  h4_next + self.gamma_goal4[j][k+1]

                if len(self.goal_list)>1:
                    constraints.append(h1 + self.gamma_goal1[j][k] >= -self.slack_cbf1[j][k] - M * (1 - z[j]))
                    constraints.append(h2 + self.gamma_goal2[j][k] >= -self.slack_cbf2[j][k] - M * (1 - z[j]))
                    constraints.append(h3 + self.gamma_goal3[j][k] >= -self.slack_cbf3[j][k] - M * (1 - z[j]))
                    constraints.append(h4 + self.gamma_goal4[j][k] >= -self.slack_cbf4[j][k] - M * (1 - z[j]))
                    constraints.append(b1_next-b1 >= self.alpha_obs*b1 - self.slack_cbf1[j][k] - M * (1 - z[j]))
                    constraints.append(b2_next-b2 >= self.alpha_obs*b2 - self.slack_cbf2[j][k] - M * (1 - z[j]))
                    constraints.append(b3_next-b3 >= self.alpha_obs*b3 - self.slack_cbf3[j][k] - M * (1 - z[j]))
                    constraints.append(b4_next-b4 >= self.alpha_obs*b4 - self.slack_cbf4[j][k] - M * (1 - z[j]))
                else:
                    constraints.append(h1 + self.gamma_goal1[j][k] >= -self.slack_cbf1[j][k])
                    constraints.append(h2 + self.gamma_goal2[j][k] >= -self.slack_cbf2[j][k])
                    constraints.append(h3 + self.gamma_goal3[j][k] >= -self.slack_cbf3[j][k])
                    constraints.append(h4 + self.gamma_goal4[j][k] >= -self.slack_cbf4[j][k])
                    constraints.append(b1_next-b1 >= self.alpha_obs*b1 - self.slack_cbf1[j][k] - M * (1 - z[j]))
                    constraints.append(b2_next-b2 >= self.alpha_obs*b2 - self.slack_cbf2[j][k] - M * (1 - z[j]))
                    constraints.append(b3_next-b3 >= self.alpha_obs*b3 - self.slack_cbf3[j][k] - M * (1 - z[j]))
                    constraints.append(b4_next-b4 >= self.alpha_obs*b4 - self.slack_cbf4[j][k] - M * (1 - z[j]))
            
            # Cost: tracking + control effort
            cost += cp.quad_form(self.x1[:, k] - self.x1_ref[:, k], self.Q)
            cost += cp.quad_form(self.u1[:, k], self.R)
        # Define terminal CBF constraints with slack
        for j in range(len(self.goal_list)):
            final_dist_to_goal1 = self.x1[0, self.horizon] - self.goal_list[j][0]
            final_dist_to_goal2 = self.x1[1, self.horizon] - self.goal_list[j][1]
            final_dist_to_goal3 = -self.x1[0, self.horizon] + self.goal_list[j][0]
            final_dist_to_goal4 = -self.x1[1, self.horizon] + self.goal_list[j][1]
            h1_terminal = self.goal_area_size - final_dist_to_goal1
            h2_terminal = self.goal_area_size - final_dist_to_goal2
            h3_terminal = self.goal_area_size - final_dist_to_goal3
            h4_terminal = self.goal_area_size - final_dist_to_goal4
            if len(self.goal_list)>1:
                constraints.append(h1_terminal + self.gamma_goal1[j][self.horizon] >= -self.slack_terminal1[j] - M * (1 - z[j]))
                constraints.append(h2_terminal + self.gamma_goal2[j][self.horizon] >= -self.slack_terminal2[j] - M * (1 - z[j]))
                constraints.append(h3_terminal + self.gamma_goal3[j][self.horizon] >= -self.slack_terminal3[j] - M * (1 - z[j]))
                constraints.append(h4_terminal + self.gamma_goal4[j][self.horizon] >= -self.slack_terminal4[j] - M * (1 - z[j]))
            else:
                constraints.append(h1_terminal + self.gamma_goal1[j][self.horizon] >= -self.slack_terminal1[j])
                constraints.append(h2_terminal + self.gamma_goal2[j][self.horizon] >= -self.slack_terminal2[j])
                constraints.append(h3_terminal + self.gamma_goal3[j][self.horizon] >= -self.slack_terminal3[j])
                constraints.append(h4_terminal + self.gamma_goal4[j][self.horizon] >= -self.slack_terminal4[j]) 
        # Terminal cost
        self.slack_cost_goal += self.slack_weight_goal*(cp.sum(cp.sum(self.slack_cbf1)) + cp.sum(cp.sum(self.slack_cbf2)) + cp.sum(cp.sum(self.slack_cbf3)) + cp.sum(cp.sum(self.slack_cbf4))
                                                       + cp.sum(self.slack_terminal1) + cp.sum(self.slack_terminal2) + cp.sum(self.slack_terminal3) + cp.sum(self.slack_terminal4))
        self.slack_cost_obs += self.slack_weight_obs*cp.sum(self.slack_obs)
        cost += cp.quad_form(self.x1[:, self.horizon] - self.x1_ref[:, self.horizon], self.Q)
        cost += self.slack_cost_goal + self.slack_cost_obs
        cost += self.slack_weight_coll * self.slack_coll

        # Solve the QP
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, x1_ref, x2_fixed, x0_1, current_iteration=0):
        self.x0_1.value = x0_1
        self.x1_ref.value = x1_ref
        self.x2_fixed.value = x2_fixed

        # update gamma(t) --> time shaping function
        gamma_goal1 = np.zeros((len(self.goal_list), self.horizon+1))
        gamma_goal2 = np.zeros((len(self.goal_list), self.horizon+1))
        gamma_goal3 = np.zeros((len(self.goal_list), self.horizon+1))
        gamma_goal4 = np.zeros((len(self.goal_list), self.horizon+1))

        time_interval = (0,self.step_to_reach_goal-current_iteration)
        switch = time_interval[1]+1
        for j in range(len(self.goal_list)):
            h0_goal1 = self.goal_area_size - (x0_1[0]-self.goal_list[j][0])
            h0_goal2 = self.goal_area_size - (x0_1[1]-self.goal_list[j][1])
            h0_goal3 = self.goal_area_size - (-x0_1[0]+self.goal_list[j][0])
            h0_goal4 = self.goal_area_size - (-x0_1[1]+self.goal_list[j][1])
            if (h0_goal1 >= self.goal_area_size)&(h0_goal2 >= self.goal_area_size)&(h0_goal3 >= self.goal_area_size)&(h0_goal4 >= self.goal_area_size):
                switch = 0
            gamma0_goal1, tau_goal1 = self.define_gamma_params(time_interval, 'eventually', h0_goal1)
            gamma0_goal2, tau_goal2 = self.define_gamma_params(time_interval, 'eventually', h0_goal2)
            gamma0_goal3, tau_goal3 = self.define_gamma_params(time_interval, 'eventually', h0_goal3)
            gamma0_goal4, tau_goal4 = self.define_gamma_params(time_interval, 'eventually', h0_goal4)
            gamma_goal1[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1, tau_goal1, switch)
            gamma_goal2[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2, tau_goal2, switch)
            gamma_goal3[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal3, tau_goal3, switch)
            gamma_goal4[j] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal4, tau_goal4, switch)
        self.gamma_goal1.value = gamma_goal1
        self.gamma_goal2.value = gamma_goal2
        self.gamma_goal3.value = gamma_goal3
        self.gamma_goal4.value = gamma_goal4

        # Compute parameters to enforce obstacle avoidance
        parameter1_obs = np.zeros((2*self.number_of_obs))
        parameter2_obs = np.zeros((self.number_of_obs))
        for j, obs in enumerate(self.obs_list):
            obs_center = np.array([obs[0], obs[1]])
            obs_radius = obs[2]
            x0_i = x0_1[0:4]
            Ap = np.array([[1, 0, self.dt, 0],
                        [0, 1, 0, self.dt]])
            Bp = np.array([[(self.dt**2)/2, 0],
                        [0, (self.dt**2)/2]])
            epsilon = 1e-4
            dist = x0_1[0:2] - obs_center
            norm_dist = np.linalg.norm(dist)+epsilon
            db_dx = dist / norm_dist
            b = norm_dist - obs_radius # b(x0, t=0)
            parameter1_obs[j*2:j*2+2] = db_dx @ (Bp/self.dt)
            parameter2_obs[j] = db_dx @ ((Ap @ x0_i - x0_i[0:2])/self.dt) + self.alpha_obs * b
        
        self.parameter1_obs.value = parameter1_obs
        self.parameter2_obs.value = parameter2_obs

        
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
        self.parameter2_coll.value = db_dx @ ((Ap @ x0_1 - x0_1[0:2]) / self.dt) + self.alpha_coll * b

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
            
    def solve_receding_horizon(self, x1_ref, x2_fixed, x0_1, max_iterations=10, current_iteration=0):
        x1_opt = [x0_1]
        u1_opt = []
        x_current = x0_1
        for t in range(max_iterations):
            u_opt, x_prev, u_prev = self.solve(x1_ref, x2_fixed, x_current, current_iteration)
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
    

    def define_gamma_params(self, time_interval,  operation_type, h0):
        a = time_interval[0]
        b = time_interval[1]
        if h0<=0:
            gamma_0 = 1.0*(-h0)
        elif h0>0:
            gamma_0 = 1.0*h0
        if operation_type=='always':
            tau = a
        elif operation_type=='eventually':
            tau = b
        else:
            print('Wrong operation type: choose between always and eventually')
        tau = max(tau, 1e-3)
        return gamma_0, tau
        
    def compute_gamma(self, T, gamma0, tau, switch):
        time_values = np.arange(0, T)
        gamma_values = gamma0 - ((gamma0 + self.goal_area_size) / tau) * time_values
        gamma_values = np.maximum(gamma_values, -self.goal_area_size)
        if T < self.horizon+1:
            gamma_values = np.concatenate((gamma_values, self.goal_area_size*np.ones(self.horizon+1-T)))
        if switch is not None:
            gamma_values[switch:]=1e3
        return gamma_values[:self.horizon+1]
