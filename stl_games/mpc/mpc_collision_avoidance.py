import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import itertools

class MPCCollisionAvoidance:
    '''
    A controller class for computing optimal trajectories using Model Predictive Control (MPC)
    integrated with Control Barrier Functions (CBFs) and time-varying sets.
    '''
    def __init__(self, nx: int, nu: int, number_of_agents: int,  horizon: int, dt: float, u_min: float, u_max: float, goal_area_size: float, obs_list: list[tuple[float, float, float]], step_to_reach_goal: np.ndarray, goal_list: list[list[list[float]]], safe_dist: float=1.5):
        '''
        A controller class for computing optimal trajectories using Model Predictive Control (MPC)
        integrated with Control Barrier Functions (CBFs) and time-varying sets.

        :param nx: Number of states for each agent (e.g., x, y, vx, vy)
        :param nu: Number of inputs for each agent (e.g., ax, ay)
        :param number_of_agents: Number of agents
        :param horizon: Prediction horizon for the MPC
        :param dt: Time step for the MPC
        :param u_min: Minimum control input (e.g., acceleration)
        :param u_max: Maximum control input (e.g., acceleration)
        :param goal_area_size: Size of the square goal area
        :param obs_list: List of obstacles, each defined by a tuple (x, y, radius)
        :param step_to_reach_goal: Number of steps to reach the goal for each agent
        :param goal_list: List of goal positions for each agent in the format [[[xᵢⱼ, yᵢⱼ, vxᵢⱼ, vyᵢⱼ] for j in goals_i] for i in robots]
        :param safe_dist: Safe distance between agents
        '''
        self.solver_time = 0
        self.horizon = horizon
        self.nx = nx    # number of states for each agent
        self.nu = nu    # number of inputs for each agent
        self.number_of_agents = number_of_agents
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.R = 0.1*np.eye(number_of_agents*nu)
        self.Q = 0.5*np.diag([0, 0, 1, 1] * (number_of_agents))
        self.goal_area_size = goal_area_size
        self.obs_list = obs_list
        self.number_of_obs = len(obs_list)
        self.step_to_reach_goal = step_to_reach_goal    # ToDo: list of time before which each agent should reach the goal (in order to allow asyncronous goal reaching)
        self.goal_list = goal_list    # [[[xg1_1,yg1_1,0.0,0.0],[xg1_2,yg1_2,0.0,0.0],...],[[xg2_1,yg2_1,0.0,0.0],[xg2_2,yg2_2,0.0,0.0],...], ...]
        self.safe_dist = safe_dist
        self.create_optimization_variables()

    def create_optimization_variables(self):
        self.x = cp.Variable((self.nx*self.number_of_agents, self.horizon + 1))
        self.u = cp.Variable((self.nu*self.number_of_agents, self.horizon))
        self.x0 = cp.Parameter(self.nx*self.number_of_agents)
        self.xG = cp.Parameter(self.nx*self.number_of_agents)
        self.gamma_goal1 = {i: cp.Parameter((len(self.goal_list[i]), self.horizon + 1)) for i in range(self.number_of_agents)}
        self.gamma_goal2 = {i: cp.Parameter((len(self.goal_list[i]), self.horizon + 1)) for i in range(self.number_of_agents)}
        self.gamma_goal3 = {i: cp.Parameter((len(self.goal_list[i]), self.horizon + 1)) for i in range(self.number_of_agents)}
        self.gamma_goal4 = {i: cp.Parameter((len(self.goal_list[i]), self.horizon + 1)) for i in range(self.number_of_agents)}
        self.x_prev = cp.Parameter((self.nx*self.number_of_agents, self.horizon + 1))
        self.parameter1_obs = {i: cp.Parameter((2*self.number_of_obs)) for i in range(self.number_of_agents)}
        self.parameter2_obs = {i: cp.Parameter((self.number_of_obs)) for i in range(self.number_of_agents)}
        self.number_of_combinations = len(list(itertools.combinations(range(self.number_of_agents), 2)))
        self.parameter1_coll = {i: cp.Parameter((self.horizon+1, 2)) for i in range(self.number_of_combinations)}
        self.parameter2_coll = {i: cp.Parameter((self.horizon+1)) for i in range(self.number_of_combinations)}
        
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
        A_full = np.kron(np.eye(self.number_of_agents), Ad)
        B_full = np.kron(np.eye(self.number_of_agents), Bd)  
        self.A  = A_full
        self.B = B_full
        return self.A @ state + self.B @ input

    def build_problem(self, x0: np.ndarray):
        '''
        :param x0: Initial state of the system, flattened as a 1D array
        '''
        xG = np.zeros((4 * self.number_of_agents))  # Stores the closest goal for each agent
        closest_goals_indices = self.find_closest_goals(x0)
        for i in range(self.number_of_agents):
            xG[i*4:i*4+2] = self.goal_list[i][closest_goals_indices[i]][0:2]
        self.xG.value = xG
        self.control_cost = 0
        self.stage_cost = 0
        self.slack_cost_goal = 0
        self.slack_cost_obs = 0
        self.slack_cost_coll = 0
        self.terminal_cost = 0
        self.smoothing_cost = 0
        constraints = []
        slack_cbf1 = {i: cp.Variable((len(self.goal_list[i]), self.horizon), nonneg=True) for i in range(self.number_of_agents)}
        slack_cbf2 = {i: cp.Variable((len(self.goal_list[i]), self.horizon), nonneg=True) for i in range(self.number_of_agents)}
        slack_cbf3 = {i: cp.Variable((len(self.goal_list[i]), self.horizon), nonneg=True) for i in range(self.number_of_agents)}
        slack_cbf4 = {i: cp.Variable((len(self.goal_list[i]), self.horizon), nonneg=True) for i in range(self.number_of_agents)}
        slack_terminal1 = {i: cp.Variable((len(self.goal_list[i])), nonneg=True) for i in range(self.number_of_agents)}
        slack_terminal2 = {i: cp.Variable((len(self.goal_list[i])), nonneg=True) for i in range(self.number_of_agents)}
        slack_terminal3 = {i: cp.Variable((len(self.goal_list[i])), nonneg=True) for i in range(self.number_of_agents)}
        slack_terminal4 = {i: cp.Variable((len(self.goal_list[i])), nonneg=True) for i in range(self.number_of_agents)}

        slack_goal_weight = 10
        slack_obs_weight = 200
        slack_coll_weight = 200
        self.alpha_obs = 0.4
        self.alpha_coll = 0.1
        gamma_goal1 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}
        gamma_goal2 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}
        gamma_goal3 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}
        gamma_goal4 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}

        constraints.append(self.x[:, 0] == self.x0)

        for i in range(self.number_of_agents):
            time_interval = (0,self.step_to_reach_goal[i])
            switch = self.step_to_reach_goal[i]+1
            for j in range(len(self.goal_list[i])):
                #h0_goal = self.goal_area_size - np.linalg.norm(x0[i*4:i*4+2]-self.goal_list[i][j][0:2])
                h0_goal1 = self.goal_area_size - (x0[i*4]-self.goal_list[i][j][0])
                h0_goal2 = self.goal_area_size - (x0[i*4+1]-self.goal_list[i][j][1])
                h0_goal3 = self.goal_area_size - (-x0[i*4]+self.goal_list[i][j][0])
                h0_goal4 = self.goal_area_size - (-x0[i*4+1]+self.goal_list[i][j][1])
                gamma0_goal1, tau_goal1 = self.define_gamma_params(time_interval, 'eventually', h0_goal1)
                gamma0_goal2, tau_goal2 = self.define_gamma_params(time_interval, 'eventually', h0_goal2)
                gamma0_goal3, tau_goal3 = self.define_gamma_params(time_interval, 'eventually', h0_goal3)
                gamma0_goal4, tau_goal4 = self.define_gamma_params(time_interval, 'eventually', h0_goal4)
                gamma_goal1[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal1, tau_goal1, switch)
                gamma_goal2[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal2, tau_goal2, switch)
                gamma_goal3[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal3, tau_goal3, switch)
                gamma_goal4[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal4, tau_goal4, switch)
            self.gamma_goal1[i].value = gamma_goal1[i]
            self.gamma_goal2[i].value = gamma_goal2[i]
            self.gamma_goal3[i].value = gamma_goal3[i]
            self.gamma_goal4[i].value = gamma_goal4[i]

        
        '''
        z = {i: cp.Variable((self.horizon, len(self.goal_list[i])), boolean=True) for i in range(self.number_of_agents)}
        for i in range(self.number_of_agents):
            if len(self.goal_list[i])>1:
                for k in range(self.horizon-1):
                    constraints.append(cp.sum(z[i][k]) >= 1)
        '''
        
        z = {i: cp.Variable((len(self.goal_list[i])), boolean=True) for i in range(self.number_of_agents)}
        for i in range(self.number_of_agents):
            if len(self.goal_list[i])>1:
                constraints.append(cp.sum(z[i]) >= 1)
        
        M = 1e6  # Large constant

        slack_obs = {i: cp.Variable((len(self.obs_list)), nonneg=True) for i in range(self.number_of_agents)}
        # Obstacle avoidance just at the first step of each iteration
        for i in range(self.number_of_agents):
            for j, obs in enumerate(self.obs_list):
                cbf_constraint = self.parameter1_obs[i][(j*2):(j*2)+2] @ (self.u[i*2:i*2+2, 0]) + self.parameter2_obs[i][j]
                constraints.append(cbf_constraint >= -slack_obs[i][j])

        slack_coll = cp.Variable((self.number_of_combinations), nonneg=True)

        for k in range(self.horizon-1):
            x_next = self.define_dynamics(self.x[:, k], self.u[:, k])
            constraints.append(self.x[:, k + 1] == x_next)
            constraints.append(self.u[:, k] >= self.u_min)
            constraints.append(self.u[:, k] <= self.u_max)

            # Goal reaching with cbf: b(x, t) = h(x) + gamma(t) >= 0
            for i in range(self.number_of_agents):
                for j in range(len(self.goal_list[i])):
                    dist_to_goal1 = self.x[i*4,k] - self.goal_list[i][j][0]
                    dist_to_goal2 = self.x[i*4+1,k] - self.goal_list[i][j][1]
                    dist_to_goal3 = -self.x[i*4,k] + self.goal_list[i][j][0]
                    dist_to_goal4 = -self.x[i*4+1,k] + self.goal_list[i][j][1]
                    h1 = self.goal_area_size - dist_to_goal1
                    h2 = self.goal_area_size - dist_to_goal2
                    h3 = self.goal_area_size - dist_to_goal3
                    h4 = self.goal_area_size - dist_to_goal4
                    if len(self.goal_list[i])>1:
                        constraints.append(h1 + self.gamma_goal1[i][j][k] >= -slack_cbf1[i][j][k] - M * (1 - z[i][j]))
                        constraints.append(h2 + self.gamma_goal2[i][j][k] >= -slack_cbf2[i][j][k] - M * (1 - z[i][j]))
                        constraints.append(h3 + self.gamma_goal3[i][j][k] >= -slack_cbf3[i][j][k] - M * (1 - z[i][j]))
                        constraints.append(h4 + self.gamma_goal4[i][j][k] >= -slack_cbf4[i][j][k] - M * (1 - z[i][j]))
                    else:
                        constraints.append(h1 + self.gamma_goal1[i][j][k] >= -slack_cbf1[i][j][k])
                        constraints.append(h2 + self.gamma_goal2[i][j][k] >= -slack_cbf2[i][j][k])
                        constraints.append(h3 + self.gamma_goal3[i][j][k] >= -slack_cbf3[i][j][k])
                        constraints.append(h4 + self.gamma_goal4[i][j][k] >= -slack_cbf4[i][j][k])

            # Collision avoidance with cbf
            for idx, (i, j) in enumerate(itertools.combinations(range(self.number_of_agents), 2)):
                # Compute the distance between the two agents
                diff_pos = self.x[i*4:i*4+2, k] - self.x[j*4:j*4+2, k]
                d_hat = (self.parameter1_coll[idx][k,:]) @ (diff_pos) + self.parameter2_coll[idx][k]
                h_collision = d_hat - self.safe_dist + self.alpha_coll * (d_hat - self.safe_dist)
                constraints.append(h_collision >= -slack_coll[idx])

        # Define terminal CBF constraints with slack
        for i in range(self.number_of_agents):
            for j in range(len(self.goal_list[i])):
                final_dist_to_goal1 = self.x[i*4, self.horizon] - self.goal_list[i][j][0]
                final_dist_to_goal2 = self.x[i*4+1, self.horizon] - self.goal_list[i][j][1]
                final_dist_to_goal3 = -self.x[i*4, self.horizon] + self.goal_list[i][j][0]
                final_dist_to_goal4 = -self.x[i*4+1, self.horizon] + self.goal_list[i][j][1]
                h1_terminal = self.goal_area_size - final_dist_to_goal1
                h2_terminal = self.goal_area_size - final_dist_to_goal2
                h3_terminal = self.goal_area_size - final_dist_to_goal3
                h4_terminal = self.goal_area_size - final_dist_to_goal4
                if len(self.goal_list[i])>1:
                    constraints.append(h1_terminal + self.gamma_goal1[i][j][self.horizon] >= -slack_terminal1[i][j] - M * (1 - z[i][j]))
                    constraints.append(h2_terminal + self.gamma_goal2[i][j][self.horizon] >= -slack_terminal2[i][j] - M * (1 - z[i][j]))
                    constraints.append(h3_terminal + self.gamma_goal3[i][j][self.horizon] >= -slack_terminal3[i][j] - M * (1 - z[i][j]))
                    constraints.append(h4_terminal + self.gamma_goal4[i][j][self.horizon] >= -slack_terminal4[i][j] - M * (1 - z[i][j]))
                else:
                    constraints.append(h1_terminal + self.gamma_goal1[i][j][self.horizon] >= -slack_terminal1[i][j])
                    constraints.append(h2_terminal + self.gamma_goal2[i][j][self.horizon] >= -slack_terminal2[i][j])
                    constraints.append(h3_terminal + self.gamma_goal3[i][j][self.horizon] >= -slack_terminal3[i][j])
                    constraints.append(h4_terminal + self.gamma_goal4[i][j][self.horizon] >= -slack_terminal4[i][j]) 
            self.control_cost += cp.quad_form(self.u[:, k], self.R)
            self.control_cost += cp.quad_form(self.x[:, k], self.Q)
            for i in range(self.number_of_agents):
                distance_from_goal = cp.norm(self.x[i*4:i*4+2, k] - self.xG[i*4:i*4+2], 2)
                self.stage_cost += 0.5*distance_from_goal**2

        for i in range(self.number_of_agents):
            self.slack_cost_goal += slack_goal_weight*(cp.sum(cp.sum(slack_cbf1[i])) + cp.sum(cp.sum(slack_cbf2[i])) + cp.sum(cp.sum(slack_cbf3[i])) + cp.sum(cp.sum(slack_cbf4[i]))
                                                       + cp.sum(slack_terminal1[i]) + cp.sum(slack_terminal2[i]) + cp.sum(slack_terminal3[i]) + cp.sum(slack_terminal4[i]))
            self.slack_cost_obs += slack_obs_weight*cp.sum(slack_obs[i])
        for i in range(self.number_of_agents):
            terminal_state = self.x[i*4:i*4+2, -1]     # position at final time step
            goal_state = self.xG[i*4:i*4+2]            # desired goal position
            terminal_dist = cp.norm(terminal_state - goal_state, 2)
            self.terminal_cost += 1.0 * terminal_dist**2
        for i in range(self.number_of_combinations):
            self.slack_cost_coll += slack_coll_weight*slack_coll[i]
        #total_cost = self.slack_cost_goal + self.stage_cost + self.slack_cost_obs + self.terminal_cost
        total_cost = self.slack_cost_goal + self.slack_cost_obs + self.slack_cost_coll + self.control_cost + self.stage_cost
        self.problem = cp.Problem(cp.Minimize(total_cost), constraints)

    def solve_mpc(self, x0: np.ndarray, x_prev: np.ndarray=None, u_prev: np.ndarray=None, current_iteration:int=0):
        '''
        :param x0: Initial state of the system (flattened as a 1D array)
        :param x_prev: Previous state trajectory (used for warm-starting the optimization)
        :param u_prev: Previous control trajectory (used for warm-starting the optimization)
        :param current_iteration: Current iteration number (used for time-varying sets)
        '''
        self.x0.value = x0
        xG = np.zeros((4 * self.number_of_agents))  # Stores the closest goal for each agent
        closest_goals_indices = self.find_closest_goals(x0)
        for i in range(self.number_of_agents):
            xG[i*4:i*4+2] = self.goal_list[i][closest_goals_indices[i]][0:2]
        #self.xG.value = xG
        if (x_prev is None):
            x_prev = self.compute_initial_trajectory(x0)
        gamma_goal1 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}
        gamma_goal2 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}
        gamma_goal3 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}
        gamma_goal4 = {i: np.zeros((len(self.goal_list[i]), self.horizon+1)) for i in range(self.number_of_agents)}

        for i in range(self.number_of_agents):
            time_interval = (0,self.step_to_reach_goal[i]-current_iteration)
            switch = time_interval[1]+1
            for j in range(len(self.goal_list[i])):
                h0_goal1 = self.goal_area_size - (x0[i*4]-self.goal_list[i][j][0])
                h0_goal2 = self.goal_area_size - (x0[i*4+1]-self.goal_list[i][j][1])
                h0_goal3 = self.goal_area_size - (-x0[i*4]+self.goal_list[i][j][0])
                h0_goal4 = self.goal_area_size - (-x0[i*4+1]+self.goal_list[i][j][1])
                gamma0_goal1, tau_goal1 = self.define_gamma_params(time_interval, 'eventually', h0_goal1)
                gamma0_goal2, tau_goal2 = self.define_gamma_params(time_interval, 'eventually', h0_goal2)
                gamma0_goal3, tau_goal3 = self.define_gamma_params(time_interval, 'eventually', h0_goal3)
                gamma0_goal4, tau_goal4 = self.define_gamma_params(time_interval, 'eventually', h0_goal4)
                gamma_goal1[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal1, tau_goal1, switch)
                gamma_goal2[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal2, tau_goal2, switch)
                gamma_goal3[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal3, tau_goal3, switch)
                gamma_goal4[i][j] = self.compute_gamma(self.step_to_reach_goal[i], gamma0_goal4, tau_goal4, switch)
            self.gamma_goal1[i].value = gamma_goal1[i]
            self.gamma_goal2[i].value = gamma_goal2[i]
            self.gamma_goal3[i].value = gamma_goal3[i]
            self.gamma_goal4[i].value = gamma_goal4[i]
        
        parameter1_obs = {i: np.zeros((2*self.number_of_obs)) for i in range(self.number_of_agents)}
        parameter2_obs = {i: np.zeros((self.number_of_obs)) for i in range(self.number_of_agents)}
        for i in range(self.number_of_agents):
            for j, obs in enumerate(self.obs_list):
                obs_center = np.array([obs[0], obs[1]])
                obs_radius = obs[2]
                x0_i = x0[i*4:i*4+4]
                #u0_i = self.u[i*2:i*2+2, 0]
                Ap = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt]])
                Bp = np.array([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2]])
                epsilon = 1e-4
                dist = x0_i[0:2] - obs_center
                norm_dist = np.linalg.norm(dist)+epsilon
                db_dx = dist / norm_dist
                b = norm_dist - obs_radius # b(x0, t=0)
                parameter1_obs[i][j*2:j*2+2] = db_dx @ (Bp/self.dt)
                parameter2_obs[i][j] = db_dx @ ((Ap @ x0_i - x0_i[0:2])/self.dt) + self.alpha_obs * b
        
        for i in range(self.number_of_agents):
            self.parameter1_obs[i].value = parameter1_obs[i]
            self.parameter2_obs[i].value = parameter2_obs[i]


        diff_pos_nom = np.zeros((2, self.horizon+1))
        norm_diff_pos_nom = np.zeros(self.horizon+1)
        parameter1_coll = {i: np.zeros((self.horizon+1, 2)) for i in range(self.number_of_combinations)}
        parameter2_coll = {i: np.zeros(self.horizon+1) for i in range(self.number_of_combinations)}
        for idx, (i, j) in enumerate(itertools.combinations(range(self.number_of_agents), 2)) :
            for k in range(self.horizon-1):
                diff_pos_nom[:,k] = x_prev[i*4:i*4+2, k] - x_prev[j*4:j*4+2, k]
                norm_diff_pos_nom[k] = np.linalg.norm(diff_pos_nom[:,k]) + 1e-6
                parameter1_coll[idx][k,:] = diff_pos_nom.T[k,:] / norm_diff_pos_nom[k]
                parameter2_coll[idx][k] = - parameter1_coll[idx][k,:] @ diff_pos_nom[:,k] + norm_diff_pos_nom[k]
            
        for i in range(self.number_of_combinations):
            self.parameter1_coll[i].value = parameter1_coll[i]
            self.parameter2_coll[i].value = parameter2_coll[i]

        if (x_prev is not None):
            x_guess = np.hstack((x_prev[:, 1:], x_prev[:, -1:]))  # Shift state forward
            self.x.value = x_guess
        if u_prev is not None:
            u_guess = np.hstack((u_prev[:, 1:], u_prev[:, -1:]))  # Shift control forward
            self.u.value = u_guess
        try:
            self.problem.solve(solver=cp.GUROBI, warm_start=True)
            self.solver_time += self.problem.solver_stats.solve_time
            if self.problem.status in ["optimal", "optimal_inaccurate"]:
                u_opt = self.u.value[:,0]
                #print("control cost: ", self.control_cost.value)
                #print("stage cost: ", self.stage_cost.value)
                #print("slack cost: ", self.slack_cost_goal.value)
                #self.plot_cbf_set(self.x.value[:,-1], xG, self.gamma_goal1[0][0].value, self.goal_area_size, self.horizon)
                return u_opt, self.x.value, self.u.value
            else:
                print("Problem is not optimal. Status:", self.problem.status)
                #if self.problem.status == "infeasible":
                #    print("Infeasible constraints:", self.problem.constraints[self.problem.infeasible_constraints])
                return None, None, None
        except cp.SolverError as e:
            print('Solver failed:', e)
            return None, None, None
        
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
        gamma_values = gamma0 - (gamma0 / tau) * time_values
        gamma_values = np.maximum(gamma_values, 0)
        if T < self.horizon+1:
            gamma_values = np.concatenate((gamma_values, np.zeros(self.horizon+1-T)))
        if switch is not None:
            gamma_values[switch:]=1e3
        return gamma_values[:self.horizon+1]
    
    def plot_cbf_set(self, t, xG, gamma_values, goal_area_size, horizon):
        """
        Plots only the set where the CBF condition b(x, t) = h(x) + gamma(t) >= 0 holds,
        and it is only plotted at the last iteration (horizon) of the MPC.
        """
        # Get the value of gamma(t) for the current time step
        gamma_t = gamma_values[horizon]
        # Generate a grid of points to evaluate b(x,t) over the space
        x_vals = np.linspace(-10, 110, 100)
        y_vals = np.linspace(-10, 110, 100)
        # Create meshgrid for evaluating b(x,t) over the area
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Initialize array to store values of b(x,t)
        b_values = np.zeros_like(X)
        
        # Loop over all points in the grid and evaluate b(x,t) = h(x) + gamma(t)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Construct the current state (x, y) as a vector
                state = np.array([X[i, j], Y[i, j]])
                
                # Compute h(x), the distance to the goal (position tolerance)
                dist_to_goal = np.linalg.norm(state - xG[:2])  # Only considering the 2D position
                
                # Calculate the control barrier function b(x,t) = h(x) + gamma(t)
                h_x = goal_area_size - dist_to_goal
                b_values[i, j] = h_x + gamma_t
        
        # Plot the set where b(x,t) >= 0 (i.e., the region where b(x,t) is non-negative)
        plt.figure(figsize=(6, 6))
        plt.title("CBF Set at Final Iteration")
        plt.xlabel("x position")
        plt.ylabel("y position")
        
        # Plot the goal position (as a red dot)
        plt.plot(xG[0], xG[1], 'ro', label="Goal Position")
        
        # Plot the region where b(x,t) >= 0, using a contour plot
        plt.contour(X, Y, b_values, levels=[0], colors='b', label="CBF Region (b(x,t) >= 0)")
        
        # Set fixed x and y limits to ensure consistency in the plot view
        plt.xlim(-10, 110)
        plt.ylim(-10, 110)

        # Show plot with labels and grid
        plt.legend()
        plt.grid(True)
        plt.show()

    def compute_initial_trajectory(self, x0):
        closest_goals_indices = self.find_closest_goals(x0)
        #ToDo: ora che hai calcolato il goal piu vicino ad ogni robot calcola il controllo necessario per avvicinarsi
        u = np.zeros((self.nu*self.number_of_agents, self.horizon))
        x = np.zeros((self.nx*self.number_of_agents, self.horizon+1))
        x[:,0] = x0
        for i in range(self.number_of_agents):
            goal = self.goal_list[i][closest_goals_indices[i]][0:2]
            #print("closest goal: ", goal)
            pos = x0[i*4:i*4+2]
            #vel = x0[i*4+2:i*4+4]
            t_values = np.linspace(0, 1, self.step_to_reach_goal[i] + 1)
            trajectory = np.outer(1 - t_values, pos) + np.outer(t_values, goal)
            trajectory = trajectory.T
            x[i*4:i*4+2, :] = trajectory[:, :self.horizon+1]
        return x
    
    def find_closest_goals(self, x0):
        closest_goals_indices = []
        for i in range(self.number_of_agents):
            min_dist = 1e6
            initial_pos = x0[i*4:i*4+2]
            for j in range(len(self.goal_list[i])):
                dist = np.linalg.norm(initial_pos-self.goal_list[i][j][0:2])
                if dist < min_dist:
                    min_dist = dist
                    closest_goal = j
            closest_goals_indices.append(closest_goal)
        return closest_goals_indices
            
        
