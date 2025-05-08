import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

class MPC_cbf_multiple_goals:
    def __init__(self, nx, nu, horizon, dt, u_min, u_max, safe_dist, position_tolerance, obs_list, step_to_reach_goal, list_xG1, list_xG2):
        self.solver_time = 0
        self.horizon = horizon
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.R = 0.01*np.diag([1, 1, 1, 1])
        self.safe_dist = safe_dist + 1
        self.position_tolerance = position_tolerance
        self.obs_list = obs_list
        self.number_of_obs = len(obs_list)
        self.slack_cost = None
        self.step_to_reach_goal = step_to_reach_goal
        self.list_xG1 = list_xG1 # list with dimension number_of_goals x 4
        self.list_xG2 = list_xG2
        self.number_of_goals1 = len(list_xG1)
        self.number_of_goals2 = len(list_xG2)
        self.create_optimization_variables()

    def create_optimization_variables(self):
        self.x = cp.Variable((self.nx, self.horizon + 1))
        self.u = cp.Variable((self.nu, self.horizon))
        self.x0 = cp.Parameter(self.nx)
        self.xG1 = cp.Parameter((self.number_of_goals1, 4))
        self.xG2 = cp.Parameter((self.number_of_goals2, 4))
        self.gamma_goal1_1 = cp.Parameter((self.number_of_goals1, self.horizon+1))
        self.gamma_goal1_2 = cp.Parameter((self.number_of_goals1, self.horizon+1))
        self.gamma_goal1_3 = cp.Parameter((self.number_of_goals1, self.horizon+1))
        self.gamma_goal1_4 = cp.Parameter((self.number_of_goals1, self.horizon+1))
        self.gamma_goal2_1 = cp.Parameter((self.number_of_goals2, self.horizon+1))
        self.gamma_goal2_2 = cp.Parameter((self.number_of_goals2, self.horizon+1))
        self.gamma_goal2_3 = cp.Parameter((self.number_of_goals2, self.horizon+1))
        self.gamma_goal2_4 = cp.Parameter((self.number_of_goals2, self.horizon+1))
        self.x_prev = cp.Parameter((self.nx, self.horizon + 1))
        self.parameter1_coll = cp.Parameter((self.horizon+1, 2))
        self.parameter2_coll = cp.Parameter(self.horizon+1)
        self.parameter1_obs_robot1 = cp.Parameter((self.horizon+1, 2*self.number_of_obs))
        self.parameter2_obs_robot1 = cp.Parameter((self.horizon+1, self.number_of_obs))
        self.parameter1_obs_robot2 = cp.Parameter((self.horizon+1, 2*self.number_of_obs))
        self.parameter2_obs_robot2 = cp.Parameter((self.horizon+1, self.number_of_obs))

    def define_dynamics(self, state, input):
        dt = self.dt
        self.A = np.array([  [1, 0, dt, 0,  0, 0, 0, 0],  
                        [0, 1, 0, dt,  0, 0, 0, 0],  
                        [0, 0, 1, 0,  0, 0, 0, 0],  
                        [0, 0, 0, 1,  0, 0, 0, 0],  
                    
                        [0, 0, 0, 0,  1, 0, dt, 0],  
                        [0, 0, 0, 0,  0, 1, 0, dt],  
                        [0, 0, 0, 0,  0, 0, 1, 0],  
                        [0, 0, 0, 0,  0, 0, 0, 1]])
        
        self.B = np.array([  [(dt**2)/2, 0, 0, 0],  
                        [0, (dt**2)/2, 0, 0],  
                        [dt, 0, 0, 0],  
                        [0, dt, 0, 0],  
                  
                        [0, 0, (dt**2)/2, 0],  
                        [0, 0, 0, (dt**2)/2],  
                        [0, 0, dt, 0],  
                        [0, 0, 0, dt]])         
        return self.A @ state + self.B @ input

    def build_problem(self, x0, list_xG1, list_xG2):
        control_cost = 0
        v_max = 10
        constraints = []
        constraints.append(self.x[:, 0] == self.x0)
        slack_cbf1_1 = cp.Variable((self.number_of_goals1, self.horizon), nonneg=True)
        slack_cbf1_2 = cp.Variable((self.number_of_goals1, self.horizon), nonneg=True)
        slack_cbf1_3 = cp.Variable((self.number_of_goals1, self.horizon), nonneg=True)
        slack_cbf1_4 = cp.Variable((self.number_of_goals1, self.horizon), nonneg=True)
        slack_cbf2_1 = cp.Variable((self.number_of_goals2, self.horizon), nonneg=True)
        slack_cbf2_2 = cp.Variable((self.number_of_goals2, self.horizon), nonneg=True)
        slack_cbf2_3 = cp.Variable((self.number_of_goals2, self.horizon), nonneg=True)
        slack_cbf2_4 = cp.Variable((self.number_of_goals2, self.horizon), nonneg=True)

        slack_terminal1_1 = cp.Variable((self.number_of_goals1), nonneg=True)
        slack_terminal1_2 = cp.Variable((self.number_of_goals1), nonneg=True)
        slack_terminal1_3 = cp.Variable((self.number_of_goals1), nonneg=True)
        slack_terminal1_4 = cp.Variable((self.number_of_goals1), nonneg=True)
        slack_terminal2_1 = cp.Variable((self.number_of_goals2), nonneg=True)
        slack_terminal2_2 = cp.Variable((self.number_of_goals2), nonneg=True)
        slack_terminal2_3 = cp.Variable((self.number_of_goals2), nonneg=True)
        slack_terminal2_4 = cp.Variable((self.number_of_goals2), nonneg=True)

        slack_cost_weight = 500
        self.alpha_coll = 1.0
        self.alpha_obs = 0.3
        time_interval = (0,self.step_to_reach_goal)
        switch = self.step_to_reach_goal+1
        gamma_goal1_1 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal1_2 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal1_3 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal1_4 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal2_1 = np.zeros((self.number_of_goals2, self.horizon+1))
        gamma_goal2_2 = np.zeros((self.number_of_goals2, self.horizon+1))
        gamma_goal2_3 = np.zeros((self.number_of_goals2, self.horizon+1))
        gamma_goal2_4 = np.zeros((self.number_of_goals2, self.horizon+1))

        for i in range(self.number_of_goals1):
            h0_goal1_1 = self.position_tolerance - (x0[0]-list_xG1[i,0])
            h0_goal1_2 = self.position_tolerance - (x0[1]-list_xG1[i,1])
            h0_goal1_3 = self.position_tolerance - (-x0[0]+list_xG1[i,0])
            h0_goal1_4 = self.position_tolerance - (-x0[1]+list_xG1[i,1])
            gamma0_goal1_1, tau_goal1_1 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_1)
            gamma0_goal1_2, tau_goal1_2 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_2)
            gamma0_goal1_3, tau_goal1_3 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_3)
            gamma0_goal1_4, tau_goal1_4 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_4)
            gamma_goal1_1[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_1, tau_goal1_1, switch)
            gamma_goal1_2[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_2, tau_goal1_2, switch)
            gamma_goal1_3[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_3, tau_goal1_3, switch)
            gamma_goal1_4[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_4, tau_goal1_4, switch)
        self.gamma_goal1_1.value = gamma_goal1_1
        self.gamma_goal1_2.value = gamma_goal1_2
        self.gamma_goal1_3.value = gamma_goal1_3
        self.gamma_goal1_4.value = gamma_goal1_4

        for i in range(self.number_of_goals2):
            h0_goal2_1 = self.position_tolerance - (x0[4]-list_xG2[i,0])
            h0_goal2_2 = self.position_tolerance - (x0[5]-list_xG2[i,1])
            h0_goal2_3 = self.position_tolerance - (-x0[4]+list_xG2[i,0])
            h0_goal2_4 = self.position_tolerance - (-x0[5]+list_xG2[i,1])
            gamma0_goal2_1, tau_goal2_1 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_1)
            gamma0_goal2_2, tau_goal2_2 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_2)
            gamma0_goal2_3, tau_goal2_3 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_3)
            gamma0_goal2_4, tau_goal2_4 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_4)
            gamma_goal2_1[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_1, tau_goal2_1, switch)
            gamma_goal2_2[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_2, tau_goal2_2, switch)
            gamma_goal2_3[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_3, tau_goal2_3, switch)
            gamma_goal2_4[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_4, tau_goal2_4, switch)       
        self.gamma_goal2_1.value = gamma_goal2_1
        self.gamma_goal2_2.value = gamma_goal2_2
        self.gamma_goal2_3.value = gamma_goal2_3
        self.gamma_goal2_4.value = gamma_goal2_4

        #ToDo: use binary variables just if there is more than one goal (do this for each robot)
        if self.number_of_goals1 > 1:
            #ToDo: try also setting sum(z)>=1, try also goal clustering or penalising sum(z)
            z1 = cp.Variable((self.number_of_goals1), boolean=True)  # Binary variable for Set 1
            constraints.append(cp.sum(z1)>=1)
        if self.number_of_goals2 > 1:
            z2 = cp.Variable((self.number_of_goals2), boolean=True)  # Binary variable for Set 2
            constraints.append(cp.sum(z2)>=1)
        M = 1e7  # Large constant
              
        for k in range(self.horizon-1):
            x_next = self.define_dynamics(self.x[:, k], self.u[:, k])
            constraints.append(self.x[:, k + 1] == x_next)
            constraints.append(self.u[:, k] >= self.u_min)
            constraints.append(self.u[:, k] <= self.u_max)
            #constraints.append(cp.abs(self.x[2, k]) <= v_max)
            #constraints.append(cp.abs(self.x[3, k]) <= v_max)
            #constraints.append(cp.abs(self.x[6, k]) <= v_max)
            #constraints.append(cp.abs(self.x[7, k]) <= v_max)

            # Goal reaching with cbf: b(x, t) = h(x) + gamma(t) >= 0
            for i in range(self.number_of_goals1):
                dist_to_goal1_1 = self.x[0, k] - self.xG1[i,0]
                dist_to_goal1_2 = self.x[1, k] - self.xG1[i,1]
                dist_to_goal1_3 = -self.x[0, k] + self.xG1[i,0]
                dist_to_goal1_4 = -self.x[1, k] + self.xG1[i,1]
                h1_1 = self.position_tolerance - dist_to_goal1_1
                h1_2 = self.position_tolerance - dist_to_goal1_2
                h1_3 = self.position_tolerance - dist_to_goal1_3
                h1_4 = self.position_tolerance - dist_to_goal1_4
                if self.number_of_goals1 > 1:
                    constraints.append(h1_1 + self.gamma_goal1_1[i,k] >= -slack_cbf1_1[i,k] - M * (1 - z1[i]))
                    constraints.append(h1_2 + self.gamma_goal1_2[i,k] >= -slack_cbf1_2[i,k] - M * (1 - z1[i]))
                    constraints.append(h1_3 + self.gamma_goal1_3[i,k] >= -slack_cbf1_3[i,k] - M * (1 - z1[i]))
                    constraints.append(h1_4 + self.gamma_goal1_4[i,k] >= -slack_cbf1_4[i,k] - M * (1 - z1[i]))
                else:
                    constraints.append(h1_1 + self.gamma_goal1_1[i,k] >= -slack_cbf1_1[i,k])
                    constraints.append(h1_2 + self.gamma_goal1_2[i,k] >= -slack_cbf1_2[i,k])
                    constraints.append(h1_3 + self.gamma_goal1_3[i,k] >= -slack_cbf1_3[i,k])
                    constraints.append(h1_4 + self.gamma_goal1_4[i,k] >= -slack_cbf1_4[i,k])

            for i in range(self.number_of_goals2):
                dist_to_goal2_1 = self.x[4, k] - self.xG2[i,0]
                dist_to_goal2_2 = self.x[5, k] - self.xG2[i,1]
                dist_to_goal2_3 = -self.x[4, k] + self.xG2[i,0]
                dist_to_goal2_4 = -self.x[5, k] + self.xG2[i,1]
                h2_1 = self.position_tolerance - dist_to_goal2_1
                h2_2 = self.position_tolerance - dist_to_goal2_2
                h2_3 = self.position_tolerance - dist_to_goal2_3
                h2_4 = self.position_tolerance - dist_to_goal2_4
                if self.number_of_goals2 > 1:
                    constraints.append(h2_1 + self.gamma_goal2_1[i,k] >= -slack_cbf2_1[i,k] - M * (1 - z2[i]))
                    constraints.append(h2_2 + self.gamma_goal2_2[i,k] >= -slack_cbf2_2[i,k] - M * (1 - z2[i]))
                    constraints.append(h2_3 + self.gamma_goal2_3[i,k] >= -slack_cbf2_3[i,k] - M * (1 - z2[i]))
                    constraints.append(h2_4 + self.gamma_goal2_4[i,k] >= -slack_cbf2_4[i,k] - M * (1 - z2[i]))
                else: 
                    constraints.append(h2_1 + self.gamma_goal2_1[i,k] >= -slack_cbf2_1[i,k])
                    constraints.append(h2_2 + self.gamma_goal2_2[i,k] >= -slack_cbf2_2[i,k])
                    constraints.append(h2_3 + self.gamma_goal2_3[i,k] >= -slack_cbf2_3[i,k])
                    constraints.append(h2_4 + self.gamma_goal2_4[i,k] >= -slack_cbf2_4[i,k])

            # Collision avoidance with linearized CBF
            diff_pos = self.x[0:2, k] - self.x[4:6, k] 
            d_hat = (self.parameter1_coll[k,:]) @ (diff_pos) + self.parameter2_coll[k]
            h_collision = d_hat - self.safe_dist + self.alpha_coll * (d_hat - self.safe_dist)
            constraints.append(h_collision >= 0)
            # Linearized Obstacle Avoidance CBF
            for i, obs in enumerate(self.obs_list):
                obs_center = np.array([obs[0], obs[1]])  # (x, y) center
                obs_radius = obs[2]  # radius
                diff_obs_robot1 = self.x[0:2, k] - obs_center
                d_obs_hat_robot1 = (self.parameter1_obs_robot1[k,(i*2):(i*2)+2]) @ (diff_obs_robot1) + self.parameter2_obs_robot1[k,i]
                h_obs_robot1 = d_obs_hat_robot1 - obs_radius + self.alpha_obs * (d_obs_hat_robot1 - obs_radius)
                #constraints.append(h_obs_robot1 >= -slack_obs_robot1[i, k])
                constraints.append(h_obs_robot1 >= 0)
                diff_obs_robot2 = self.x[4:6, k] - obs_center
                d_obs_hat_robot2 = (self.parameter1_obs_robot2[k,(i*2):(i*2)+2]) @ (diff_obs_robot2) + self.parameter2_obs_robot2[k,i]
                h_obs_robot2 = d_obs_hat_robot2 - obs_radius + self.alpha_obs * (d_obs_hat_robot2 - obs_radius)
                #constraints.append(h_obs_robot2 >= -slack_obs_robot2[i, k])
                constraints.append(h_obs_robot2 >= 0)
            control_cost += cp.quad_form(self.u[:, k], self.R)
        # Define terminal CBF constraints with slack
        for i in range(self.number_of_goals1):
            final_dist_to_goal1_1 = self.x[0, self.horizon] - self.xG1[i,0]
            final_dist_to_goal1_2 = self.x[1, self.horizon] - self.xG1[i,1]
            final_dist_to_goal1_3 = -self.x[0, self.horizon] + self.xG1[i,0]
            final_dist_to_goal1_4 = -self.x[1, self.horizon] + self.xG1[i,1]
            h1_terminal1 = self.position_tolerance - final_dist_to_goal1_1
            h1_terminal2 = self.position_tolerance - final_dist_to_goal1_2
            h1_terminal3 = self.position_tolerance - final_dist_to_goal1_3
            h1_terminal4 = self.position_tolerance - final_dist_to_goal1_4
            if self.number_of_goals1 > 1:
                constraints.append(h1_terminal1 + self.gamma_goal1_1[i,self.horizon] >= -slack_terminal1_1[i] - M * (1 - z1[i]))
                constraints.append(h1_terminal2 + self.gamma_goal1_2[i,self.horizon] >= -slack_terminal1_2[i] - M * (1 - z1[i]))
                constraints.append(h1_terminal3 + self.gamma_goal1_3[i,self.horizon] >= -slack_terminal1_3[i] - M * (1 - z1[i]))
                constraints.append(h1_terminal4 + self.gamma_goal1_4[i,self.horizon] >= -slack_terminal1_4[i] - M * (1 - z1[i]))
            else:
                constraints.append(h1_terminal1 + self.gamma_goal1_1[i,self.horizon] >= -slack_terminal1_1[i])
                constraints.append(h1_terminal2 + self.gamma_goal1_2[i,self.horizon] >= -slack_terminal1_2[i])
                constraints.append(h1_terminal3 + self.gamma_goal1_3[i,self.horizon] >= -slack_terminal1_3[i])
                constraints.append(h1_terminal4 + self.gamma_goal1_4[i,self.horizon] >= -slack_terminal1_4[i])

        for i in range(self.number_of_goals2):
            final_dist_to_goal2_1 = self.x[4, self.horizon] - self.xG2[i,0]
            final_dist_to_goal2_2 = self.x[5, self.horizon] - self.xG2[i,1]
            final_dist_to_goal2_3 = -self.x[4, self.horizon] + self.xG2[i,0]
            final_dist_to_goal2_4 = -self.x[5, self.horizon] + self.xG2[i,1]
            h2_terminal1 = self.position_tolerance - final_dist_to_goal2_1
            h2_terminal2 = self.position_tolerance - final_dist_to_goal2_2
            h2_terminal3 = self.position_tolerance - final_dist_to_goal2_3
            h2_terminal4 = self.position_tolerance - final_dist_to_goal2_4
            if self.number_of_goals2 > 1:
                constraints.append(h2_terminal1 + self.gamma_goal2_1[i,self.horizon] >= -slack_terminal2_1[i] - M * (1 - z2[i]))
                constraints.append(h2_terminal2 + self.gamma_goal2_2[i,self.horizon] >= -slack_terminal2_2[i] - M * (1 - z2[i]))
                constraints.append(h2_terminal3 + self.gamma_goal2_3[i,self.horizon] >= -slack_terminal2_3[i] - M * (1 - z2[i]))
                constraints.append(h2_terminal4 + self.gamma_goal2_4[i,self.horizon] >= -slack_terminal2_4[i] - M * (1 - z2[i]))
            else:
                constraints.append(h2_terminal1 + self.gamma_goal2_1[i,self.horizon] >= -slack_terminal2_1[i])
                constraints.append(h2_terminal2 + self.gamma_goal2_2[i,self.horizon] >= -slack_terminal2_2[i])
                constraints.append(h2_terminal3 + self.gamma_goal2_3[i,self.horizon] >= -slack_terminal2_3[i])
                constraints.append(h2_terminal4 + self.gamma_goal2_4[i,self.horizon] >= -slack_terminal2_4[i])

        slack_cost_goal = slack_cost_weight*(cp.sum(cp.sum(slack_cbf1_1)) + cp.sum(cp.sum(slack_cbf1_2)) + cp.sum(cp.sum(slack_cbf1_3)) + cp.sum(cp.sum(slack_cbf1_4)) + 
                                             cp.sum(cp.sum(slack_cbf2_1)) + cp.sum(cp.sum(slack_cbf2_2)) + cp.sum(cp.sum(slack_cbf2_3)) + cp.sum(cp.sum(slack_cbf2_4)) + 
                                             cp.sum(slack_terminal1_1) + cp.sum(slack_terminal1_2) + cp.sum(slack_terminal1_3) + cp.sum(slack_terminal1_4) + 
                                             cp.sum(slack_terminal2_1) + cp.sum(slack_terminal2_2) + cp.sum(slack_terminal2_3) + cp.sum(slack_terminal2_4))
        total_cost = control_cost + slack_cost_goal
        self.problem = cp.Problem(cp.Minimize(total_cost), constraints)

    def solve_mpc(self, x0, list_xG1, list_xG2, x_prev=None, u_prev=None, current_iteration=0):
        self.x0.value = x0 
        self.xG1.value = list_xG1
        self.xG2.value = list_xG2
        time_interval = (0,self.step_to_reach_goal-current_iteration)
        switch = time_interval[1]+1
        gamma_goal1_1 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal1_2 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal1_3 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal1_4 = np.zeros((self.number_of_goals1, self.horizon+1))
        gamma_goal2_1 = np.zeros((self.number_of_goals2, self.horizon+1))
        gamma_goal2_2 = np.zeros((self.number_of_goals2, self.horizon+1))
        gamma_goal2_3 = np.zeros((self.number_of_goals2, self.horizon+1))
        gamma_goal2_4 = np.zeros((self.number_of_goals2, self.horizon+1))
        for i in range(self.number_of_goals1):
            h0_goal1_1 = self.position_tolerance - (x0[0]-list_xG1[i,0])
            h0_goal1_2 = self.position_tolerance - (x0[1]-list_xG1[i,1])
            h0_goal1_3 = self.position_tolerance - (-x0[0]+list_xG1[i,0])
            h0_goal1_4 = self.position_tolerance - (-x0[1]+list_xG1[i,1])
            gamma0_goal1_1, tau_goal1_1 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_1)
            gamma0_goal1_2, tau_goal1_2 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_2)
            gamma0_goal1_3, tau_goal1_3 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_3)
            gamma0_goal1_4, tau_goal1_4 = self.define_gamma_params(time_interval, 'eventually', h0_goal1_4)
            gamma_goal1_1[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_1, tau_goal1_1, switch)
            gamma_goal1_2[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_2, tau_goal1_2, switch)
            gamma_goal1_3[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_3, tau_goal1_3, switch)
            gamma_goal1_4[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal1_4, tau_goal1_4, switch)
        self.gamma_goal1_1.value = gamma_goal1_1
        self.gamma_goal1_2.value = gamma_goal1_2
        self.gamma_goal1_3.value = gamma_goal1_3
        self.gamma_goal1_4.value = gamma_goal1_4

        for i in range(self.number_of_goals2):
            h0_goal2_1 = self.position_tolerance - (x0[4]-list_xG2[i,0])
            h0_goal2_2 = self.position_tolerance - (x0[5]-list_xG2[i,1])
            h0_goal2_3 = self.position_tolerance - (-x0[4]+list_xG2[i,0])
            h0_goal2_4 = self.position_tolerance - (-x0[5]+list_xG2[i,1])
            gamma0_goal2_1, tau_goal2_1 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_1)
            gamma0_goal2_2, tau_goal2_2 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_2)
            gamma0_goal2_3, tau_goal2_3 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_3)
            gamma0_goal2_4, tau_goal2_4 = self.define_gamma_params(time_interval, 'eventually', h0_goal2_4)        
            gamma_goal2_1[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_1, tau_goal2_1, switch)
            gamma_goal2_2[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_2, tau_goal2_2, switch)
            gamma_goal2_3[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_3, tau_goal2_3, switch)
            gamma_goal2_4[i,:] = self.compute_gamma(self.step_to_reach_goal, gamma0_goal2_4, tau_goal2_4, switch)
        self.gamma_goal2_1.value = gamma_goal2_1
        self.gamma_goal2_2.value = gamma_goal2_2
        self.gamma_goal2_3.value = gamma_goal2_3
        self.gamma_goal2_4.value = gamma_goal2_4

        '''if x_prev is None:
            x_prev = np.linspace(x0, xG, self.horizon + 1).T'''
        # warm-startin strategy
        diff_pos_nom = np.zeros((2, self.horizon+1))
        norm_diff_pos_nom = np.zeros(self.horizon+1)
        parameter1_coll = np.zeros((self.horizon+1, 2))
        parameter2_coll = np.zeros(self.horizon+1)
        diff_obs_nom_robot1 = np.zeros((2*self.number_of_obs, self.horizon+1))
        norm_diff_obs_nom_robot1 = np.zeros((self.horizon+1, self.number_of_obs))
        diff_obs_nom_robot2 = np.zeros((2*self.number_of_obs, self.horizon+1))
        norm_diff_obs_nom_robot2 = np.zeros((self.horizon+1, self.number_of_obs))
        parameter1_obs_robot1 = np.zeros((self.horizon+1, 2*self.number_of_obs))
        parameter2_obs_robot1 = np.zeros((self.horizon+1, self.number_of_obs))
        parameter1_obs_robot2 = np.zeros((self.horizon+1, 2*self.number_of_obs))
        parameter2_obs_robot2 = np.zeros((self.horizon+1, self.number_of_obs))

        for k in range(self.horizon-1):
            diff_pos_nom[:,k] = x_prev[0:2, k] - x_prev[4:6, k]
            norm_diff_pos_nom[k] = np.linalg.norm(diff_pos_nom[:,k]) + 1e-6
            parameter1_coll[k,:] = diff_pos_nom.T[k,:] / norm_diff_pos_nom[k]
            parameter2_coll[k] = - parameter1_coll[k,:] @ diff_pos_nom[:,k] + norm_diff_pos_nom[k]
            for i, obs in enumerate(self.obs_list):
                obs_center = np.array([obs[0], obs[1]])  # (x, y) center
                diff_obs_nom_robot1[(i*2):(i*2)+2,k] = x_prev[0:2, k] - obs_center
                norm_diff_obs_nom_robot1[k,i] = np.linalg.norm(diff_obs_nom_robot1[(i*2):(i*2)+2,k]) + 1e-6 
                parameter1_obs_robot1[k, (i*2):(i*2)+2] = diff_obs_nom_robot1.T[k,(i*2):(i*2)+2] / norm_diff_obs_nom_robot1[k,i]
                parameter2_obs_robot1[k,i] = -parameter1_obs_robot1[k, (i*2):(i*2)+2] @ diff_obs_nom_robot1[(i*2):(i*2)+2,k] + norm_diff_obs_nom_robot1[k,i]
                diff_obs_nom_robot2[(i*2):(i*2)+2,k] = x_prev[4:6, k] - obs_center
                norm_diff_obs_nom_robot2[k,i] = np.linalg.norm(diff_obs_nom_robot2[(i*2):(i*2)+2,k]) + 1e-6 
                parameter1_obs_robot2[k, (i*2):(i*2)+2] = diff_obs_nom_robot2.T[k,(i*2):(i*2)+2] / norm_diff_obs_nom_robot2[k,i]
                parameter2_obs_robot2[k,i] = -parameter1_obs_robot2[k, (i*2):(i*2)+2] @ diff_obs_nom_robot2[(i*2):(i*2)+2,k] + norm_diff_obs_nom_robot2[k,i]
        self.parameter1_coll.value = parameter1_coll
        self.parameter2_coll.value = parameter2_coll
        self.parameter1_obs_robot1.value = parameter1_obs_robot1
        self.parameter2_obs_robot1.value = parameter2_obs_robot1
        self.parameter1_obs_robot2.value = parameter1_obs_robot2
        self.parameter2_obs_robot2.value = parameter2_obs_robot2

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
                #self.plot_cbf_set(self.x.value[:,-1], xG, self.gamma_goal1.value, self.position_tolerance, self.horizon)
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
            gamma_0 = 1.2*(-h0)
        elif h0>0:
            gamma_0 = 1.2*h0
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
    
    def plot_cbf_set(self, t, xG, gamma_values, position_tolerance, horizon):
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
                h_x = position_tolerance - dist_to_goal
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