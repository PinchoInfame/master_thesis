import numpy as np
import itertools
import random
import time
import matplotlib.pyplot as plt
from generate_valid_positions import GenerateValidPositions
from obstacle_avoidance_stl_specs import ObstacleAvoidanceSTLSpecs
from augment_control_inputs import AugmentControlInput
from compute_trajectories import ComputeTrajectories
from mpc_high_level2 import MPCHighLevelPlanner
from goal_distance_stl_specs import GoalDistanceSTLSpecs
from stlpy.systems import LinearSystem
from product_dynamical_system import ProductDynamicalSystem
from stlpy.solvers.gurobi.gurobi_micp import GurobiMICPSolver
from plot_result import PlotResult
from compute_additional_points import ComputeAdditionalPoints
from mpc_cbf_linearized_multipleGoals import MPC_cbf_multiple_goals

# Set random seed for reproducibility
random.seed(1)
number_of_robots = 2

# Define parameters
eps = 1.0
T = 40
safe_dist = 3
safe_dist_obs = 1
grid_size = 100

# Define initial state and obstacles
obstacle_bounds_list_mpc = [
    (15, 25, 30, 40),
    (60, 70, 80, 90),
    (5, 15, 50, 60),
    (40, 50, 10, 20),
    (70, 80, 20, 30),
    (25, 35, 75, 85),
    (85, 95, 55, 65)]
generate_valid_pos = GenerateValidPositions()
generate_valid_pos.generate_valid_start_positions(grid_size, number_of_robots, obstacle_bounds_list_mpc, safe_dist+1, safe_dist_obs+1)
start_positions = generate_valid_pos.start_positions
print("start positions: ", start_positions)
x0 = start_positions.flatten()

# Modify trajectory to avoid collision with mpc receiding horizon approach
start_time = time.time()
dt = 0.25
horizon_mpc = 4
nx = 4
nu = 2
u_min = -3
u_max = 3
position_tolerance = eps
number_of_goals_total = 4
number_of_goals = ([2, 2])
robot_id_list_associated_goals = sum([[i] * number_of_goals[i] for i in range(len(number_of_goals))], [])
generate_valid_pos = GenerateValidPositions()
generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, obstacle_bounds_list_mpc, safe_dist+1, safe_dist_obs+1)
goal_positions = generate_valid_pos.goal_positions
goal_positions_array = np.array(goal_positions)
print("goal positions: ", goal_positions)
#Todo: generate xG starting from goal positions
xG = []
for i in range(number_of_robots):
    xGi = []
    for j in range(number_of_goals[i]):
        xGi.append(goal_positions[j])
    goal_positions = goal_positions[number_of_goals[i]:]
    xG.append(xGi)

Ad = np.array([[1, 0, dt, 0],  
                [0, 1, 0, dt],  
                [0, 0, 1, 0],  
                [0, 0, 0, 1]])
Bd = np.array([[(dt**2)/2, 0],  
                [0, (dt**2)/2],  
                [dt, 0],  
                [0, dt]])
Cd = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])
Dd = np.array([[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1]])

A_full = np.kron(np.eye(number_of_robots), Ad)
B_full = np.kron(np.eye(number_of_robots), Bd)
C_full = np.kron(np.eye(number_of_robots), Cd)
D_full = np.kron(np.eye(number_of_robots), Dd)
combined_system = LinearSystem(A_full, B_full, C_full, D_full)


'''
goal_spec_ = GoalDistanceSTLSpecs()
goal_spec_(eps, goal_positions_array, number_of_goals, number_of_robots, T)
goal_spec = goal_spec_.goal_distance_spec
obstacle_avoidance_spec_ = ObstacleAvoidanceSTLSpecs()
obstacle_avoidance_spec_(obstacle_bounds_list_mpc, number_of_robots, T, safe_dist_obs)
obstacle_avoidance_spec = obstacle_avoidance_spec_.obstacle_avoidance_spec
combined_spec = obstacle_avoidance_spec & goal_spec
solver = GurobiMICPSolver(combined_spec, combined_system, x0, T)
solver.AddQuadraticCost(Q=1e-1*np.diag([0, 0, 1, 1] * (number_of_robots)), R=1e-1*np.eye(number_of_robots*2))
solver.AddControlBounds(u_min=-3, u_max=3)
#solver.AddRobustnessConstraint(rho_min=1.0)
solver.AddRobustnessCost()
x, u, rho, solver_time = solver.Solve()

plot = PlotResult()
plt.figure(figsize=(10, 6))
plot(x, x0, goal_positions_array, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list_mpc, safe_dist)
plt.show()
compute_traj = ComputeTrajectories()
compute_traj.compute_y(x, u, number_of_robots)
y_milp = compute_traj.y
robustness_goal_reaching_milp = goal_spec.robustness(y_milp, 0)
robustness_obstacle_avoidance_milp = obstacle_avoidance_spec.robustness(y_milp,0)
print('robustness for reaching goal: ', robustness_goal_reaching_milp)
print('robustness for obstacle avoidance: ', robustness_obstacle_avoidance_milp)
'''

additional_points = int((1/dt)-1)
time_to_reach_goals = np.array((T, T, T))
step_to_reach_goal = time_to_reach_goals*(1+additional_points)

mpc = MPCHighLevelPlanner(nx, nu, number_of_robots, horizon_mpc, dt, u_min, u_max, safe_dist, position_tolerance, obstacle_bounds_list_mpc, step_to_reach_goal, xG)
mpc.build_problem(x0)

state_trajectory = [x0]
control_trajectory = []
x_current = x0
max_iters_mpc = (1+additional_points)*T
print(max_iters_mpc)
#x_prev_init = x[:, :horizon_mpc+1]
#u_prev_init = u[:, :horizon_mpc]
u_opt, x_prev, u_prev = mpc.solve_mpc(x0, None, None, current_iteration=0)
if u_opt is None:
    print("Solver failed")
    exit()
x_current = mpc.define_dynamics(x0, u_opt)
state_trajectory.append(x_current)
control_trajectory.append(u_opt)
for t in range(1, max_iters_mpc):
    u_opt, x_prev, u_prev = mpc.solve_mpc(x_current, x_prev, u_prev, t)
    if u_opt is None:
        print("Solver failed")
        exit()
    x_current = mpc.define_dynamics(x_current, u_opt)
    state_trajectory.append(x_current)
    control_trajectory.append(u_opt)

state_trajectory = np.array(state_trajectory).T
control_trajectory = np.array(control_trajectory).T
end_time = time.time()
print("Solver time: ", mpc.solver_time)
#print("Robot1 velocity: ", np.max(np.abs(state_trajectory[:,2:4])))
#print("Robot2 velocity: ", np.max(np.abs(state_trajectory[:,6:8])))
plot = PlotResult()
plt.figure(figsize=(10, 6))
plot(state_trajectory, x0, goal_positions_array, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list_mpc, safe_dist)
plt.show()
total_time = end_time-start_time
print("Total execution time: ", total_time)

'''
'x_prev = state_trajectory[:, :horizon_mpc+1]
u_prev = control_trajectory[:, :horizon_mpc]
state_trajectory = [x0]
control_trajectory = []
x_current = x0
for t in range(max_iters_mpc):
    u_opt, x_prev, u_prev = mpc.solve_mpc(x_current, x_prev, u_prev, t)
    if u_opt is None:
        print("Solver failed")
        exit()
    x_current = mpc.define_dynamics(x_current, u_opt)
    state_trajectory.append(x_current)
    control_trajectory.append(u_opt)

state_trajectory = np.array(state_trajectory).T
control_trajectory = np.array(control_trajectory).T
plot = PlotResult()
plt.figure(figsize=(10, 6))
plot(state_trajectory, x0, goal_positions_array, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list_mpc, safe_dist)
plt.show()'
'''

control_trajectory = np.hstack((control_trajectory, np.zeros((control_trajectory.shape[0],1))))
compute_traj = ComputeTrajectories()
compute_traj.compute_y(state_trajectory, control_trajectory, number_of_robots)
y = compute_traj.y
goal_spec_ = GoalDistanceSTLSpecs()
goal_spec_(eps, goal_positions_array, number_of_goals, number_of_robots, y.shape[1]-1)
goal_spec = goal_spec_.goal_distance_spec
obstacle_avoidance_spec_ = ObstacleAvoidanceSTLSpecs()
obstacle_avoidance_spec_(obstacle_bounds_list_mpc, number_of_robots, y.shape[1]-1, safe_dist_obs)
obstacle_avoidance_spec = obstacle_avoidance_spec_.obstacle_avoidance_spec
combined_spec = obstacle_avoidance_spec & goal_spec
robustness_goal_reaching = goal_spec.robustness(y, 0)
robustness_obstacle_avoidance = obstacle_avoidance_spec.robustness(y,0)
print('robustness for reaching goal: ', robustness_goal_reaching)
print('robustness for obstacle avoidance: ', robustness_obstacle_avoidance)

data_pairs = state_trajectory[4:6,:].T  

# Open file and write
with open('trajectory.txt', 'w') as f:
    for idx, (x, y) in enumerate(data_pairs):
        f.write(f'[{x:.1f}, {y:.1f}]')
        if (idx + 1) % 5 == 0:  # Every 5 entries
            f.write(', ')
            f.write('\n')
        else:
            f.write(', ')


'''
compute_traj.compute_y(x, u, number_of_robots)
y_milp = compute_traj.y
goal_spec_ = GoalDistanceSTLSpecs()
goal_spec_(eps, , robot_id_list_associated_goals, number_of_robots, y_milp.shape[1]-1)
goal_spec = goal_spec_.goal_distance_spec
obstacle_avoidance_spec_ = ObstacleAvoidanceSTLSpecs()
obstacle_avoidance_spec_(obstacle_bounds_list_mpc, number_of_robots, y_milp.shape[1]-1, safe_dist_obs)
obstacle_avoidance_spec = obstacle_avoidance_spec_.obstacle_avoidance_spec
combined_spec = obstacle_avoidance_spec & goal_spec
robustness_goal_reaching_milp = goal_spec.robustness(y_milp, 0)
robustness_obstacle_avoidance_milp = obstacle_avoidance_spec.robustness(y_milp,0)
print('robustness for reaching goal MILP: ', robustness_goal_reaching_milp)
print('robustness for obstacle avoidance MILP: ', robustness_obstacle_avoidance_milp)
exit()
solver = GurobiMICPSolver(combined_spec, combined_system, x0, T)
solver.AddQuadraticCost(Q=1e-1*np.diag([0, 0, 1, 1] * (number_of_robots)), R=1e-1*np.eye(number_of_robots*2))
solver.AddControlBounds(u_min=-5, u_max=5)
#solver.AddRobustnessConstraint(rho_min=0.1)
solver.AddRobustnessCost()
x, u, rho, solver_time = solver.Solve()

plot = PlotResult()
plt.figure(figsize=(10, 6))
plot(x, x0, goal_positions_array, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list_mpc, safe_dist)
plt.show()'
'''