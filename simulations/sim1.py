import numpy as np
import random
import time
import matplotlib.pyplot as plt
from stlpy.systems import LinearSystem

from stl_games.environment.generate_valid_positions import GenerateValidPositions
from stl_games.trajectory.trajectory_handler import ComputeTrajectories
from stl_games.stl.stl_specs import ObstacleAvoidanceSTLSpecs, GoalDistanceSTLSpecs
from stl_games.mpc.mpc_high_level2 import MPCHighLevelPlanner
from stl_games.plot.plot_result import PlotResult


# Set random seed for reproducibility
random.seed(1)

# Define parameters for the experiment
number_of_robots = 2
eps = 2.0       # Tolerance for goal distance
T = 30          # Time for reaching goal
safe_dist = 3   # Safety distance from other robots
safe_dist_obs = 1 # Safety distance from obstacles
grid_size = 100

# Define the linear system dynamics for each robot (double integrator model)
dt = 0.25
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

# Define the environment (obstacles, goal positions, initial positions)

# Define the bounds of the obstacles in the format (x_min, x_max, y_min, y_max)
obstacle_bounds_list_mpc = [
    (15, 25, 30, 40),
    (60, 70, 80, 90),
    (5, 15, 50, 60),
    (40, 50, 10, 20),
    (70, 80, 20, 30),
    (25, 35, 75, 85),
    (85, 95, 55, 65)]

# Generate valid start positions for the robots (not too close to obstacles or to each other)
generate_valid_pos = GenerateValidPositions()
generate_valid_pos.generate_valid_start_positions(grid_size, number_of_robots, obstacle_bounds_list_mpc, safe_dist+1, safe_dist_obs+1)
start_positions = generate_valid_pos.start_positions
print("start positions: ", start_positions)
x0 = start_positions.flatten()

# Generate valid goal positions for the robots (not too close to obstacles)
number_of_goals_total = 4
number_of_goals = ([2, 2])  # Number of goals for each robot
generate_valid_pos = GenerateValidPositions()
generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, obstacle_bounds_list_mpc, safe_dist+1, safe_dist_obs+1, number_of_robots, number_of_goals)
goal_positions = generate_valid_pos.goal_positions
xG = generate_valid_pos.xG
print("goal positions for each robot: ", xG)

# Define the parameters for the MPC
dt = 0.25
horizon_mpc = 4
nx = 4  # Number of states (x, y, vx, vy) for each robot
nu = 2  # Number of control inputs (ax, ay) for each robot
u_min = -3  # Minimum control input (acceleration)
u_max = 3   # Maximum control input (acceleration)

# Define the time each robot has to reach its goal
additional_points = int((1/dt)-1)
time_to_reach_goals = np.array((T, T, T))
step_to_reach_goal = time_to_reach_goals*(1+additional_points)

# Build MPC problem
start_time = time.time()
mpc = MPCHighLevelPlanner(nx, nu, number_of_robots, horizon_mpc, dt, u_min, u_max, safe_dist, eps, obstacle_bounds_list_mpc, step_to_reach_goal, xG)
mpc.build_problem(x0)

# Solve the MPC in a receiding horizon fashion
state_trajectory = [x0]
control_trajectory = []
x_current = x0
max_iters_mpc = (1+additional_points)*T
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
#print("Robot1 max velocity: ", np.max(np.abs(state_trajectory[2:4,:])))
#print("Robot2 max velocity: ", np.max(np.abs(state_trajectory[6:8,:])))

# Plot results
plot = PlotResult()
plot(state_trajectory, x0, goal_positions, number_of_goals, [], number_of_robots, obstacle_bounds_list_mpc, safe_dist, eps)
total_time = end_time-start_time
print("Total execution time: ", total_time)

# Analyze robustness of the resulting trajectory to STL specifications
control_trajectory = np.hstack((control_trajectory, np.zeros((control_trajectory.shape[0],1))))
compute_traj = ComputeTrajectories()
compute_traj.compute_y(state_trajectory, control_trajectory, number_of_robots)
y = compute_traj.y
goal_spec_ = GoalDistanceSTLSpecs()
goal_spec_(eps, goal_positions, number_of_goals, number_of_robots, y.shape[1]-1)
goal_spec = goal_spec_.goal_distance_spec
obstacle_avoidance_spec_ = ObstacleAvoidanceSTLSpecs()
obstacle_avoidance_spec_(obstacle_bounds_list_mpc, number_of_robots, y.shape[1]-1, safe_dist_obs)
obstacle_avoidance_spec = obstacle_avoidance_spec_.obstacle_avoidance_spec
combined_spec = obstacle_avoidance_spec & goal_spec
robustness_goal_reaching = goal_spec.robustness(y, 0)
robustness_obstacle_avoidance = obstacle_avoidance_spec.robustness(y,0)
print('robustness for reaching goal: ', robustness_goal_reaching)
print('robustness for obstacle avoidance: ', robustness_obstacle_avoidance)

# Write the trajectory to a file (used for the visualization in Rviz)
'''
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