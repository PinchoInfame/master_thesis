import numpy as np
import random
import time
from stlpy.systems import LinearSystem

from stl_games.environment.generate_valid_positions import GenerateValidPositions_4States
from stl_games.trajectory.trajectory_handler import ComputeTrajectories
from stl_games.stl.stl_specs import GoalDistanceSTLSpecs
from stl_games.mpc.mpc_high_level2 import MPCHighLevelPlanner
from stl_games.plot.plot_result import PlotResult
from stl_games.stl.compute_robustness import ComputeRobustness


# Set random seed for reproducibility
random.seed(2)

##### PARAMETERS #####
number_of_robots = 4
goal_size = 2.0     # Goal square size
T = 40              # Time for reaching goal
safe_dist = 3       # Safety distance from other robots
safe_dist_obs = 1   # Safety distance from obstacles
grid_size = 100
time_to_reach_goals = np.array((15, T, T, T))    # Time each robot has to reach one goal

##### LINEAR SYSTEM #####
# Define the linear system dynamics for each robot (double integrator model)
dt = 0.25   # Time step should be the same as in the MPC
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

##### ENVIRONMENT #####
# Obstacles in format (x_centre, y_centre, radius)
obstacles = [   (31, 84, 9), (67, 29, 8), 
                (52, 52, 6), (22, 41, 7),
                (77, 77, 9), (12, 90, 6), 
                (89, 11, 8), (45, 68, 9),
                (60, 35, 7), (30, 20, 5)]

# Valid start positions (not too close to obstacles or to other robots)
generate_valid_pos = GenerateValidPositions_4States()
start_positions = generate_valid_pos.generate_valid_start_positions(grid_size, number_of_robots, obstacles, safe_dist+1, safe_dist_obs+1)
print("start positions: ", start_positions)
x0 = start_positions.flatten()

# Valid goal positions (not too close to obstacles or other goals)
number_of_goals = ([2, 2, 2, 2])      # Number of goals for each robot
number_of_goals_total = 8       # Total number of goals
goal_positions, goal_list = generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, number_of_robots, number_of_goals, obstacles, safe_dist+1, safe_dist_obs+1)
print("goal positions for each robot: ", goal_list)

##### MPC #####
# Parameters
dt = 0.25
horizon_mpc = 4
nx = 4          # Number of states (x, y, vx, vy) for each robot
nu = 2          # Number of control inputs (ax, ay) for each robot
u_min = -3      # Minimum control input (acceleration)
u_max = 3       # Maximum control input (acceleration)


additional_points = int((1/dt)-1)
step_to_reach_goal = time_to_reach_goals*(1+additional_points)  # Number of steps to reach goal

# Build MPC problem
start_time = time.time()
mpc = MPCHighLevelPlanner(nx, nu, number_of_robots, horizon_mpc, dt, u_min, u_max, goal_size, obstacles, step_to_reach_goal, goal_list)
mpc.build_problem(x0)

# Solve MPC in a receiding horizon fashion
state_trajectory = [x0]
control_trajectory = []
x_current = x0
max_iters_mpc = (1+additional_points)*T
for t in range(max_iters_mpc):
    if t == 0:
        u_opt, x_prev, u_prev = mpc.solve_mpc(x_current, None, None, t)
    else:
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
print(state_trajectory.shape)
print(control_trajectory.shape)
exit()

##### RESULTS #####
print("Solver time: ", mpc.solver_time)
total_time = end_time-start_time
print("Total execution time: ", total_time)
#print("Robot1 max velocity: ", np.max(np.abs(state_trajectory[2:4,:])))
#print("Robot2 max velocity: ", np.max(np.abs(state_trajectory[6:8,:])))
# Plotting trajectory
plot = PlotResult()
plot(state_trajectory, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size)

# Analyze robustness of the resulting trajectory to STL specifications
compute_traj = ComputeTrajectories()
compute_traj.compute_y(state_trajectory, control_trajectory, number_of_robots)
y = compute_traj.y  # y = output of the system (states + inputs)
goal_spec_ = GoalDistanceSTLSpecs()
goal_spec_(goal_size, goal_positions, number_of_goals, number_of_robots, step_to_reach_goal)
goal_spec = goal_spec_.goal_distance_spec
robustness_goal_reaching = goal_spec.robustness(y, 0)
compute_robustness = ComputeRobustness()
min_dist = compute_robustness.min_distance_to_obstacles(state_trajectory, obstacles, number_of_robots)
print('robustness for reaching goal: ', robustness_goal_reaching)
print('robustness for obstacle avoidance: ', min_dist)

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