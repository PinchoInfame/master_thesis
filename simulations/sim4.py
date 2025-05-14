import numpy as np
import random
import time
import itertools
from stlpy.systems import LinearSystem

from stl_games.environment.generate_valid_positions import GenerateValidPositions_4States
from stl_games.trajectory.trajectory_handler import ComputeTrajectories
from stl_games.stl.stl_specs import GoalDistanceSTLSpecs, CollisionAvoidanceSTLSpecs
from stl_games.mpc.mpc_high_level import MPCHighLevelPlanner
from stl_games.plot.plot_result import PlotResult
from stl_games.stl.compute_robustness import ComputeRobustness

from stl_games.collision.collision_handler import CollisionHandler
from stl_games.mpc.mpc_collision_avoidance import MPCCollisionAvoidance


# Set random seed for reproducibility
random.seed(2)

##### PARAMETERS #####
number_of_robots = 6
goal_size = 2.0     # Goal square size
T = 30              # Time for reaching goal
safe_dist = 3       # Safety distance from other robots
safe_dist_obs = 1   # Safety distance from obstacles
grid_size = 100
time_to_reach_goals = np.array((T, T, T, T, T, T))                       # Time each robot has to reach one goal


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
obstacles = [
    (20, 35, 7),
    (65, 85, 7),
    (10, 55, 7),
    (45, 15, 7),
    (75, 25, 7),
    (30, 80, 7),
    (90, 60, 7)]

# Valid start positions (not too close to obstacles or to other robots)
generate_valid_pos = GenerateValidPositions_4States()
start_positions = generate_valid_pos.generate_valid_start_positions(grid_size, number_of_robots, obstacles, safe_dist+1, safe_dist_obs+1)
print("start positions: ", start_positions)
x0 = start_positions.flatten()

# Valid goal positions (not too close to obstacles or other goals)
number_of_goals = ([2, 2, 2, 2, 2, 2])      # Number of goals for each robot
number_of_goals_total = 12       # Total number of goals
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

##### RESULTS (before collision avoidance) ##### 
#print("Robot1 max velocity: ", np.max(np.abs(state_trajectory[2:4,:])))
#print("Robot2 max velocity: ", np.max(np.abs(state_trajectory[6:8,:])))
# Plotting trajectory
plot = PlotResult()
plot.plot_sim(state_trajectory, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size)

# Analyze robustness of the resulting trajectory to STL specifications
compute_traj = ComputeTrajectories()
y = compute_traj.compute_y_concatenate(state_trajectory, control_trajectory, number_of_robots) # y = output of the system (states + inputs)
goal_spec_handler = GoalDistanceSTLSpecs()
goal_spec = goal_spec_handler.compute_stl_spec_square(goal_size, goal_positions, number_of_goals, number_of_robots, step_to_reach_goal)
robustness_goal_reaching = goal_spec.robustness(y, 0)
compute_robustness = ComputeRobustness()
min_dist_obs = compute_robustness.min_distance_to_obstacles(state_trajectory, obstacles, number_of_robots)
print('robustness for reaching goal (before collision avoidance): ', robustness_goal_reaching)
print('robustness for obstacle avoidance (before collision avoidance): ', min_dist_obs)

##### COLLISION AVOIDANCE #####
# Check for collisions
step_before_collision = 10
collision_handler = CollisionHandler()
collision_detected, trajectories_to_be_modified, inputs_to_be_modified, collision_times, collision_indices = collision_handler.handle_collision(state_trajectory, control_trajectory,number_of_robots, safe_dist, step_before_collision)
trajectory_to_be_modified = trajectories_to_be_modified[0]
input_to_be_modified = inputs_to_be_modified[0]

# While collision is detected, modify the trajectory
while (collision_detected==True):
    trajectory_to_be_modified = trajectories_to_be_modified[0]
    input_to_be_modified = inputs_to_be_modified[0]
    step_of_collision = collision_times[0]
    
    x0_coll = trajectory_to_be_modified[:,0]
    u0_new = input_to_be_modified[:,0]
    horizon_mpc_new = 8
    dt_new = 0.25
    x_prev = trajectory_to_be_modified[:,0:horizon_mpc_new]
    u_prev = input_to_be_modified[:,0:horizon_mpc_new]
    
    step_to_reach_goal_new = step_to_reach_goal - (step_of_collision - step_before_collision) - 1
    goal_list_new = []
    for i in collision_indices[0]:
        goal_list_new.append(goal_list[i])

    mpc_new = MPCCollisionAvoidance(nx, nu, 2, horizon_mpc_new, dt_new, u_min, u_max, goal_size, obstacles, step_to_reach_goal_new, goal_list_new, safe_dist)
    state_trajectory_new = [x0_coll]
    control_trajectory_new = []
    x_current_new = x0_coll
    max_iters_mpc_new = np.max(step_to_reach_goal_new) + 1
    mpc_new.build_problem(x0_coll)
    for t in range(max_iters_mpc_new):
        if t == 0:
            u_opt_new, x_prev_new, u_prev_new = mpc_new.solve_mpc(x_current_new, None, None, t)
        else:
            u_opt_new, x_prev_new, u_prev_new = mpc_new.solve_mpc(x_current_new, x_prev_new, u_prev_new, t)
        if u_opt_new is None:
            print("Solver failed")
            exit()
        x_current_new = mpc_new.define_dynamics(x_current_new, u_opt_new)
        state_trajectory_new.append(x_current_new)
        control_trajectory_new.append(u_opt_new)

    state_trajectory_new = np.array(state_trajectory_new).T
    control_trajectory_new = np.array(control_trajectory_new).T
    # Update the state and control trajectories
    for j, i in enumerate(collision_indices[0]):
        state_trajectory[i*4:i*4+4, step_of_collision-step_before_collision:] = state_trajectory_new[j*4:j*4+4, :]
        control_trajectory[i*2:i*2+2, step_of_collision-step_before_collision:] = control_trajectory_new[j*2:j*2+2, :]
    # Check for collisions again 
    collision_detected, trajectories_to_be_modified, input_to_be_modified, collision_times, collision_indices = collision_handler.handle_collision(state_trajectory, control_trajectory,number_of_robots, safe_dist, step_before_collision)

end_time = time.time()

##### RESULTS (after collision avoidance) #####
total_time = end_time-start_time
print("Total execution time: ", total_time)
# Plotting trajectory
plot = PlotResult()
plot.plot_sim(state_trajectory, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size)

# Analyze robustness of the resulting trajectory to STL specifications (after collision avoidance)
y = compute_traj.compute_y_concatenate(state_trajectory, control_trajectory, number_of_robots) # y = output of the system (states + inputs)
goal_spec = goal_spec_handler.compute_stl_spec_square(goal_size, goal_positions, number_of_goals, number_of_robots, step_to_reach_goal)
robustness_goal_reaching = goal_spec.robustness(y, 0)
min_dist_obs = compute_robustness.min_distance_to_obstacles(state_trajectory, obstacles, number_of_robots)
min_dist_agents = compute_robustness.min_distance_agents(state_trajectory, number_of_robots)
print('robustness for reaching goal (after collision avoidance): ', robustness_goal_reaching)
print('robustness for obstacle avoidance (after collision avoidance): ', min_dist_obs)
print('robustness collision avoidance: ', min_dist_agents)