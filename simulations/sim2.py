import numpy as np
import random
import time
from scipy.integrate import solve_ivp

from stl_games.linear_system.continuous_time_dynamics import continuous_time_dynamics
from stl_games.environment.generate_valid_positions import GenerateValidPositions_4States
from stl_games.trajectory.trajectory_handler import ComputeTrajectories
from stl_games.stl.stl_specs import GoalDistanceSTLSpecs
from stl_games.mpc.mpc_high_level import MPCHighLevelPlanner
from stl_games.plot.plot_result import PlotResult
from stl_games.stl.compute_robustness import ComputeRobustness
from stl_games.collision.collision_handler import CollisionHandler
from stl_games.mpc.mpc_collision_avoidance import MPCCollisionAvoidance

# Set random seed for reproducibility
random.seed(0)

##### PARAMETERS #####
number_of_robots = 2
goal_size = 4.0     # Goal square size
T = 5              # Time for reaching goal
safe_dist = 3       # Safety distance from other robots
safe_dist_obs = 1   # Safety distance from obstacles
grid_size = 100
time_to_reach_goals = np.array((T, T))                       # Time each robot has to reach one goal


##### LINEAR SYSTEM #####
# Define the linear system dynamics for each robot (double integrator model)
system_dynamics = continuous_time_dynamics

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
number_of_goals = ([2, 2])      # Number of goals for each robot
number_of_goals_total = 4       # Total number of goals
goal_positions, goal_list = generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, number_of_robots, number_of_goals, obstacles, safe_dist+1, safe_dist_obs+1)
print("goal positions for each robot: ", goal_list)

##### MPC #####
# Parameters
dt = 0.1
horizon_mpc = 4
nx = 4          # Number of states (x, y, vx, vy) for each robot
nu = 2          # Number of control inputs (ax, ay) for each robot


additional_points = int((1/dt)-1)
step_to_reach_goal = time_to_reach_goals*(1+additional_points)  # Number of steps to reach goal

# Build MPC problem
start_time = time.time()
mpc = MPCHighLevelPlanner(nx, nu, number_of_robots, horizon_mpc, dt, goal_size, obstacles, step_to_reach_goal, goal_list)
mpc.build_problem(x0)

# Solve MPC in a receiding horizon fashion
state_trajectory = [x0]
control_trajectory = []
dense_state_trajectory = [x0]   # First state
dense_time = [0.0]              # Initial time
t_total = 0.0
dense_steps = 10                # Number of points between each control step
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
    
    t_dense = np.linspace(0, dt, dense_steps)
    f = lambda t, x: system_dynamics(t, x, u_opt, number_of_robots)
    sol = solve_ivp(f, [0, dt], x_current, t_eval=t_dense)
    x_current = sol.y[:, -1]
    state_trajectory.append(x_current)
    control_trajectory.append(u_opt)
    dense_time.extend([t_total + t_i for t_i in sol.t[1:]])           # corrected
    dense_state_trajectory.extend(sol.y[:, 1:].T.tolist())
    t_total += dt

state_trajectory = np.array(state_trajectory).T
control_trajectory = np.array(control_trajectory).T
dense_state_array = np.array(dense_state_trajectory).T
dense_control_array = np.zeros_like(dense_state_array)
end_time = time.time()

##### RESULTS #####
print("Solver time: ", mpc.solver_time)
total_time = end_time-start_time
print("Total execution time: ", total_time)
# Plotting trajectory
plot = PlotResult()
plot.plot_sim(state_trajectory, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size, dense_state_array=dense_state_array)

# Analyze robustness of the resulting trajectory to STL specifications
compute_traj = ComputeTrajectories()
y = compute_traj.compute_y_concatenate(dense_state_array, dense_control_array, number_of_robots) # y = output of the system (states + inputs)
goal_spec_handler = GoalDistanceSTLSpecs()
step_to_reach_goal_dense = time_to_reach_goals*(1+additional_points)*(dense_steps-1)
goal_spec = goal_spec_handler.compute_stl_spec_square(goal_size, goal_positions, number_of_goals, number_of_robots, step_to_reach_goal_dense)
robustness_goal_reaching = goal_spec.robustness(y, 0)
compute_robustness = ComputeRobustness()
min_dist = compute_robustness.min_distance_to_obstacles(dense_state_array, obstacles, number_of_robots)
min_dist_agents = compute_robustness.min_distance_agents(dense_state_array, number_of_robots)
print('robustness for reaching goal: ', robustness_goal_reaching)
print('robustness for obstacle avoidance: ', min_dist)
print('robustness for collision avoidance: ', min_dist_agents)

##### COLLISION AVOIDANCE #####
# Check for collisions
step_before_collision = 3
collision_handler = CollisionHandler()
collision_detected, trajectories_to_be_modified, inputs_to_be_modified, collision_times, collision_indices = collision_handler.handle_collision(state_trajectory, control_trajectory,number_of_robots, safe_dist, step_before_collision)
if not collision_detected:
    print("No collision detected!")
    exit()
trajectory_to_be_modified = trajectories_to_be_modified[0]
input_to_be_modified = inputs_to_be_modified[0]

# While collision is detected, modify the trajectory
while (collision_detected==True):
    trajectory_to_be_modified = trajectories_to_be_modified[0]
    input_to_be_modified = inputs_to_be_modified[0]
    step_of_collision = collision_times[0]
    
    x0_coll = trajectory_to_be_modified[:,0]
    u0_new = input_to_be_modified[:,0]
    horizon_mpc_new = 4
    dt_new = 0.1
    
    step_to_reach_goal_new = step_to_reach_goal - (step_of_collision - step_before_collision) - 1
    goal_list_new = []
    for i in collision_indices[0]:
        goal_list_new.append(goal_list[i])

    mpc_new = MPCCollisionAvoidance(nx, nu, 2, horizon_mpc_new, dt_new, goal_size, obstacles, step_to_reach_goal_new, goal_list_new, safe_dist)
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
        t_dense = np.linspace(0, dt_new, dense_steps)
        f = lambda t, x: system_dynamics(t, x, u_opt_new, 2)
        sol = solve_ivp(f, [0, dt_new], x_current_new, t_eval=t_dense)
        x_current_new = sol.y[:, -1]
        #x_current_new = mpc_new.define_dynamics(x_current_new, u_opt_new)
        state_trajectory_new.append(x_current_new)
        control_trajectory_new.append(u_opt_new)

    state_trajectory_new = np.array(state_trajectory_new).T
    control_trajectory_new = np.array(control_trajectory_new).T
    # Update the state and control trajectories
    for j, i in enumerate(collision_indices[0]):
        if state_trajectory[i*4:i*4+4, step_of_collision-step_before_collision:].shape != state_trajectory_new[j*4:j*4+4, :].shape:
            print("Choose a different step before collision!")
            exit()
        state_trajectory[i*4:i*4+4, step_of_collision-step_before_collision:] = state_trajectory_new[j*4:j*4+4, :]
        control_trajectory[i*2:i*2+2, step_of_collision-step_before_collision:] = control_trajectory_new[j*2:j*2+2, :]
    # Check for collisions again 
    collision_detected, trajectories_to_be_modified, input_to_be_modified, collision_times, collision_indices = collision_handler.handle_collision(state_trajectory, control_trajectory,number_of_robots, safe_dist, step_before_collision)
    #plot.plot_sim(state_trajectory, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size)


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
print(y.shape)
print(step_to_reach_goal)
robustness_goal_reaching = goal_spec.robustness(y, 0)
min_dist_obs = compute_robustness.min_distance_to_obstacles(state_trajectory, obstacles, number_of_robots)
min_dist_agents = compute_robustness.min_distance_agents(state_trajectory, number_of_robots)
print('robustness for reaching goal (after collision avoidance): ', robustness_goal_reaching)
print('robustness for obstacle avoidance (after collision avoidance): ', min_dist_obs)
print('robustness collision avoidance: ', min_dist_agents)