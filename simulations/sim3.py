import numpy as np
import random
import time
from scipy.integrate import solve_ivp

from stl_games.linear_system.continuous_time_dynamics import double_integrator
from stl_games.environment.generate_obstacles import generate_obstacles
from stl_games.environment.generate_valid_positions import GenerateValidPositions_4States
from stl_games.trajectory.trajectory_handler import ComputeTrajectories
from stl_games.stl.stl_specs import GoalDistanceSTLSpecs
from stl_games.mpc.mpc_collision_avoidance import MPCCentralized
from stl_games.plot.plot_result import PlotResult
from stl_games.stl.compute_robustness import ComputeRobustness
from stl_games.collision.detect_collision import detect_collision
from stl_games.mpc.mpc_player1 import MPCPlayer1
from stl_games.mpc.mpc_player2 import MPCPlayer2

# Set random seed for reproducibility
random.seed()

for i in range(10):

    ##### PARAMETERS #####
    number_of_robots = 5
    goal_size = 4.0     # Goal square size
    T = 20              # Time for reaching goal
    safe_dist = 3       # Safety distance from other robots
    safe_dist_obs = 1   # Safety distance from obstacles
    grid_size = 100
    time_to_reach_goals = np.array((T, T, T, T, T))                       # Time each robot has to reach one goal


    ##### LINEAR SYSTEM #####
    # Define the linear system dynamics for each robot (double integrator model)
    system_dynamics = double_integrator

    ##### ENVIRONMENT #####
    # Obstacles in format (x_centre, y_centre, radius)
    number_of_obstacles = 10
    max_radius = 10
    obstacles = generate_obstacles(number_of_obstacles, grid_size, max_radius)

    # Valid start positions (not too close to obstacles or to other robots)
    generate_valid_pos = GenerateValidPositions_4States()
    start_positions = generate_valid_pos.generate_valid_start_positions(grid_size, number_of_robots, obstacles, safe_dist+(goal_size*2), safe_dist_obs+(goal_size*2))
    #print("start positions: ", start_positions)
    x0 = start_positions.flatten()

    # Valid goal positions (not too close to obstacles or other goals)
    number_of_goals = ([2, 2, 2, 2, 2])      # Number of goals for each robot
    number_of_goals_total = 10       # Total number of goals
    goal_positions, goal_list = generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, number_of_robots, number_of_goals, obstacles, safe_dist+(goal_size*2), safe_dist_obs+(goal_size*2))
    #print("goal positions for each robot: ", goal_list)

    ##### MPC #####
    # Parameters
    dt = 0.25
    horizon_mpc = 4
    nx = 4          # Number of states (x, y, vx, vy) for each robot
    nu = 2          # Number of control inputs (ax, ay) for each robot


    additional_points = int((1/dt)-1)
    step_to_reach_goal = time_to_reach_goals*(1+additional_points)  # Number of steps to reach goal

    # Build MPC problem
    start_time = time.time()
    mpc = MPCCentralized(nx, nu, number_of_robots, horizon_mpc, dt, goal_size, obstacles, step_to_reach_goal, goal_list, safe_dist)
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
    #plot.plot_sim(state_trajectory, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size, dense_state_array=dense_state_array)

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

    with open("simulation_results.txt", "a") as file:
        file.write(f"Rg: {float(robustness_goal_reaching[0]):.2f}, Robs: {float(min_dist):.2f}, Rcoll: {float(min_dist_agents):.2f}, Time: {float(total_time)}\n")