import numpy as np
import itertools
import random
import time
import matplotlib.pyplot as plt
from stlpy.systems import LinearSystem
from stlpy.solvers.gurobi.gurobi_micp import GurobiMICPSolver

from stl_games.linear_system.product_dynamical_system import ProductDynamicalSystem
from stl_games.environment.generate_valid_positions import GenerateValidPositions
from stl_games.stl.stl_specs import ObstacleAvoidanceSTLSpecs, GoalDistanceSTLSpecs, CollisionAvoidanceSTLSpecs
from stl_games.collision.collision_handler import CollisionHandler
from stl_games.plot.plot_result import PlotResult
from stl_games.trajectory.trajectory_handler import ComputeAdditionalPoints, AugmentControlInput, ComputeTrajectories
from stl_games.mpc.mpc_cbf_linearized_multipleGoals import MPC_cbf_multiple_goals

# Set random seed for reproducibility
random.seed(11)
number_of_robots = 4

# Define parameters
eps = 0.5
T = 15
safe_dist = 3
safe_dist_obs = 1
grid_size = 100

# Define initial state and obstacles
obstacle_bounds_list = [
    (15, 25, 30, 40),
    (60, 70, 80, 90),
    (5, 15, 50, 60),
    (40, 50, 10, 20),
    (70, 80, 20, 30),
    (25, 35, 75, 85),
    (85, 95, 55, 65)]
generate_valid_pos = GenerateValidPositions()
generate_valid_pos.generate_valid_start_positions(grid_size, number_of_robots, obstacle_bounds_list, safe_dist+1, safe_dist_obs+1)
start_positions = generate_valid_pos.start_positions
x0 = start_positions.flatten()

number_of_goals_total = 4
number_of_goals = ([1, 1, 1, 1])
robot_id_list_associated_goals = sum([[i] * number_of_goals[i] for i in range(len(number_of_goals))], [])
generate_valid_pos = GenerateValidPositions()
generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, obstacle_bounds_list, safe_dist+1, safe_dist_obs+1)
goal_positions = generate_valid_pos.goal_positions
goal_positions_array = np.array(goal_positions)

Ad = np.array([[1, 0, 1, 0],  
                [0, 1, 0, 1],  
                [0, 0, 1, 0],  
                [0, 0, 0, 1]])
Bd = np.array([[0.5, 0],  
                [0, 0.5],  
                [1, 0],  
                [0, 1]])
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
system_a = LinearSystem(Ad, Bd, Cd, Dd)
system_b = LinearSystem(Ad, Bd, Cd, Dd)
system_c = LinearSystem(Ad, Bd, Cd, Dd)
system_d = LinearSystem(Ad, Bd, Cd, Dd)
subsystems = [system_a, system_b, system_c, system_d]
dynamics = ProductDynamicalSystem()
dynamics(subsystems)
combined_system = dynamics.combined_system

goal_spec_ = GoalDistanceSTLSpecs()
goal_spec_(eps, goal_positions_array, number_of_goals, number_of_robots, T)
goal_spec = goal_spec_.goal_distance_spec

obstacle_avoidance_spec_ = ObstacleAvoidanceSTLSpecs()
obstacle_avoidance_spec_(obstacle_bounds_list, number_of_robots, T, safe_dist_obs)
obstacle_avoidance_spec = obstacle_avoidance_spec_.obstacle_avoidance_spec

# Define STL specification for collision avoidance between robots
collision_avoidance_spec_ = CollisionAvoidanceSTLSpecs()
for robot_ids_collision_avoidance in itertools.combinations(range(number_of_robots), 2):
    collision_avoidance_spec_(safe_dist, list(robot_ids_collision_avoidance), number_of_robots, T)
    collision_avoidance_spec = collision_avoidance_spec_.collision_avoidance_spec

combined_spec = obstacle_avoidance_spec & goal_spec
combined_spec.simplify()
solver = GurobiMICPSolver(combined_spec, combined_system, x0, T)
solver.AddQuadraticCost(Q=1e-1*np.diag([0, 0, 1, 1] * (number_of_robots)), R=1e-1*np.eye(number_of_robots*2))
solver.AddControlBounds(u_min=-1, u_max=1)
solver.AddRobustnessCost()
x, u, rho, solver_time = solver.Solve()

plot = PlotResult()
plt.figure(figsize=(10, 6))
plot(x, x0, goal_positions_array, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list, safe_dist)
plt.show()

if x is None:
    print("Solver failed")
    exit()

# Modify trajectory to avoid collision with mpc receiding horizon approach
start_time = time.time()
collision_handler = CollisionHandler()
modified_trajectory_length = 4
delta_t = modified_trajectory_length // 2
collision_handler(x, u, number_of_robots, safe_dist, delta_t)
trajectory_to_be_modified_list = collision_handler.trajectory_to_be_modified
while (len(trajectory_to_be_modified_list)>0):
    trajectory_to_be_modified = collision_handler.trajectory_to_be_modified[0]
    input_to_be_modified = collision_handler.input_to_be_modified[0]
    time_of_collision = collision_handler.collision_times[0]
    trajectory_to_be_modified = np.array(trajectory_to_be_modified)
    input_to_be_modified = np.array(input_to_be_modified)
    collision_indices = collision_handler.collision_indices
    print("Time of collision: ", time_of_collision)
    dt = 0.25
    additional_points = int((1/dt)-1)
    compute_additional_points = ComputeAdditionalPoints()
    compute_additional_points(trajectory_to_be_modified, input_to_be_modified, additional_points) 
    augmented_trajectory = compute_additional_points.augmented_trajectory
    augmented_input = compute_additional_points.augmented_input

    horizon_mpc = 10
    if horizon_mpc > np.size((augmented_trajectory),axis=1)-1:
        horizon_mpc = np.size((augmented_trajectory),axis=1)-1

    nx = np.size(augmented_trajectory, axis=0)
    nu = np.size(augmented_input, axis=0)
    u_min = -5
    u_max = 5
    position_tolerance = eps
    velocity_tolerance = 0.1
    step_to_reach_goal = np.size((augmented_trajectory),axis=1)-1
    x_prev = augmented_trajectory[:,:horizon_mpc+1]
    u_prev = augmented_input[:,:horizon_mpc]
    x0_mpc = augmented_trajectory[:,0]
    xG_mpc = np.zeros(8)
    xG1_mpc = list()
    xG2_mpc = list()
    for j in range(len(robot_id_list_associated_goals)):
        if robot_id_list_associated_goals[j]==collision_indices[0][0]:
            xG1_mpc.append(goal_positions[j])
        if robot_id_list_associated_goals[j]==collision_indices[0][1]:
            xG2_mpc.append(goal_positions[j])

    xG1_mpc = np.array(xG1_mpc)
    xG2_mpc = np.array(xG2_mpc)
    
    mpc = MPC_cbf_multiple_goals(nx, nu, horizon_mpc, dt, u_min, u_max, safe_dist, position_tolerance, obstacle_bounds_list, step_to_reach_goal, xG1_mpc, xG2_mpc)
    mpc.build_problem(x0_mpc, xG1_mpc, xG2_mpc)

    state_trajectory = [x0_mpc]
    control_trajectory = []
    x_current = x0_mpc
    max_iters_mpc = np.size((augmented_trajectory),axis=1)
    print("Max iterations: ", max_iters_mpc)
    #print("dt: ", dt)
    #print("additional points: ", additional_points)
    for t in range(max_iters_mpc):
        u_opt, x_prev, u_prev = mpc.solve_mpc(x_current, xG1_mpc, xG2_mpc, x_prev, u_prev, t)
        if u_opt is None:
            print("Solver failed")
            exit()
        x_current = mpc.define_dynamics(x_current, u_opt)
        state_trajectory.append(x_current)
        control_trajectory.append(u_opt)
    state_trajectory = np.array(state_trajectory)
    control_trajectory = np.array(control_trajectory)
    end_time = time.time()
    print("Solver time: ", mpc.solver_time)
    print("Robot1 velocity: ", np.max(np.abs(state_trajectory[:,2:4])))
    print("Robot2 velocity: ", np.max(np.abs(state_trajectory[:,6:8])))
    plt.figure(figsize=(10, 6))
    plot(x, x0, goal_positions, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list, safe_dist)
    plot(state_trajectory.T, x0_mpc, goal_list=[xG_mpc[0:2], xG_mpc[4:6]], robot_id_list_associated_goals=[0, 1], battery_list=[], number_of_robots=2, obstacle_bounds_list=[], safe_dist=safe_dist)
    plt.show()

    if (np.size((u), axis=1))==T+1:
        augment_control_inputs = AugmentControlInput()
        augment_control_inputs(u, additional_points)
        augmented_u = augment_control_inputs.augmented_input
    elif (np.size((u), axis=1))==(T*(additional_points+1)+1):
        augmented_u=u
    else:
        print("error in u dimensions")
        exit()
    augmented_u[(collision_indices[0][0]*2):(collision_indices[0][0]*2+2),(additional_points+1)*(time_of_collision-delta_t):] = (control_trajectory.T)[0:2,:]
    augmented_u[(collision_indices[0][1]*2):(collision_indices[0][1]*2+2),(additional_points+1)*(time_of_collision-delta_t):] = (control_trajectory.T)[2:4,:]
    compute_traj = ComputeTrajectories()
    compute_traj.compute_x(combined_system, number_of_robots, dt, augmented_u[:,:-1], x0)
    augmented_x = compute_traj.x
    compute_traj.compute_y(augmented_x, augmented_u, number_of_robots)
    augmented_y = compute_traj.y

    plt.figure(figsize=(10, 6))
    plot(augmented_x, x0, goal_positions, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list, safe_dist)
    plt.show()

    # Check robustness of augmented state trajectory (taking one element each 10)
    x_to_check_robustness = augmented_x[:, ::(additional_points+1)]
    num_groups = T
    reshaped_matrix = augmented_u[:, :-1].reshape(8, num_groups, additional_points+1)
    averaged_matrix = reshaped_matrix.mean(axis=2)
    last_element = augmented_u[:, -1].reshape(8, 1)
    u_to_check_robustness = np.concatenate((averaged_matrix, last_element), axis=1)
    compute_traj.compute_y(x_to_check_robustness, u_to_check_robustness, number_of_robots)
    y_to_check_robustness = compute_traj.y
    obstacle_avoidance_spec_final_ = ObstacleAvoidanceSTLSpecs()
    obstacle_avoidance_spec_final_(obstacle_bounds_list, number_of_robots, T, eps)
    obstacle_avoidance_spec_final = obstacle_avoidance_spec_final_.obstacle_avoidance_spec
    robustness_goal = goal_spec.robustness(y_to_check_robustness, 0)
    robustness_obs = obstacle_avoidance_spec_final.robustness(y_to_check_robustness, 0)
    collision_avoidance_spec_list = []
    for robot_ids_collision_avoidance in itertools.combinations(range(number_of_robots), 2):
        collision_avoidance_spec_(safe_dist, list(robot_ids_collision_avoidance), number_of_robots, T)
        spec = collision_avoidance_spec_.collision_avoidance_spec
        collision_avoidance_spec_list.append(spec)
    collision_avoidance_spec = collision_avoidance_spec_list[0]
    for stl_spec in collision_avoidance_spec_list[1:]:
        collision_avoidance_spec = collision_avoidance_spec & stl_spec
    collision_avoidance_spec.simplify()
    robustness_collision = collision_avoidance_spec.robustness(y_to_check_robustness, 0)
    robustness = min(robustness_goal, robustness_obs, robustness_collision)
    print('robustness: ', robustness)
    print('goal robustness: ', robustness_goal)
    print('obstacle robustness: ', robustness_obs)
    print('collision robustness: ', robustness_collision)
    plt.figure(figsize=(10, 6))
    plot(x_to_check_robustness, x0, goal_positions, robot_id_list_associated_goals, [], number_of_robots, obstacle_bounds_list, safe_dist)
    plt.show()

    # Update the trajectory to be modified 
    collision_handler(x_to_check_robustness, u_to_check_robustness, number_of_robots, safe_dist, delta_t)
    if collision_handler.collision_detected:
        trajectory_to_be_modified_list = collision_handler.trajectory_to_be_modified
        x = augmented_x
        u = augmented_u
    else:
        trajectory_to_be_modified_list = []
    
total_time = end_time-start_time
print("Total execution time: ", total_time)