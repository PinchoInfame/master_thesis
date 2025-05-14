import numpy as np
import random
import time
import itertools
from stlpy.systems import LinearSystem

from stl_games.environment.generate_valid_positions import GenerateValidPositions_4States
from stl_games.stl.stl_specs import GoalDistanceSTLSpecs, ObstacleAvoidanceSTLSpecs, CollisionAvoidanceSTLSpecs
from stl_games.plot.plot_result import PlotResult
from stlpy.solvers.gurobi.gurobi_micp import GurobiMICPSolver




# Set random seed for reproducibility
random.seed(2)

##### PARAMETERS #####
number_of_robots = 3
goal_size = 2.0     # Goal square size
T = 30              # Time for reaching goal
safe_dist = 3       # Safety distance from other robots
safe_dist_obs = 1   # Safety distance from obstacles
grid_size = 100
time_to_reach_goals = np.array((T, T, T))                       # Time each robot has to reach one goal


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
number_of_goals = ([2, 2, 2])      # Number of goals for each robot
number_of_goals_total = 6       # Total number of goals
goal_positions, goal_list = generate_valid_pos.generate_valid_goal_positions(grid_size, number_of_goals_total, number_of_robots, number_of_goals, obstacles, safe_dist+1, safe_dist_obs+1)
print("goal positions for each robot: ", goal_list)

##### STL SPECIFICATIONS #####
# STL specs for goal reaching and obstacle avoidance
goal_spec_handler = GoalDistanceSTLSpecs()
goal_spec = goal_spec_handler.compute_stl_spec_square(goal_size, goal_positions, number_of_goals, number_of_robots, time_to_reach_goals)
# STL specs for obstacle avoidance
obs_spec_handler = ObstacleAvoidanceSTLSpecs()
obs_spec = obs_spec_handler.compute_stl_obs_spec(obstacles, number_of_robots, T, goal_size)

coll_spec_handler = CollisionAvoidanceSTLSpecs()
coll_spec_list = []
for robot_ids in itertools.combinations(range(number_of_robots), 2):
    coll_spec_handler(safe_dist, list(robot_ids), number_of_robots, T)
    coll_spec_list.append(coll_spec_handler.collision_avoidance_spec)
collision_avoidance_spec = coll_spec_list[0]
for pred in coll_spec_list[1:]:
    collision_avoidance_spec = collision_avoidance_spec & pred
collision_avoidance_spec.simplify()

# Combine the STL specifications, create the solver and solve the problem
combined_spec = goal_spec & obs_spec & collision_avoidance_spec
solver = GurobiMICPSolver(combined_spec, combined_system, x0, T)
solver.AddQuadraticCost(Q=1e-1*np.diag([0, 0, 1, 1] * (number_of_robots)), R=1e-1*np.eye(number_of_robots*2))
#solver.AddControlBounds(u_min=-3, u_max=3)
solver.AddRobustnessCost()
x, u, rho, solver_time = solver.Solve()
plot = PlotResult()
plot.plot_sim(x, x0, goal_positions, number_of_goals, number_of_robots, obstacles, safe_dist, goal_size, grid_size)
