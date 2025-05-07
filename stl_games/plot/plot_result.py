import matplotlib.pyplot as plt
import numpy as np
import itertools
from stl_games.collision.collision_handler import CollisionDetection

class PlotResult:
    def __init__(self):
        pass
    def __call__(self, x, x0, goal_list, number_of_goals, battery_list, number_of_robots, obstacle_bounds_list, safe_dist):
        plt.figure(figsize=(10, 6))
        robot_id_list_associated_goals = sum([[i] * number_of_goals[i] for i in range(len(number_of_goals))], [])
        robot_positions = []
        for i in range(number_of_robots):
            robot_x=(x[i*4])
            robot_y=(x[(i*4)+1])
            x_0 = x0[i*4]
            y_0 = x0[(i*4)+1]
            robot_label = f"Robot {i+1}"
            plt.plot(robot_x, robot_y, label=robot_label, linestyle='-', marker='o')
            plt.plot(x_0, y_0, marker='o', color='black')
            robot_positions.append((robot_x, robot_y))
        plt.plot(x_0, y_0, label="start", marker='o', color='black')
        if len(goal_list)>0:
            for i in range(len(goal_list)):
                goal_label = f"Goal {(robot_id_list_associated_goals[i])+1}"
                plt.plot(goal_list[i][0], goal_list[i][1], label=goal_label, marker='X')
        if len(battery_list)>0:
            for i in range(len(battery_list)-1):
                plt.plot(battery_list[i][0], battery_list[i][1], marker='o', color='black')
            plt.plot(battery_list[-1][0], battery_list[-1][1], label="Battery", marker='o', color='black')
        if (len(obstacle_bounds_list)>0):
            for i in range(len(obstacle_bounds_list)):
                x_vertices = (obstacle_bounds_list[i][0], obstacle_bounds_list[i][1], obstacle_bounds_list[i][1], obstacle_bounds_list[i][0], obstacle_bounds_list[i][0])
                y_vertices = (obstacle_bounds_list[i][2], obstacle_bounds_list[i][2], obstacle_bounds_list[i][3], obstacle_bounds_list[i][3], obstacle_bounds_list[i][2])
                plt.plot(x_vertices, y_vertices, linestyle='-', color='red')
        collision_detection = CollisionDetection()
        collision_detection(x, number_of_robots, safe_dist)
        collision_points_x = collision_detection.collision_points_x_plot
        collision_points_y = collision_detection.collision_points_y_plot
        if len(collision_points_x)>0:
            for i in range(len(collision_points_x)-1):
                plt.scatter(collision_points_x[i], collision_points_y[i], color='red', zorder=5, marker='X')
            plt.scatter(collision_points_x[-1], collision_points_y[-1], color='red', label='Collision', zorder=5, marker='X')
        plt.title("Robot Trajectories and Collisions")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.xlim((-10,110))
        plt.ylim((-10, 110))
        plt.grid(True)
        plt.show()
        return True