import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import itertools

from stl_games.collision.collision_handler import CollisionHandler

class PlotResult:
    """
    Class to plot the results of the simulation, including robot trajectories, goals, and obstacles.
    """
    def __init__(self):
        pass
    def plot_sim(self, x: np.ndarray, x0: np.ndarray, goal_list: list[tuple[float, float]], number_of_goals: list[int], number_of_robots: int, obstacles: list[tuple[float, float, float]], safe_dist: float, goal_size: float, grid_size: int, battery_list: list[tuple[float, float]] =[], dense_state_array: np.ndarray = None):
        '''
        Plot the results of the simulation.
        
        :param x: State trajectory of the robots (array of shape (nx*number_of_robots, number_of_steps)).
        :param x0: Initial state of the robots (flattened).
        :param goal_list: List of goal positions (ordered as indicated by number_of_goals).
        :param number_of_goals: Number of goals for each robot.
        :param number_of_robots: Number of robots.
        :param obstacles: List of obstacles in the format [(x_centre, y_centre, radius), ...].
        :param safe_dist: Safety distance between robots.
        :param goal_size: Size of the goal area (square).
        :param grid_size: Size of the grid (assumed square).
        :param battery_list: List of battery positions (optional).
        '''
        plt.figure(figsize=(10, 6))
        robot_id_list_associated_goals = sum([[i] * number_of_goals[i] for i in range(len(number_of_goals))], [])
        robot_positions = []
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i in range(number_of_robots):
            x_0 = x0[i*4]
            y_0 = x0[(i*4)+1]
            plt.plot(x_0, y_0, marker='o', color='black')
        plt.plot(x_0, y_0, label="start", marker='o', color='black')
        if dense_state_array is not None:
            self.plot_dense_trajectory(dense_state_array, number_of_robots)
        else:
            for i in range(number_of_robots):
                robot_x=(x[i*4])
                robot_y=(x[(i*4)+1])
                robot_label = f"Robot {i+1}"
                plt.plot(robot_x, robot_y, label=robot_label, linestyle='-')
                robot_positions.append((robot_x, robot_y))
        '''
        if len(goal_list)>0:
            for i in range(len(goal_list)):
                goal_label = f"Goal {(robot_id_list_associated_goals[i])+1}"
                plt.plot(goal_list[i][0], goal_list[i][1], label=goal_label, marker='X')'''
        if len(goal_list) > 0:
            for i in range(len(goal_list)):
                goal_x = goal_list[i][0]
                goal_y = goal_list[i][1]
                eps = goal_size*2  # Define the size of the square
                # Create a rectangle centered at (goal_x, goal_y) with the given size
                # Rectangle(xy, width, height)
                rect = Rectangle(
                    (goal_x - eps / 2, goal_y - eps / 2),  # Bottom-left corner of the square
                    eps, eps,  # Width and height of the square
                    linewidth=0.1,
                    edgecolor='black',
                    facecolor=colors[robot_id_list_associated_goals[i]%len(colors)],  # Fill with green
                    alpha=0.5,  # Transparency, you can adjust this
                    label = f"Goal {(robot_id_list_associated_goals[i])+1}"
                    )

                # Add the rectangle to the plot
                plt.gca().add_patch(rect)
        if len(battery_list)>0:
            for i in range(len(battery_list)-1):
                plt.plot(battery_list[i][0], battery_list[i][1], marker='o', color='black')
            plt.plot(battery_list[-1][0], battery_list[-1][1], label="Battery", marker='o', color='black')

        obstacle_legend_added = False
        if len(obstacles) > 0:
            for obs in obstacles:
                # Compute center of the square
                cx = obs[0]
                cy = obs[1]
                # Compute radius to enclose the square (half the diagonal)
                radius = obs[2]
                # Plot the enclosing circle
                circle = Circle((cx, cy), radius, color='red', fill=False)
                if not obstacle_legend_added:
                    circle.set_label('Obstacle Area')
                    obstacle_legend_added = True
                plt.gca().add_patch(circle)
        collision_handler = CollisionHandler()
        _, _, collision_points_x, collision_points_y = collision_handler.detect_collision(x, number_of_robots, safe_dist)
        if len(collision_points_x)>0:
            for i in range(len(collision_points_x)-1):
                plt.scatter(collision_points_x[i], collision_points_y[i], color='red', zorder=5, marker='X')
            plt.scatter(collision_points_x[-1], collision_points_y[-1], color='red', label='Collision', zorder=5, marker='X')
        plt.title("Robot Trajectories and Collisions")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(loc='upper right')
        plt.xlim((-grid_size/10,grid_size+grid_size/10))
        plt.ylim((-grid_size/10,grid_size+grid_size/10))
        plt.grid(True)
        plt.show()
        return True
    
    def plot_dense_trajectory(self, dense_state_array: np.ndarray, number_of_robots: int):
        """
        Plot the dense trajectory of the robots.
        
        :param dense_state_array: Dense state array of shape (time_steps, nx * N).
        :param number_of_robots: Number of robots.
        """
        for i in range(number_of_robots):
            # Indices for this robot in the flat state vector
            idx_x = 4 * i
            idx_y = 4 * i + 1

            x_pos = dense_state_array[idx_x, :]
            y_pos = dense_state_array[idx_y, :]

            plt.plot(x_pos, y_pos, label=f'Robot {i+1}')