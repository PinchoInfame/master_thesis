import itertools
import numpy as np
from stl_games.collision_detection import CollisionDetection


class CollisionHandler:
    def __init__(self):
        self.trajectory_to_be_modified=[]
        self.input_to_be_modified = []
        self.collision_times = []
        self.collision_detected = False
        self.collision_indices = []
    def __call__(self, x, u, number_of_robots, safe_dist, delta_t):
        self.trajectory_to_be_modified=[]
        self.input_to_be_modified = []
        self.collision_times = []
        self.collision_detected = False
        self.collision_indices = []
        collision_detection = CollisionDetection()
        collision_detection(x, number_of_robots, safe_dist)
        if collision_detection.collision_detected:
            print("Collision detected")
            self.collision_detected = True
            collisions = collision_detection.collision_times
            for (ind1, ind2, t) in collisions:
                if t>=2:
                    self.trajectory_to_be_modified.append(np.concatenate((x[(ind1*4):(ind1*4+4), (t-delta_t):], x[(ind2*4):(ind2*4+4), (t-delta_t):]), axis=0))
                    self.input_to_be_modified.append(np.concatenate((u[(ind1*2):(ind1*2+2), t-delta_t:], u[(ind2*2):(ind2*2+2), t-delta_t:]), axis=0))
                    self.collision_times.append(t)
                    self.collision_indices.append((ind1, ind2))
            #self.trajectory_to_be_modified = np.array(self.trajectory_to_be_modified)
            #self.input_to_be_modified = np.array(self.input_to_be_modified)
            #self.collision_times = np.array(self.collision_times)
        else:
            print("No collision detected")
            self.collision_detected = False
            self.trajectory_to_be_modified = []
            self.input_to_be_modified = []