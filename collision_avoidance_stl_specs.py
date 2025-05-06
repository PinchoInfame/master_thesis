import numpy as np
from stlpy.STL import LinearPredicate
class CollisionAvoidanceSTLSpecs:
    def __init__(self):
        self.collision_avoidance_spec = None
    def __call__(self, safe_dist, robot_ids, number_of_robots, T):
        if robot_ids == None:
            a_pred = np.zeros((1, number_of_robots*6))
            b_pred = 0
            pred = LinearPredicate(a_pred, b_pred)
            self.collision_avoidance_spec = pred
            return -1
        a_pred1 = np.zeros((1, number_of_robots*6))
        a_pred1[0,0+robot_ids[0]*6]=1
        a_pred1[0,0+robot_ids[1]*6]=-1
        b_pred1 = safe_dist
        
        a_pred2 = np.zeros((1, number_of_robots*6))
        a_pred2[0,0+robot_ids[0]*6]=-1
        a_pred2[0,0+robot_ids[1]*6]=1
        b_pred2 = safe_dist

        a_pred3 = np.zeros((1, number_of_robots*6))
        a_pred3[0,1+robot_ids[0]*6]=1
        a_pred3[0,1+robot_ids[1]*6]=-1
        b_pred3 = safe_dist

        a_pred4 = np.zeros((1, number_of_robots*6))
        a_pred4[0,1+robot_ids[0]*6]=-1
        a_pred4[0,1+robot_ids[1]*6]=1
        b_pred4 = safe_dist

        predicate1 = LinearPredicate(a_pred1, b_pred1)
        predicate2 = LinearPredicate(a_pred2, b_pred2)
        predicate3 = LinearPredicate(a_pred3, b_pred3)
        predicate4 = LinearPredicate(a_pred4, b_pred4)
        combined_predicate = predicate1 | predicate2 | predicate3 | predicate4
        self.collision_avoidance_spec = combined_predicate.always(0, T)
        self.collision_avoidance_spec.simplify()