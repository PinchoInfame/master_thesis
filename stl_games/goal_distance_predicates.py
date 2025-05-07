import numpy as np
from stlpy.STL import LinearPredicate
class GoalDistancePredicates:
    def __init__(self):
        self.predicates = None
    def __call__(self, eps, goals_list, robot_id, number_of_robots, T):
        self.predicates = []
        for i in range(len(goals_list)):
            px_goal = goals_list[i][0]
            py_goal = goals_list[i][1]

            a_pred1 = np.zeros((1, number_of_robots*6))
            a_pred1[0,0+robot_id*6]=-1
            b_pred1 = -px_goal-eps

            a_pred2 = np.zeros((1, number_of_robots*6))
            a_pred2[0,0+robot_id*6]=1
            b_pred2 = px_goal-eps

            a_pred3 = np.zeros((1, number_of_robots*6))
            a_pred3[0,1+robot_id*6]=-1
            b_pred3 = -py_goal-eps

            a_pred4 = np.zeros((1, number_of_robots*6))
            a_pred4[0,1+robot_id*6]=1
            b_pred4 = py_goal-eps

            left = LinearPredicate(a_pred1, b_pred1)
            right = LinearPredicate(a_pred2, b_pred2)
            down = LinearPredicate(a_pred3, b_pred3)
            up = LinearPredicate(a_pred4, b_pred4)
            combined_predicate = left & right & down & up
            combined_predicate = combined_predicate.eventually(0, T)
            self.predicates.append(combined_predicate)