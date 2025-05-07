from goal_distance_predicates import GoalDistancePredicates
import numpy as np
from stlpy.STL import LinearPredicate
class GoalDistanceSTLSpecs:
    def __init__(self):
        self.goal_distance_spec = None
        self.goal_distance_spec_list = []
    def __call__(self, eps, goals_list, number_of_goals, number_of_robots, T):
        if (len(goals_list)==0) or (number_of_goals==None) or (len(number_of_goals)==0):
            a_pred = np.zeros((1, number_of_robots*6))
            b_pred = 0
            pred = LinearPredicate(a_pred, b_pred)
            self.goal_distance_spec = pred
            return -1           
        goal_predicates_ = GoalDistancePredicates()
        for i in range(number_of_robots):
            goal_predicates_(eps, goals_list[0:number_of_goals[i]], i, number_of_robots, T)
            goals_list = goals_list[number_of_goals[i]:]
            goal_predicates = goal_predicates_.predicates
            goal_distance_spec = goal_predicates[0]
            #ToDo:handle multiple goals for each robot
            for i in range(len(goal_predicates)-1):
                goal_distance_spec = goal_distance_spec | goal_predicates[i+1]
            goal_distance_spec.simplify()
            self.goal_distance_spec_list.append(goal_distance_spec)

        self.goal_distance_spec = self.goal_distance_spec_list[0]
        for i in range(1, len(self.goal_distance_spec_list)):
            self.goal_distance_spec = self.goal_distance_spec & self.goal_distance_spec_list[i]