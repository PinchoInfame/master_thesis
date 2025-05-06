import numpy as np
from stlpy.STL import LinearPredicate
class ObstacleAvoidanceSTLSpecs:
    def __init__(self):
        self.obstacle_avoidance_spec = None
    def __call__(self, obstacle_bounds_list, number_of_robots, T, eps):
        predicate_list = []
        number_of_obstacles = len(obstacle_bounds_list)
        for i in range(number_of_obstacles):
            x_min_obs = obstacle_bounds_list[i][0]
            x_max_obs = obstacle_bounds_list[i][1]
            y_min_obs = obstacle_bounds_list[i][2]
            y_max_obs = obstacle_bounds_list[i][3]

            for j in range(number_of_robots):
                a_pred1 = np.zeros((1, number_of_robots*6))
                a_pred1[0,0+j*6]=-1
                b_pred1 = -x_min_obs+eps

                a_pred2 = np.zeros((1, number_of_robots*6))
                a_pred2[0,0+j*6]=1
                b_pred2 = x_max_obs+eps

                a_pred3 = np.zeros((1, number_of_robots*6))
                a_pred3[0,1+(j*6)]=-1
                b_pred3 = -y_min_obs+eps

                a_pred4 = np.zeros((1, number_of_robots*6))
                a_pred4[0,1+(j*6)]=1
                b_pred4 = y_max_obs+eps

                predicate1 = LinearPredicate(a_pred1, b_pred1)
                predicate2 = LinearPredicate(a_pred2, b_pred2)
                predicate3 = LinearPredicate(a_pred3, b_pred3)
                predicate4 = LinearPredicate(a_pred4, b_pred4)
                combined_predicate = predicate1 | predicate2 | predicate3 | predicate4
                combined_predicate = combined_predicate.always(0,T)
                predicate_list.append(combined_predicate)
        self.obstacle_avoidance_spec=(predicate_list[0]) 
        for pred in predicate_list[1:]:
            self.obstacle_avoidance_spec = self.obstacle_avoidance_spec & pred
        self.obstacle_avoidance_spec.simplify()