import numpy as np
from stlpy.STL import LinearPredicate

class GoalDistanceSTLSpecs:
    def __init__(self):
        self.goal_distance_spec = None
        self.goal_distance_spec_list = []
    def __call__(self, eps, goals_list, number_of_goals, number_of_robots, Time_list):
        if (len(goals_list)==0) or (number_of_goals==None) or (len(number_of_goals)==0):
            a_pred = np.zeros((1, number_of_robots*6))
            b_pred = 0
            pred = LinearPredicate(a_pred, b_pred)
            self.goal_distance_spec = pred
            return -1           
        goal_predicates_ = GoalDistancePredicates()
        for i in range(number_of_robots):
            goal_predicates_(eps, goals_list[0:number_of_goals[i]], i, number_of_robots, Time_list[i])
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

class ObstacleAvoidanceSTLSpecs:
    def __init__(self):
        self.obstacle_avoidance_spec = None
    def __call__(self, obstacles, number_of_robots, T, eps=0):
        predicate_list = []
        number_of_obstacles = len(obstacles)
        for i in range(number_of_obstacles):
            x_min_obs = obstacles[i][0] - obstacles[i][2]
            x_max_obs = obstacles[i][0] + obstacles[i][2]
            y_min_obs = obstacles[i][1] - obstacles[i][2]
            y_max_obs = obstacles[i][1] + obstacles[i][2]

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