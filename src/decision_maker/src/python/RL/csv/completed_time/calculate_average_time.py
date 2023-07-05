import sys
sys.path.append(
    '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL')

from csv_handler import *
from agent import *


if __name__ == "__main__":
    gazebo_env = 'aws_house'
    algo = 'dueling_ddqn'
    repeat_count = 15
    model_name = f'{algo}_{str(repeat_count)}'
    completed_time_path_csv = f'/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/completed_time/{gazebo_env}/{algo}/{model_name}.csv'
    print(calculate_average_from_csv(completed_time_path_csv))
