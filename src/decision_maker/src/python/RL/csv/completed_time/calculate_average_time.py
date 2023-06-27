import sys
sys.path.append(
    '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/')

from csv_handler import *
from agent import *


if __name__ == "__main__":
    gazebo_env = 'aws_house'
    folder = 'train_result'
    repeat_count = 20
    completed_time_path_csv = f'/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/completed_time/{gazebo_env}/{folder}_{str(repeat_count)}.csv'
    print(calculate_average_from_csv(completed_time_path_csv))
