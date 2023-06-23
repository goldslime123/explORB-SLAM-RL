import pandas as pd
from dqn import DQNAgent
from ddqn import DDQNAgent
from dueling_dqn import DuelingDQNAgent
from dueling_ddqn import DuelingDDQNAgent
from csv_handler import *
from agent import *

import sys
sys.path.append('/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/')
from variables import *

# parameters
# gazebo_env = 'aws_house'
# algo = 'dqn'
# gamma = 0.99
# learning_rate = 0.01
# tau = 0.001
# epsilon = 0.5
# save_interval = 5
# epochs = 10
# batch_size = 1
# penalty = 0.5

# folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv/' + gazebo_env
# output_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/' + \
#     gazebo_env + '.csv'


def train_model():
    combine_csv(folder_path, output_path)
    # read dataframe
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
        output_path)
    print(robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids)

    # create model
    model = Agent(algo, gazebo_env, gamma, learning_rate, tau, epsilon,
                  save_interval, epochs, batch_size, penalty,
                  robot_positions, robot_orientations,
                  centroid_records, info_gain_records, best_centroids)

    if algo == 'dqn':
        model.initialize_dqn()
    elif algo == 'ddqn':
        model.initialize_ddqn()
    elif algo == 'dueling_dqn':
        model.initialize_dueling_dqn()
    elif algo == 'dueling_ddqn':
        model.initialize_dueling_ddqn()

    # train model
    model.train()

    # show result for each row
    for i in range(len(robot_positions)):
        predicted_centroid, max_info_gain_centroid_idx = model.predict_centroid(
            robot_positions[i], robot_orientations[i], centroid_records[i], info_gain_records[i])
        print(
            f"The centroid with the highest information gain for row {i+1} is {predicted_centroid} Index: {max_info_gain_centroid_idx}")


if __name__ == "__main__":
    train_model()

    # robot_position = 1.6998447315016334, 3.52581991835878
    # robot_orientation = - \
    #     0.0030015861938741785, 0.002416949504063036, 0.8137270796538675, 0.5812343736311842
    # centroid_record = [
    #     4.20577174, -1.40001973], [0.58587992, -0.60072875], [0, 0], [0, 0], [0, 0]
    # info_gain_record = [8.421770797635304], [148.53895792332335], [0], [0], [0]
