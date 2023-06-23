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




def train_model():
    combine_csv(folder_path, output_path)
    # read dataframe
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
        output_path)
    # print(robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids)

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
