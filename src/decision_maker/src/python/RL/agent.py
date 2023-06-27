import pandas as pd
from dqn import DQNAgent
from ddqn import DDQNAgent
from dueling_dqn import DuelingDQNAgent
from dueling_ddqn import DuelingDDQNAgent
from csv_handler import *

# import sys
# sys.path.append('/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python')
# from paremeter import *


class Agent:
    def __init__(self, model_path, algo, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                 save_interval, epochs, batch_size, penalty, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
        # parameters
        self.robot_post_arr = robot_post_arr
        self.robot_orie_arr = robot_orie_arr
        self.centr_arr = centr_arr
        self.info_arr = info_arr
        self.best_centr_arr = best_centr_arr

        self.algorithm = algo
        self.gazebo_env = gazebo_env
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.save_interval = save_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty

        # Initialize the model attribute
        self.model = None

        self.model_path = model_path

        # Initialize the specific model based on the chosen algorithm
        if algo == 'dqn':
            self.model = DQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif algo == 'ddqn':
            self.model = DDQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif algo == 'dueling_dqn':
            self.model = DuelingDQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif algo == 'dueling_ddqn':
            self.model = DuelingDDQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        else:
            raise ValueError("Invalid algorithm.")

    def save_model(self):
        """Saves the target DQN model."""
        self.model.save_model()

    def load_model(self):
        """Loads the saved model into the target DQN."""
        self.model.load_model()

    def initialize_dqn(self):
        self.model.initialize_dqn()

    def initialize_ddqn(self):
        self.model.initialize_ddqn()

    def initialize_dueling_dqn(self):
        self.model.initialize_dueling_dqn()

    def initialize_dueling_ddqn(self):
        self.model.initialize_dueling_ddqn()

    def train(self):
        self.model.train()  # Call the train method of the specific model

    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        return self.model.predict_centroid(robot_position, robot_orientation, centroid_records, info_gain_records)


# if __name__ == "__main__":
#     gazebo_env = 'aws_house'
#     algo = 'dqn'
#     gamma = 0.99
#     learning_rate = 0.01
#     tau = 0.001
#     epsilon = 0.5
#     save_interval = 5
#     epochs = 10
#     batch_size = 1
#     penalty = 0.5

#     robot_position = [[0.11, 2.5]]
#     robot_orientation = [[1.0, 0.00222544, 0.92653255, -0.37618656]]
#     centroid_records = [([4.20577174,-1.40001973],[0.58587992,-0.60072875],[0,0],[0,0],[0,0],[0,0])]
#     info_gain_records=[([8.421770797635304],[148.53895792332335],[0],[0],[0],[0])]
#     best_centroid = [[4.20577174,-1.40001973]]

#     # robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
#     #     output_path)
#     # print(robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids)

#     # create model
#     model = Agent(algo, gazebo_env, gamma, learning_rate, tau, epsilon,
#                   save_interval, epochs, batch_size, penalty,
#                   robot_position, robot_orientation,
#                   centroid_records, info_gain_records, best_centroid)

#     predicted_centroid,_ = model.predict_centroid(robot_position[0], robot_orientation[0],
#                 centroid_records[0], info_gain_records[0])
#     print(predicted_centroid)
