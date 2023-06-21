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
    def __init__(self, algo, gazebo_env, gamma, learning_rate, tau, epsilon,
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
        self.tau = tau
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty

        # Initialize the model attribute
        self.model = None

        # Initialize the specific model based on the chosen algorithm
        if algo == 'dqn':
            self.model = DQNAgent(
                self.gazebo_env, self.gamma, self.learning_rate, self.tau, self.epsilon,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif algo == 'ddqn':
            self.model = DDQNAgent(
                self.gazebo_env, self.gamma, self.learning_rate, self.tau, self.epsilon,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif algo == 'dueling_dqn':
            self.model = DuelingDQNAgent(
                self.gazebo_env, self.gamma, self.learning_rate, self.tau, self.epsilon,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif algo == 'dueling_ddqn':
            self.model = DuelingDDQNAgent(
                self.gazebo_env, self.gamma, self.learning_rate, self.tau, self.epsilon,
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
