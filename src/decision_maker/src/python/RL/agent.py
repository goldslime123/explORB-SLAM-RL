from dqn import DQNAgent
from ddqn import DDQNAgent
from dueling_dqn import DuelingDQNAgent
from dueling_ddqn import DuelingDDQNAgent
from csv_handler import *


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
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.algorithm == 'dqn':
            self.model = DQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif self.algorithm == 'ddqn':
            self.model = DDQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif self.algorithm == 'dueling_dqn':
            self.model = DuelingDQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        elif self.algorithm == 'dueling_ddqn':
            self.model = DuelingDDQNAgent(
                self.model_path, self.gazebo_env, self.gamma, self.learning_rate,
                self.epsilon, self.epsilon_min, self.epsilon_decay,
                self.save_interval, self.epochs, self.batch_size, self.penalty,
                self.robot_post_arr, self.robot_orie_arr,
                self.centr_arr, self.info_arr, self.best_centr_arr)

        else:
            raise ValueError("Invalid algorithm.")

        return self.model

    def save_model(self):
        """Saves the target DQN model."""
        self.model.save_model()

    def load_model(self):
        """Loads the saved model into the target DQN."""
        self.model.load_model()

    def train(self):
        self.model.train()  # Call the train method of the specific model

    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        return self.model.predict_centroid(robot_position, robot_orientation, centroid_records, info_gain_records)
