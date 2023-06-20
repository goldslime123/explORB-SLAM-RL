import ast
import csv
import pandas as pd
from dqn import DQNAgent
from ddqn import DDQNAgent
from dueling_dqn import DuelingDQNAgent
from dueling_ddqn import DuelingDDQNAgent


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
                gazebo_env, gamma, learning_rate, tau, epsilon,
                save_interval, epochs, batch_size, penalty,
                robot_positions, robot_orientations,
                centroid_records, info_gain_records, best_centroids)

        elif algo == 'ddqn':
            self.model = DDQNAgent(
                gazebo_env, gamma, learning_rate, tau, epsilon,
                save_interval, epochs, batch_size, penalty,
                robot_positions, robot_orientations,
                centroid_records, info_gain_records, best_centroids)
        elif algo == 'dueling_dqn':
            self.model = DuelingDQNAgent(
                gazebo_env, gamma, learning_rate, tau, epsilon,
                save_interval, epochs, batch_size, penalty,
                robot_positions, robot_orientations,
                centroid_records, info_gain_records, best_centroids)

        elif algo == 'dueling_ddqn':
            self.model = DuelingDDQNAgent(
                gazebo_env, gamma, learning_rate, tau, epsilon,
                save_interval, epochs, batch_size, penalty,
                robot_positions, robot_orientations,
                centroid_records, info_gain_records, best_centroids)

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


def read_csv(directory):
    # Read the CSV file
    df = pd.read_csv(directory, sep=';', header=None)

    # Extract specified columns
    df.columns = ['robot_position', 'robot_orientation',
                  'centroid_record', 'info_gain_record', 'best_centroid']

    robot_position = df['robot_position'].apply(ast.literal_eval)
    robot_orientation = df['robot_orientation'].apply(ast.literal_eval)
    centroid_record = df['centroid_record'].apply(ast.literal_eval)
    info_gain_record = df['info_gain_record'].apply(ast.literal_eval)
    best_centroid = df['best_centroid'].apply(ast.literal_eval)

    return (
        robot_position.tolist(),
        robot_orientation.tolist(),
        centroid_record.tolist(),
        info_gain_record.tolist(),
        best_centroid.tolist()
    )


# def read_csv2(directory):
#     """Reads the input data from a CSV file."""

#     raw_data = pd.read_csv(directory, sep=" ", header=None)

#     robot_position = raw_data[0].apply(
#         lambda x: [float(i) for i in x.split(",")])
#     robot_orientation = raw_data[1].apply(
#         lambda x: [float(i) for i in x.split(",")])
#     centroid_record = raw_data[2].apply(ast.literal_eval)
#     info_gain_record = raw_data[3].apply(ast.literal_eval)
#     best_centroid = raw_data[4].apply(ast.literal_eval)


#     return (
#         robot_position.tolist(),
#         robot_orientation.tolist(),
#         centroid_record.tolist(),
#         info_gain_record.tolist(),
#         best_centroid.tolist()
#     )
if __name__ == "__main__":
    # read dataframe
    directory = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv/aws_house/ed1a5e3.csv'
    # directory='/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/a.csv'
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
        directory)
    # print(robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids)

    # parameters
    gazebo_env = 'aws_house'
    algo = 'dqn'
    gamma = 0.99
    learning_rate = 0.01
    tau = 0.001
    epsilon = 0.5
    save_interval = 5
    epochs = 10
    batch_size = 1
    penalty = 0.5

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
