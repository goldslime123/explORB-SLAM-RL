from agent import *
from csv_handler import *
import sys
sys.path.append(
    '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/')

# RL Parementers
gazebo_env = 'aws_house'
algo = 'dueling_ddqn'
repeat_count = 5
gamma = 0.90
learning_rate = 0.01
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.01
epochs = 100
save_interval = 10
batch_size = 1
penalty = 10

# csv path
folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/train_data/' + \
    gazebo_env + '/' + str(repeat_count)

combined_output_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/combined_results/' + gazebo_env + '/' + \
    gazebo_env + '_' + str(repeat_count) + '.csv'

model_path = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/{algo}/{algo}_{repeat_count}.pth"

# combine csv, train the model (select algo in the paremeters), save plot if required
def train_model():
    combine_csv(folder_path, combined_output_path)
    # read dataframe
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
        combined_output_path)
    # print(robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids)

    #  model
    model = Agent(model_path, algo, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                  save_interval, epochs, batch_size, penalty,
                  robot_positions, robot_orientations,
                  centroid_records, info_gain_records, best_centroids)

    # train model
    model.train()
    # model.save_plot()

# test model by prining the predicted result of each row in the combined csv files
def test_model():
    # read dataframe
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
        combined_output_path)

    # model
    model = Agent(model_path, algo, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                  save_interval, epochs, batch_size, penalty,
                  robot_positions, robot_orientations,
                  centroid_records, info_gain_records, best_centroids)

    model.load_model()

    # show result for each row
    for i in range(len(robot_positions)):
        predicted_centroid, max_info_gain_centroid_idx = model.predict_centroid(
            robot_positions[i], robot_orientations[i], centroid_records[i], info_gain_records[i])
        print(
            f"The centroid with the highest information gain for row {i+1} is {predicted_centroid} Index: {max_info_gain_centroid_idx}")


if __name__ == "__main__":
    # train_model()
    test_model()
