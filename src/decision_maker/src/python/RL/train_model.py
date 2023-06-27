import sys
sys.path.append(
    '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/')
from csv_handler import *
from agent import *

# RL Parementers
gazebo_env = 'aws_house'
algo = 'dueling_ddqn'
repeat_count = 5

gamma = 0.90
learning_rate = 0.0001
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 5e-7
epochs = 50
save_interval = 10
batch_size = 1
penalty = 0.5

# csv
folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/train_data/' + \
    gazebo_env + '/' + str(repeat_count)

combined_output_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/combined_results/' + gazebo_env +'/'+ \
    gazebo_env + '_'+ str(repeat_count) + '.csv'

model_path = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/{algo}/{algo}_{repeat_count}.pth"

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


def test_model():
    # read dataframe
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_csv(
        combined_output_path)

    # model
    model = Agent(model_path, algo, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
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

    # show result for each row
    for i in range(len(robot_positions)):
        predicted_centroid, max_info_gain_centroid_idx = model.predict_centroid(
            robot_positions[i], robot_orientations[i], centroid_records[i], info_gain_records[i])
        print(
            f"The centroid with the highest information gain for row {i+1} is {predicted_centroid} Index: {max_info_gain_centroid_idx}")


if __name__ == "__main__":

    
    
    train_model()
    test_model()

    
