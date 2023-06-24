# train script
repeat_count = 4
gazebo_env = 'aws_house'
decision_maker = 'train_autonomous_agent'

# records
"""
aws -house 
400 150 
517
422
459
"""
if gazebo_env == "aws_house":
    explore_time = 500

    # output network size
    output_size = 10

    # no frontier counter
    no_frontier_counter = 10


# agents
algo = 'dqn'
# future rewards are weighted strongly
gamma = 0.95
learning_rate = 0.0001

# initial epsilon
epsilon = 1
# agent will still explore 10% of the time.
epsilon_min = 0.1
# decay rate
epsilon_decay = 5e-7

# number of times passed through dataset
epochs = 100
# update target network
save_interval = 10
# model will be updated based on one experience at a time
batch_size = 1
# agent is discouraged from selecting centroids with zero coordinates.
penalty = 0.5

# csv
folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/' + gazebo_env + '/'+ str(repeat_count)
output_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv_results/' + \
    gazebo_env + '_'+ str(repeat_count)+ '.csv'

# print(f'q_values shape: {q_values.shape}')
# print(f'rewards shape: {rewards.shape}')
# print(f'max_next_q_values shape: {max_next_q_values.shape}')