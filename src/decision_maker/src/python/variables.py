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
gamma = 0.99
learning_rate = 0.01
tau = 0.001

epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995

save_interval = 5
epochs = 50
batch_size = 1

penalty = 0.1

# csv
folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/' + gazebo_env + '/'+ str(repeat_count)
output_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv_results/' + \
    gazebo_env + '_'+ str(repeat_count)+ '.csv'
