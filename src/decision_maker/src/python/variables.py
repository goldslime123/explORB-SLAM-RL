# train script
repeat_count = 1
gazebo_env = 'aws_house'
decision_maker = 'train_autonomous_agent'

# records
"""
aws -house 
400 150 
517
422
"""
if gazebo_env == "aws_house":
    explore_time = 9000

    # output network size
    output_size = 10

    # no frontier counter
    no_frontier_counter = 10


# agents
algo = 'dqn'
gamma = 0.99
learning_rate = 0.01
tau = 0.001
epsilon = 0.5
save_interval = 5
epochs = 10
batch_size = 1
penalty = 0.5

# csv
folder_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/csv/' + gazebo_env + '/'+ str(repeat_count)
output_path = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/csv/' + \
    gazebo_env + '_'+ str(repeat_count)+ '.csv'
