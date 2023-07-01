"""
Train script
repeat_count - number of times trains_script will launch
gazebo_env - aws_house, aws_bookstore, aws_warehouse
decision maker - using autonmous agent
explore_time - in seconds
output_size - based on number of detected for different env
no_frontier_counter - counter for "No Frontier", as it indicates end of exploration
"""
"""
RL Paremeters
algo - dqn, ddqn, dueling_dqn, dueling_ddqn
gamma - future reward (higher value, rewards weighted strongly)
learning rate - step to find min gradient
epsilon - exploration/exploitation
epsilon_min - decay rate (eg. 0.1 - explore 10%)
epsilon_decay - decay rate to epsilon_min
epochs - number of times going through entire dataset
save_interval - update target network
batch_size - model update one experience at a time (will always be 1)
penalty - agent discourage from selecting centroid with [0,0]
"""
gazebo_env = 'aws_house'

if gazebo_env == "aws_house":
    # dqn, ddqn, dueling_dqn, dueling_ddqn
    algo = 'dueling_ddqn'
    # dqn_5, dqn_10, dqn_15, dqn_20
    model_name = f'{algo}_15'

    repeat_count = 15
    explore_time = 700
    
    output_size = 10
    no_frontier_counter = 10

    # RL Parementers - need to load model (not used for training)
    gamma = 0.90
    learning_rate = 0.01
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.01
    epochs = 100
    save_interval = 10
    batch_size = 1
    penalty = 10

