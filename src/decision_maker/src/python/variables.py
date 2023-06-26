"""
Train script
repeat_count - number of times trains_script will launch
gazebo_env - aws_house, aws_bookstore, aws_warehouse
decision maker - using autonmous agent
explore_time - in seconds
output_size - based on number of detected for different env
no_frontier_counter - counter for "No Frontier", as it indicates end of exploration
"""
repeat_count = 25
gazebo_env = 'aws_house'
decision_maker = 'train_autonomous_agent'



# diff env paremeters
if gazebo_env == "aws_house":
    # explore_time = 600
    explore_time = 15*60

    output_size = 10
    no_frontier_counter = 8


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
algo = 'dqn'
gamma = 0.95
learning_rate = 0.0001

epsilon = 1
epsilon_min = 0.1
epsilon_decay = 5e-7

epochs = 100
save_interval = 10
batch_size = 1
penalty = 0.5


# records
"""
aws -house 
400 150 
517
422
459
"""
