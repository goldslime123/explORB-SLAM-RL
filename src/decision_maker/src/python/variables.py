"""
Train script
repeat_count - number of times trains_script will launch
gazebo_env - aws_house, aws_bookstore, aws_warehouse
decision maker - using autonmous agent
explore_time - in seconds
output_size - based on number of detected for different env
no_frontier_counter - counter for "No Frontier", as it indicates end of exploration
"""
# select gazebo env
gazebo_env = 'aws_house'

# able to map all areas
if gazebo_env == "aws_house":

    repeat_count = 1
    explore_time = 700
    
    output_size = 10
    no_frontier_counter = 10


# tested aws_bookstore env, unable to fully map the area
if gazebo_env == "aws_bookstore":
    repeat_count = 1
    explore_time = 30*60
    output_size = 10
    no_frontier_counter = 10

"""
RL Paremeters - need to load model (not used for training), optimal paremeters tested
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
# dqn_5, dqn_10, dqn_15, dqn_20
algo = 'dueling_ddqn'
model_name = f'{algo}_15'

gamma = 0.90
learning_rate = 0.01
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.01
epochs = 100
save_interval = 10
batch_size = 1
penalty = 10




