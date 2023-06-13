import torch
import torch.nn as nn
import numpy as np
import random
import ast
import pandas as pd
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *random.sample(self.buffer, batch_size)
        )
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class DQNModel:
    def __init__(self, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
        # paremeters
        self.robot_post_arr = robot_post_arr[0]
        self.robot_orie_arr = robot_orie_arr[0]
        self.centr_arr = centr_arr[0]
        self.info_arr = info_arr[0]
        self.best_centr_arr = best_centr_arr[0]

        self.robot_post_arr2 = robot_post_arr
        self.robot_orie_arr2 = robot_orie_arr
        self.centr_arr2 = centr_arr
        self.info_arr2 = info_arr
        self.best_centr_arr2 = best_centr_arr

        self.gamma = 0.99
        self.learning_rate = 0.01
        self.tau = 0.001
        self.save_interval = 5
        self.epochs = 10
        self.filepath = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/target_network{self.epochs}.pth"
        self.epsilon = 0.5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dones = None
        self.batch_size = 1

        # Initialize the replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Initialize the DQN network
        self.initialize_dqn()

    def prepare_input(self, robot_pos, robot_orie, centr, info):
        """Prepares the input for the DQN model."""
        # Convert the NumPy arrays to PyTorch tensors
        robot_position = torch.tensor(robot_pos, dtype=torch.float32)
        robot_orientation = torch.tensor(robot_orie, dtype=torch.float32)
        centroid_record = torch.tensor(centr, dtype=torch.float32)
        info_gain_record = torch.tensor(info, dtype=torch.float32)

        # Concatenate the robot's state
        robot_state = torch.cat((robot_position, robot_orientation))

        # Concatenate the robot state with the centroid record and info gain record
        combined_data = torch.cat((centroid_record, info_gain_record), dim=1)
        sorted_data = combined_data[combined_data[:, -
                                                  1].argsort(descending=True)]

        # Extract the sorted centroid record and info gain record
        sorted_centroid_record = sorted_data[:, :-1]
        sorted_info_gain_record = sorted_data[:, -1]

        # Flatten and concatenate the robot state, sorted centroid record, and sorted info gain record
        network_input = torch.cat(
            (robot_state, sorted_centroid_record.flatten(), sorted_info_gain_record.flatten()))

        # Reshape the network input
        input_size = network_input.numel()
        network_input = network_input.reshape(1, input_size)

        # Determine the output size based on the shape of the sorted centroid record
        output_size = 5

        return network_input, output_size, sorted_centroid_record

    def initialize_dqn(self):
        """Initializes the DQN and target DQN models, and the optimizer and loss function."""
        network_input, output_size, _ = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
        )
        self.dqn = DQN(network_input.shape[1], output_size).to(self.device)
        self.target_dqn = DQN(
            network_input.shape[1], output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.save_model()

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """Add a new experience to the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_from_replay_buffer(self, batch_size):
        """Randomly sample a batch of experiences from the replay buffer."""
        return random.sample(self.replay_buffer, batch_size)

    def update_target_network(self):
        """Updates the target DQN parameters using the DQN parameters."""
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        self.save_model()

    def save_model(self):
        """Saves the target DQN model."""
        torch.save(self.target_dqn.state_dict(), self.filepath)

    def load_model(self):
        """Loads the saved model into the target DQN."""
        self.target_dqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))
        self.target_dqn.eval()

    def select_action(self, state, output_size):
        """Selects an action using the epsilon-greedy approach."""
        if random.random() < self.epsilon:
            action = random.randint(0, output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn(state)
                # Select the desired subset of Q-values (e.g., the first two values)
                # Adjust the range as needed
                selected_q_values = q_values[:, :5]

                action = selected_q_values.argmax(dim=1).item()
        return action

    def calculate_reward(self):
        """Calculates the reward for a given centroid."""
        predicted_centroid, _ = self.get_max_info_gain_centroid()

        target_centroid = torch.tensor(
            self.best_centr_arr, dtype=torch.float32, device=self.device)

        # Check if the predicted centroid matches the best centroid
        match = torch.all(torch.eq(predicted_centroid, target_centroid))

        # Increase the reward for the chosen predicted centroid if it matches the target centroid
        if match:
            reward = 1
        else:
            reward = 0

        return reward, predicted_centroid

    def train(self):
        zero_centroid = torch.tensor([0.0, 0.0], device=self.device)
        self.dones = torch.zeros((1,), device=self.device)

        self.replay_buffer = ReplayBuffer(10000)  # Initialize replay buffer
        for epoch in range(self.epochs):
            for i in range(len(self.robot_post_arr2)-1):
                self.load_model()
                network_input, output_size, _ = self.prepare_input(
                    self.robot_post_arr2[i], self.robot_orie_arr2[i], self.centr_arr2[i], self.info_arr2[i]
                )

                # select action
                actions = self.select_action(network_input, output_size)

                rewards, predicted_centroid = self.calculate_reward()

                # save next state
                next_state, _, _ = self.prepare_input(
                    self.robot_post_arr2[i+1], self.robot_orie_arr2[i +
                                                                    1], self.centr_arr2[i+1], self.info_arr2[i+1]
                )
                done = torch.all(torch.eq(predicted_centroid,
                                 torch.tensor([0.0, 0.0], device=self.device)))

                # store into replay buffer current state
                self.replay_buffer.push(
                    network_input, actions, rewards, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
                    # Get a batch of experiences from the replay buffer
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                        self.batch_size)

                    states = torch.stack(states).to(self.device)
                    actions = torch.tensor(
                        actions, dtype=torch.long).unsqueeze(-1).to(self.device)
                    rewards = torch.tensor(
                        rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
                    next_states = torch.stack(next_states).to(self.device)
                    dones = torch.tensor(
                        dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

                    q_values = self.dqn(states)
                    next_q_values = self.target_dqn(next_states)
                    max_next_q_values = next_q_values.max(
                        dim=1, keepdim=True)[0]

                    targets = q_values + self.learning_rate * \
                        (rewards * q_values + self.gamma *
                         max_next_q_values - q_values)

                    targets = targets.expand_as(q_values)
                    loss = self.criterion(q_values, targets)

                    # Add penalty if the predicted centroid matches [0, 0]
                    if torch.all(torch.eq(predicted_centroid, zero_centroid)):
                        penalty = 0.5  # Adjust penalty value as needed
                        loss += penalty

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.update_epsilon(epoch)

            if self.epochs % self.save_interval == 0:
                self.update_target_network()
                self.save_model()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    def update_epsilon(self, epoch):
        """Decays epsilon over time."""
        self.epsilon = max(0.01, 0.1 - (0.09 / self.epochs) * epoch)

    def get_max_info_gain_centroid(self):
        self.load_model()
        """Finds the centroid with the highest information gain."""
        network_input, _, sorted_centroid_record = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
        )
        with torch.no_grad():
            output = self.target_dqn(network_input.to(self.device))
        max_info_gain_centroid_idx = np.argmax(output.cpu().numpy())
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx

    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        """Predicts the best centroid based on the given robot position and orientation using the target network."""
        self.load_model()

        centr_arr = [[1, 2], [3, 4], [0, 0], [0, 0], [0, 0]]
        info_arr = [[10], [20], [0], [0], [0]]

        # print(robot_position, robot_orientation,
        #       centroid_records, info_gain_records)
        """Finds the centroid with the highest information gain."""
        network_input, _, sorted_centroid_record = self.prepare_input(
            robot_position, robot_orientation, centroid_records, info_gain_records
        )
        with torch.no_grad():
            output = self.target_dqn(network_input.to(self.device))
        max_info_gain_centroid_idx = np.argmax(output.cpu().numpy())
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid,max_info_gain_centroid_idx


def read_from_csv():
    """Reads the input data from a CSV file."""
    directory = '/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/a.csv'
    raw_data = pd.read_csv(directory, sep=" ", header=None)
    robot_position = raw_data[0].apply(
        lambda x: [float(i) for i in x.split(",")])
    robot_orientation = raw_data[1].apply(
        lambda x: [float(i) for i in x.split(",")])
    centroid_record = raw_data[2].apply(ast.literal_eval)
    info_gain_record = raw_data[3].apply(ast.literal_eval)
    best_centroid = raw_data[4].apply(ast.literal_eval)
    return (
        robot_position.tolist(),
        robot_orientation.tolist(),
        centroid_record.tolist(),
        info_gain_record.tolist(),
        best_centroid.tolist()
    )


if __name__ == "__main__":
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_from_csv()

    # Create a new model for the first row
    model = DQNModel(robot_positions, robot_orientations,
                     centroid_records, info_gain_records, best_centroids)
    model.initialize_dqn()

    model.train()

    for i in range(len(robot_positions)):

        predicted_centroid,max_info_gain_centroid_idx = model.predict_centroid(
            robot_positions[i], robot_orientations[i], centroid_records[i], info_gain_records[i])
        print(
            f"The centroid with the highest information gain for row {i+1} is {predicted_centroid} Index: {max_info_gain_centroid_idx}")

    # predicted_centroid = model.predict_centroid(
    #     robot_positions[0], robot_orientations[0], centroid_records[0], info_gain_records[0])
    # print("Predicted centroid:", predicted_centroid)


# if __name__ == "__main__":
#     robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroids = read_from_csv()

#     model = None  # Initialize the model outside the loop
#     predicted_centroids = []

#     for i in range(len(robot_positions)):
#         if model is None:
#             # Create a new model for the first row
#             model = DQNModel(robot_positions, robot_orientations,
#                              centroid_records, info_gain_records, best_centroids[i])
#             model.initialize_dqn()
#         else:
#             # Update the model for subsequent rows
#             model.robot_post_arr = robot_positions[i]
#             model.robot_orie_arr = robot_orientations[i]
#             model.centr_arr = centroid_records[i]
#             model.info_arr = info_gain_records[i]
#             model.best_centr_arr = best_centroids[i]

#         model.train()
#         predicted_centroid = model.get_max_info_gain_centroid()
#         predicted_centroids.append(predicted_centroid)

#     for i, centroid in enumerate(predicted_centroids):
#         print(
#             f"The centroid with the highest information gain for row {i+1} is {centroid}")

#     predicted_centroid = model.predict_centroid(
#         robot_positions[0], robot_orientations[0])
#     print("Predicted centroid:", predicted_centroid)
