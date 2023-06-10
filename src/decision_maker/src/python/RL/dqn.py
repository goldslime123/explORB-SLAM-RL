import torch
import torch.nn as nn
import numpy as np
import random

from read_csv import *


class DQNModel:
    def __init__(self, robot_post_arr, robot_orie_arr, centr_arr, info_arr):
        self.robot_post_arr = robot_post_arr
        self.robot_orie_arr = robot_orie_arr
        self.centr_arr = centr_arr
        self.info_arr = info_arr
        self.tau = 0.001
        self.save_interval = 10
        self.epochs = 1000
        self.filepath = f"/home/kenji/ws/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/target_network{self.epochs}.pth"

        # Initialize the DQN network
        self.initialize_dqn()

        self.epsilon = 0.1  # Set the initial value of epsilon

    def initialize_dqn(self):
        # Convert the NumPy arrays to PyTorch tensors
        robot_position = torch.tensor(self.robot_post_arr, dtype=torch.float32)
        robot_orientation = torch.tensor(
            self.robot_orie_arr, dtype=torch.float32)
        centroid_record = torch.tensor(self.centr_arr, dtype=torch.float32)
        info_gain_record = torch.tensor(self.info_arr, dtype=torch.float32)

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
        output_size = sorted_centroid_record.shape[0] * sorted_centroid_record.shape[1]

        # Define the DQN network
        class DQN(nn.Module):
            def __init__(self, input_size, output_size):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, output_size)

            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x

        # Create instances of the DQN and TargetDQN networks
        self.dqn = DQN(input_size, output_size)
        self.target_dqn = DQN(input_size, output_size)

        # Define the criterion and the optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters())

        # After each update step, update the target network
        self.update_target_network()

    def update_target_network(self):
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    # Save the model
    def save_model(self):
        torch.save(self.target_dqn.state_dict(), self.filepath)

    def load_model(self, device):
        self.target_dqn.load_state_dict(
            torch.load(self.filepath, map_location=device))
        self.target_dqn.eval()

    def select_action(self, state, output_size):
        if random.random() < self.epsilon:
            # Explore: Choose a random action
            action = random.randint(0, output_size - 1)
        else:
            # Exploit: Choose the best action based on the current Q-values
            with torch.no_grad():
                q_values = self.dqn(state)
                action = q_values.argmax().item()

        return action

    def train(self):
        for epoch in range(self.epochs):
            # Convert the NumPy arrays to PyTorch tensors
            robot_position = torch.tensor(
                self.robot_post_arr, dtype=torch.float32)
            robot_orientation = torch.tensor(
                self.robot_orie_arr, dtype=torch.float32)
            centroid_record = torch.tensor(
                self.centr_arr, dtype=torch.float32)
            info_gain_record = torch.tensor(
                self.info_arr, dtype=torch.float32)

            # Concatenate the reshaped robot_position and robot_orientation
            robot_state = torch.cat((robot_position, robot_orientation))

            # Concatenate the robot state with the centroid record and info gain record
            combined_data = torch.cat(
                (centroid_record, info_gain_record), dim=1)
            sorted_data = combined_data[combined_data[:, -1].argsort(descending=True)]

            # Extract the sorted centroid record and info gain record
            sorted_centroid_record = sorted_data[:, :-1]
            sorted_info_gain_record = sorted_data[:, -1]

            # Flatten and concatenate the robot state, sorted centroid record, and sorted info gain record
            network_input = torch.cat(
                (robot_state, sorted_centroid_record.flatten(), sorted_info_gain_record.flatten()))

            # Reshape the network input
            input_size = network_input.numel()
            network_input = network_input.reshape(1, input_size)

            # Define the target
            target = sorted_centroid_record

            # Pass the network input through the DQN network
            output = self.dqn(network_input)

            # Reshape the output to match the shape of sorted_centroid_record
            output = output.view(sorted_centroid_record.shape)

            # Compute the loss
            loss = self.criterion(output, target)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Choose the action with epsilon-greedy policy
            action = self.select_action(network_input, output_size=sorted_centroid_record.numel())

            # Perform a backward pass (backpropagation)
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # After each update step, update the target network
            self.update_target_network()

            # Update epsilon
            self.update_epsilon(epoch)

            # Save the target network model at the specified interval
            if (epoch + 1) % self.save_interval == 0:
                self.save_model()

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}, Action: {action}")

    def update_epsilon(self, epoch):
        # Decay epsilon over time
        self.epsilon = max(0.01, 0.1 - (0.09 / self.epochs) * epoch)

    def get_max_info_gain_centroid(self):
        # Ensure the model is in evaluation mode
        self.target_dqn.eval()

        # Convert the NumPy arrays to PyTorch tensors
        robot_position = torch.tensor(
            self.robot_post_arr, dtype=torch.float32)
        robot_orientation = torch.tensor(
            self.robot_orie_arr, dtype=torch.float32)
        centroid_record = torch.tensor(
            self.centr_arr, dtype=torch.float32)
        info_gain_record = torch.tensor(
            self.info_arr, dtype=torch.float32)

        # Concatenate the robot's state
        robot_state = torch.cat((robot_position, robot_orientation))

        # Concatenate the robot state with the centroid record and info gain record
        combined_data = torch.cat(
            (centroid_record, info_gain_record), dim=1)
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

        # Pass the network input through the target network
        with torch.no_grad():
            output = self.target_dqn(network_input)
            output = output.reshape(sorted_centroid_record.shape)

        # Find the index of the maximum predicted gain, which gives us the centroid with the highest predicted gain
        max_info_gain_centroid_idx = np.argmax(output.numpy())

        # Ensure the index is within bounds
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[0]

        # Return the centroid with the highest info gain
        return sorted_centroid_record[max_info_gain_centroid_idx]


if __name__ == '__main__':
    robot_post_arr, robot_orie_arr, centr_arr, info_arr = read_from_csv()
    model = DQNModel(robot_post_arr, robot_orie_arr, centr_arr, info_arr)

    model.train()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_model(device)

    # Find the centroid with the highest info gain
    max_info_gain_centroid = model.get_max_info_gain_centroid()
    print(f"The centroid with the highest information gain is {max_info_gain_centroid}")