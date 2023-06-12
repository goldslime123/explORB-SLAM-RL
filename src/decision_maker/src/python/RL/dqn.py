import torch
import torch.nn as nn
import numpy as np
import random
import ast
import pandas as pd


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


class DQNModel:
    def __init__(self, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
        # paremeters
        self.robot_post_arr = robot_post_arr
        self.robot_orie_arr = robot_orie_arr
        self.centr_arr = centr_arr
        self.info_arr = info_arr
        self.best_centr_arr = best_centr_arr
        self.gamma = 0.99
        self.tau = 0.001
        self.save_interval = 10
        self.epochs = 100
        self.filepath = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/target_network{self.epochs}.pth"
        self.epsilon = 0.1
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dones = None

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
        self.update_target_network()

    def update_target_network(self):
        """Updates the target DQN parameters using the DQN parameters."""
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self):
        """Saves the target DQN model."""
        torch.save(self.target_dqn.state_dict(), self.filepath)

    def load_model(self):
        """Loads the saved model into the target DQN."""
        self.target_dqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))
        self.target_dqn.eval()
        # Get the centroid with the highest information gain
        max_info_gain_centroid, _ = self.get_max_info_gain_centroid()
        return max_info_gain_centroid

    def select_action(self, state, output_size):
        """Selects an action using the epsilon-greedy approach."""
        if random.random() < self.epsilon:
            action = random.randint(0, output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn(state)
                # Select the desired subset of Q-values (e.g., the first two values)
                # Adjust the range as needed
                selected_q_values = q_values[:, :2]
                action = selected_q_values.argmax(dim=1).item()
        return action

    def calculate_reward(self):
        """Calculates the reward for a given centroid."""
        predicted_centroid, predicted_centroid_idx = self.get_max_info_gain_centroid()

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

        for epoch in range(self.epochs):
            network_input, output_size, _ = self.prepare_input(
                self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
            )

            output = self.dqn(network_input.to(self.device))

            # Compute Targets
            target_network_input, _, _ = self.prepare_input(
                self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
            )

            target_q_values = self.target_dqn(
                target_network_input.to(self.device))
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            rewards, predicted_centroid = self.calculate_reward()

            # check terminating condition
            targets = rewards + self.gamma * \
                (1 - self.dones) * max_target_q_values.detach()

            # Match output tensor shape (1, 5)
            targets = targets.expand_as(output)

            loss = self.criterion(output, targets)

            # Add penalty if the predicted centroid matches [0, 0]
            if torch.all(torch.eq(predicted_centroid, zero_centroid)):
                penalty = 0.1  # Adjust penalty value as needed
                loss += penalty

            self.optimizer.zero_grad()

            action = self.select_action(network_input, output_size)

            loss.backward()

            self.optimizer.step()

            self.update_target_network()
            self.update_epsilon(epoch)

            if (epoch + 1) % self.save_interval == 0:
                self.save_model()
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}, Action: {action}")

    def update_epsilon(self, epoch):
        """Decays epsilon over time."""
        self.epsilon = max(0.01, 0.1 - (0.09 / self.epochs) * epoch)

    def get_max_info_gain_centroid(self):
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
        best_centroid.tolist(),
    )


if __name__ == "__main__":
    robot_positions, robot_orientations, centroid_records, info_gain_records, best_centroid = read_from_csv()

    for i in range(len(robot_positions)):
        model = DQNModel(
            robot_positions[i], robot_orientations[i], centroid_records[i], info_gain_records[i], best_centroid[i]
        )
        model.train()
        
    print(f"The centroid with the highest information gain is {model.load_model()}")
