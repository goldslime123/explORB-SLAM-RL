import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from replay_buffer import ReplayBuffer

class DDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DDQN, self).__init__()
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


class DDQNAgent:
    def __init__(self, gazebo_env, gamma, learning_rate, epsilon, epsilon_min, epsilon_decay,
                 save_interval, epochs, batch_size, penalty, robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
        # Parameters
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

        self.gamma = gamma
        self.learning_rate = learning_rate

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.save_interval = save_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty

        self.gazebo_env = gazebo_env
        self.filepath = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/ddqn_{self.epochs}.pth"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dones = None

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(1000)

        # Initialize the DDQN network
        self.initialize_ddqn()

    def prepare_input(self, robot_pos, robot_orie, centr, info):
        """Prepares the input for the DDQN model."""
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
        output_size = sorted_centroid_record.shape[0]

        return network_input, output_size, sorted_centroid_record

    def initialize_ddqn(self):
        """Initializes the DDQN and target DDQN models, and the optimizer and loss function."""
        network_input, output_size, _ = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
        )
        self.ddqn = DDQN(network_input.shape[1], output_size).to(self.device)
        self.target_ddqn = DDQN(
            network_input.shape[1], output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.ddqn.parameters())
        self.save_model()

    def update_target_network(self):
        """Updates the target DDQN parameters using the DDQN parameters."""
        self.target_ddqn.load_state_dict(self.ddqn.state_dict())

    def save_model(self):
        """Saves the target DDQN model."""
        torch.save(self.target_ddqn.state_dict(), self.filepath)

    def load_model(self):
        """Loads the saved model into the target DDQN."""
        self.ddqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))
        self.target_ddqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))

    def select_action(self, state, output_size, sorted_centroid_record):
        """Selects an action using the epsilon-greedy approach."""
        if random.random() < self.epsilon:
            action = random.randint(0, output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.ddqn(state).clone()

                # Get the indices of the centroids which are [0.0, 0.0]
                indices = (sorted_centroid_record == torch.tensor(
                    [0.0, 0.0])).all(1).nonzero(as_tuple=True)[0]

                # Apply penalty to q_values at those indices
                q_values[0, indices] = -self.penalty

                action = q_values.argmax(dim=1).item()

        # Apply epsilon decay if epsilon is more than its minimum value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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

         # Apply penalty to the reward if the predicted centroid is [0.0, 0.0]
        zero_centroid = torch.tensor([0.0, 0.0], device=self.device)
        if torch.all(torch.eq(predicted_centroid, zero_centroid)):
            reward -= self.penalty

        return reward, predicted_centroid

    def train(self):
        self.dones = torch.zeros((1,), device=self.device)
        for epoch in range(self.epochs):
            for i in range(len(self.robot_post_arr2) - 1):
                self.load_model()
                network_input, output_size, sorted_centroid_record = self.prepare_input(
                    self.robot_post_arr2[i], self.robot_orie_arr2[i], self.centr_arr2[i], self.info_arr2[i]
                )

                # Select action
                actions = self.select_action(
                    network_input, output_size, sorted_centroid_record)

                rewards, predicted_centroid = self.calculate_reward()

                # Save next state
                next_state, _, _ = self.prepare_input(
                    self.robot_post_arr2[i + 1], self.robot_orie_arr2[i +
                                                                      1], self.centr_arr2[i + 1], self.info_arr2[i + 1]
                )
                done = torch.all(torch.eq(predicted_centroid,
                                 torch.tensor([0.0, 0.0], device=self.device)))

                # Store current state in the replay buffer
                self.replay_buffer.push(
                    network_input, actions, rewards, next_state, done
                )

                if len(self.replay_buffer) >= self.batch_size:
                    # Get a batch of experiences from the replay buffer
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                        self.batch_size
                    )

                    states = torch.stack(states).to(self.device)
                    actions = torch.tensor(
                        actions, dtype=torch.long).unsqueeze(-1).to(self.device)
                    rewards = torch.tensor(
                        rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
                    next_states = torch.stack(next_states).to(self.device)
                    dones = torch.tensor(
                        dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

                    # Compute the Q-values for the next states using the online network
                    q_values = self.ddqn(states)

                    # Select the action with the highest Q-value using the online network
                    next_q_values = self.ddqn(next_states)
                    
                    # diff here
                    next_actions = next_q_values.argmax(dim=1, keepdim=True)

                    # Estimate the corresponding Q-values for the next states using the target network
                    target_next_q_values = self.target_ddqn(next_states)

                    # Extract the maximum Q-values for the next states based on the selected actions
                    max_next_q_values = target_next_q_values.gather(
                        1, next_actions).detach()

                    # Calculate the target Q-values by combining the immediate rewards, discounted maximum Q-values, and episode termination
                    targets = rewards + self.gamma * \
                        (max_next_q_values * (1 - dones))

                    targets = targets.expand_as(q_values)
                    loss = self.criterion(q_values, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.update_epsilon()

            if (epoch + 1) % self.save_interval == 0:
                self.update_target_network()
                self.save_model()

            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        self.save_model()

    def update_epsilon(self):
        """Decays epsilon over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_max_info_gain_centroid(self):
        self.target_ddqn.eval()
        """Finds the centroid with the highest information gain."""
        network_input, _, sorted_centroid_record = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr
        )
        with torch.no_grad():
            output = self.target_ddqn(network_input.to(self.device))
        max_info_gain_centroid_idx = np.argmax(output.cpu().numpy())
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx

    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        """Predicts the best centroid based on the given robot position and orientation using the target network."""
        self.target_ddqn.eval()

        """Finds the centroid with the highest information gain."""
        network_input, _, sorted_centroid_record = self.prepare_input(
            robot_position, robot_orientation, centroid_records, info_gain_records
        )
        with torch.no_grad():
            output = self.target_ddqn(network_input.to(self.device))
        max_info_gain_centroid_idx = np.argmax(output.cpu().numpy())
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]

        return max_info_gain_centroid, max_info_gain_centroid_idx
