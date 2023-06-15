import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()

        # Feature layer extracts the common features from the input state
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 128),  # Fully connected layer
            nn.ReLU()  # Activation function
        )

        # Value stream estimates the state value
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(64, 1)  # Fully connected layer with single output unit
        )

        # Advantage stream estimates the advantages of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),  # Fully connected layer
            nn.ReLU(),  # Activation function
            # Fully connected layer with output units matching the number of actions
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        """
        The value stream captures the value of being in a particular state, which represents the expected return or utility of that state. It helps the agent understand how good or promising a state is on its own, regardless of the specific action taken.
        The advantage stream measures the advantage of each action compared to the average value of all actions in that state. By subtracting the mean advantage from the advantages of each action, the agent can assess the relative importance or benefit of choosing one action over another in a given state.
        """
        # Pass the input state through the feature layer
        features = self.feature_layer(x)

        # Compute the state value by passing the features through the value stream
        values = self.value_stream(features)

        # Compute the advantages of each action by passing the features through the advantage stream
        advantages = self.advantage_stream(features)

        # Combine the value and advantages to obtain the Q-values
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


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


class DuelingDQNAgent:
    def __init__(self, gazebo_env, gamma, learning_rate, tau, epsilon,
                 save_interval, epochs, batch_size, penalty,robot_post_arr, robot_orie_arr, centr_arr, info_arr, best_centr_arr):
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
        self.tau = tau
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty

        self.gazebo_env = gazebo_env
        self.filepath = f"/home/kenji_leong/explORB-SLAM-RL/src/decision_maker/src/python/RL/models/{gazebo_env}/dueling_dqn_{self.epochs}.pth"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dones = None

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(1000)

        # Initialize the Dueling DQN network
        self.initialize_dueling_dqn()

    def prepare_input(self, robot_pos, robot_orie, centr, info):
        robot_position = torch.tensor(robot_pos, dtype=torch.float32)
        robot_orientation = torch.tensor(robot_orie, dtype=torch.float32)
        centroid_record = torch.tensor(centr, dtype=torch.float32)
        info_gain_record = torch.tensor(info, dtype=torch.float32)

        robot_state = torch.cat((robot_position, robot_orientation))
        combined_data = torch.cat((centroid_record, info_gain_record), dim=1)
        sorted_data = combined_data[combined_data[:, -
                                                  1].argsort(descending=True)]

        sorted_centroid_record = sorted_data[:, :-1]
        sorted_info_gain_record = sorted_data[:, -1]

        network_input = torch.cat(
            (robot_state, sorted_centroid_record.flatten(), sorted_info_gain_record.flatten()))
        input_size = network_input.numel()
        network_input = network_input.reshape(1, input_size)
        output_size = sorted_centroid_record.shape[0]

        return network_input, output_size, sorted_centroid_record

    def initialize_dueling_dqn(self):
        network_input, output_size, _ = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr)
        self.dqn = DuelingDQN(
            network_input.shape[1], output_size).to(self.device)
        self.target_dqn = DuelingDQN(
            network_input.shape[1], output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.dqn.parameters())
        self.save_model()

    def update_target_network(self):
        for target_param, param in zip(self.target_dqn.parameters(), self.dqn.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        self.save_model()

    def save_model(self):
        torch.save(self.target_dqn.state_dict(), self.filepath)

    def load_model(self):
        self.target_dqn.load_state_dict(torch.load(
            self.filepath, map_location=self.device))
        self.target_dqn.eval()

    def select_action(self, state, output_size):
        if random.random() < self.epsilon:
            action = random.randint(0, output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.dqn(state)
                selected_q_values = q_values[:, :5]
                action = selected_q_values.argmax(dim=1).item()
        return action

    def calculate_reward(self):
        predicted_centroid, _ = self.get_max_info_gain_centroid()
        target_centroid = torch.tensor(
            self.best_centr_arr, dtype=torch.float32, device=self.device)
        match = torch.all(torch.eq(predicted_centroid, target_centroid))
        if match:
            reward = 1
        else:
            reward = 0
        return reward, predicted_centroid

    def train(self):
        zero_centroid = torch.tensor([0.0, 0.0], device=self.device)
        self.dones = torch.zeros((1,), device=self.device)

        self.replay_buffer = ReplayBuffer(10000)
        for epoch in range(self.epochs):
            for i in range(len(self.robot_post_arr2)-1):
                self.load_model()
                network_input, output_size, _ = self.prepare_input(
                    self.robot_post_arr2[i], self.robot_orie_arr2[i], self.centr_arr2[i], self.info_arr2[i])

                actions = self.select_action(network_input, output_size)

                rewards, predicted_centroid = self.calculate_reward()

                next_state, _, _ = self.prepare_input(
                    self.robot_post_arr2[i+1], self.robot_orie_arr2[i+1], self.centr_arr2[i+1], self.info_arr2[i+1])
                done = torch.all(torch.eq(predicted_centroid,
                                 torch.tensor([0.0, 0.0], device=self.device)))

                self.replay_buffer.push(
                    network_input, actions, rewards, next_state, done)

                if len(self.replay_buffer) >= self.batch_size:
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

                    if torch.all(torch.eq(predicted_centroid, zero_centroid)):
                        loss += self.penalty

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.update_epsilon(epoch)

            if self.epochs % self.save_interval == 0:
                self.update_target_network()
                self.save_model()

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    def update_epsilon(self, epoch):
        self.epsilon = max(0.01, 0.1 - (0.09 / self.epochs) * epoch)

    def get_max_info_gain_centroid(self):
        self.load_model()
        network_input, _, sorted_centroid_record = self.prepare_input(
            self.robot_post_arr, self.robot_orie_arr, self.centr_arr, self.info_arr)
        with torch.no_grad():
            output = self.target_dqn(network_input.to(self.device))
        max_info_gain_centroid_idx = np.argmax(output.cpu().numpy())
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]
        return max_info_gain_centroid, max_info_gain_centroid_idx

    def predict_centroid(self, robot_position, robot_orientation, centroid_records, info_gain_records):
        self.load_model()
        network_input, _, sorted_centroid_record = self.prepare_input(
            robot_position, robot_orientation, centroid_records, info_gain_records)
        with torch.no_grad():
            output = self.target_dqn(network_input.to(self.device))
        max_info_gain_centroid_idx = np.argmax(output.cpu().numpy())
        max_info_gain_centroid_idx = max_info_gain_centroid_idx % sorted_centroid_record.shape[
            0]
        max_info_gain_centroid = sorted_centroid_record[max_info_gain_centroid_idx]
        return max_info_gain_centroid, max_info_gain_centroid_idx


