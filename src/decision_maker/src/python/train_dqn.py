import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

# Initialize hyperparameters
state_dim = 4  # Dimension of the state space
action_dim = 2  # Dimension of the action space
batch_size = 32  # Batch size for training
learning_rate = 0.001  # Learning rate
gamma = 0.99  # Discount factor

# Create Q-network and target network
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Create optimizer
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Create replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select action using epsilon-greedy policy
        epsilon = max(0.01, 0.08 - 0.01 * episode)
        if np.random.rand() < epsilon:
            action = np.random.randint(action_dim)
        else:
            q_values = q_network(torch.FloatTensor(state))
            action = q_values.argmax().item()

        # Take action and observe next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Store experience in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Sample a minibatch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        # Compute Q-values and target Q-values
        q_values = q_network(torch.FloatTensor(state_batch))
        next_q_values = target_network(torch.FloatTensor(next_state_batch))
        target_q_values = reward_batch + gamma * next_q_values.max(1)[0] * (1 - done_batch)

        # Compute loss and update Q-network
        loss = nn.MSELoss()(q_values.gather(1, torch.LongTensor(action_batch).unsqueeze(1)), target_q_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        target_network.load_state_dict(q_network.state_dict())

        # Update state and total reward
        state = next_state
        total_reward += reward

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Test the trained Q-network
state = env.reset()
done = False
total_reward = 0

while not done:
    q_values = q_network(torch.FloatTensor(state))
    action = q_values.argmax().item()
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state

print(f"Test Total Reward: {total_reward}")
