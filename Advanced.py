import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt  #Import matplotlib
import sys


# Save all outputs as a file
log_file = open('.\outputs\Advanced_log.txt', "w")
sys.stdout = log_file
sys.stderr = log_file  # Redirect errors as well


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Value stream
        self.value_fc = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Value stream
        value_x = F.relu(self.value_fc(x))
        value = self.value(value_x)

        # Advantage stream
        advantage_x = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage_x)

        # Combine Value and Advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


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
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Hyperparameters
learning_rate = 1e-3
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
batch_size = 32
target_update_frequency = 100
num_episodes = 1000

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = DuelingDQN(state_size, action_size).to(device)
target_network = DuelingDQN(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())

# Optimizer
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Replay buffer
replay_buffer = ReplayBuffer(10000)

# Training loop
epsilon = epsilon_start
episode_rewards = []
losses = [] # Store loss values for analysis


#######################################################################
#                             Step 7: DQN Implementation            #
#######################################################################

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # Train if enough samples
        if len(replay_buffer) > batch_size:
            # Sample batch
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Convert to tensors
            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            # Current Q values
            current_q = q_network(states).gather(1, actions.unsqueeze(1))

            # Target Q values
            with torch.no_grad():
                next_q = target_network(next_states).max(1)[0]
                target_q = rewards + gamma * (1 - dones) * next_q

            # Compute loss
            loss = F.mse_loss(current_q.squeeze(), target_q)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item()) #Store the losses

    # Update target network periodically
    if episode % target_update_frequency == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    # Print progress every 50 episodes
    if episode % 50 == 0 or episode == num_episodes - 1:
        avg_reward = np.mean(episode_rewards[-50:]) if episode >= 50 else np.mean(episode_rewards)
        print(f"Episode: {episode:4d} | "
              f"Reward: {total_reward:6.1f} | "
              f"Avg Reward (last 50): {avg_reward:6.1f} | "
              f"Epsilon: {epsilon:.3f}")

env.close()  #Close env here before analysis to avoid errors


#######################################################################
#                Step 8: Analyse the results quantitatively            #
#######################################################################
def plot_rewards(rewards, window=50):
    """Plots the rewards and a moving average."""
    smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5, label='Episode Reward')
    plt.plot(range(window - 1, len(rewards)), smoothed_rewards, label=f'{window}-episode average')
    plt.title('Training Progress - Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_loss(losses, window=500):
    """Plots the training loss and a moving average."""
    smoothed_losses = np.convolve(losses, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.5, label='Training Loss')
    plt.plot(range(window - 1, len(losses)), smoothed_losses, label=f'{window}-step average')
    plt.title('Training Progress - Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()



def record_video(env_id, q_network, device, filename="dqn_agent.mp4", num_episodes=3):
    """Records a video of the trained agent."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda episode: True)

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

    env.close() #Close the environment
    print(f"Video recorded at: video/{filename}")

# ---Quantitative Analysis---
plot_rewards(episode_rewards)  # Plot rewards
plot_loss(losses)  # Plot loss

# ---Qualitative Analysis---
env_id = 'CartPole-v1'
record_video(env_id, q_network, device, filename="dqn_cartpole.mp4", num_episodes=3)  # Record a video

print("Training and Analysis Complete!")

# End of saving outputs as a file 
# Restore stdout and stderr to default (console)
log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("Experiment finished. All logs saved in '..\\outputs\\Advanced_log.txt'.")