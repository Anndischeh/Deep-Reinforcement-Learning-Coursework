import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import sys

# Save all outputs as a file
log_file = open('.\outputs\Extras_log.txt', "w")
sys.stdout = log_file
sys.stderr = log_file  # Redirect errors as well




# --- Hyperparameters ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
REPLAY_CAPACITY = 100000
BATCH_SIZE = 256
NUM_EPISODES = 500
TARGET_ENTROPY = None  # Auto-tune entropy

# --- Gaussian Policy Network ---
class GaussianPolicy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(GaussianPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_size, action_size)
        self.log_std_layer = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp for stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

# --- Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        # Ensure both state and action have the same number of dimensions
        if len(state.shape) < len(action.shape):
            state = state.unsqueeze(1)  # Add a dimension of 1 at position 1
        elif len(action.shape) < len(state.shape):
             action = action.unsqueeze(1)

        x = torch.cat([state, action], dim=1)
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # Use deque for efficient append and pop

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- Training Loop ---
def train_sac(env, num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, gamma=GAMMA, tau=TAU, replay_capacity=REPLAY_CAPACITY, alpha=ALPHA, target_entropy=TARGET_ENTROPY):
    state_size = env.observation_space.shape[0]
    action_size =  1 # Gaussian policy outputs single action

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_net = GaussianPolicy(state_size, action_size).to(device)
    q1_net = QNetwork(state_size, action_size).to(device)
    q2_net = QNetwork(state_size, action_size).to(device)
    target_q1_net = QNetwork(state_size, action_size).to(device)
    target_q2_net = QNetwork(state_size, action_size).to(device)

    target_q1_net.load_state_dict(q1_net.state_dict())
    target_q2_net.load_state_dict(q2_net.state_dict())

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_q1 = optim.Adam(q1_net.parameters(), lr=learning_rate)
    optimizer_q2 = optim.Adam(q2_net.parameters(), lr=learning_rate)

    replay_buffer = ReplayBuffer(replay_capacity)

    # Automatic entropy tuning
    if target_entropy is None:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    optimizer_alpha = optim.Adam([log_alpha], lr=learning_rate)

    alpha = torch.tensor(alpha, requires_grad=False).to(device)
    episode_rewards = []
    q1_losses = []
    q2_losses = []
    policy_losses = []
    alpha_losses = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        total_reward = 0
        done, truncated = False, False

        while not done and not truncated:
            action, log_prob, mean, log_std = policy_net.sample(state.unsqueeze(0))
            action = action.cpu().detach().numpy().flatten()

            # Scale and convert continuous action to discrete
            action = np.clip(action, -1, 1)
            discrete_action = 1 if action > 0 else 0

            next_state, reward, terminated, truncated, _ = env.step(discrete_action)
            done = terminated
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                state_batch = torch.tensor(np.array(states), dtype=torch.float32, device=device)
                action_batch = torch.tensor(np.array(actions), dtype=torch.float32, device=device) #Shape: [batch_size]
                reward_batch = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
                done_batch = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                with torch.no_grad():
                    next_actions, next_log_prob, _, _ = policy_net.sample(next_state_batch)
                    q1_next_target = target_q1_net(next_state_batch, next_actions)
                    q2_next_target = target_q2_net(next_state_batch, next_actions)
                    q_next_target = torch.min(q1_next_target, q2_next_target) - alpha * next_log_prob
                    q_target = reward_batch + gamma * (1 - done_batch) * q_next_target

                q1_pred = q1_net(state_batch, action_batch)
                q2_pred = q2_net(state_batch, action_batch)
                q1_loss = F.mse_loss(q1_pred, q_target)
                q2_loss = F.mse_loss(q2_pred, q_target)

                optimizer_q1.zero_grad()
                q1_loss.backward()
                torch.nn.utils.clip_grad_norm_(q1_net.parameters(), max_norm=0.5)  # Gradient clipping
                optimizer_q1.step()

                optimizer_q2.zero_grad()
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(q2_net.parameters(), max_norm=0.5)  # Gradient clipping
                optimizer_q2.step()

                new_actions, new_log_prob, _, _ = policy_net.sample(state_batch)
                q1_new = q1_net(state_batch, new_actions)
                q2_new = q2_net(state_batch, new_actions)
                q_new = torch.min(q1_new, q2_new)
                policy_loss = (alpha * new_log_prob - q_new).mean()

                optimizer_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)  # Gradient clipping
                optimizer_policy.step()

                alpha_loss = -(log_alpha * (new_log_prob + target_entropy).detach()).mean()

                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                alpha = torch.exp(log_alpha).item()

                # --- Soft Target Network Updates (Correct Tau Application) ---
                for target_param, param in zip(target_q1_net.parameters(), q1_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                for target_param, param in zip(target_q2_net.parameters(), q2_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                q1_losses.append(q1_loss.item())
                q2_losses.append(q2_loss.item())
                policy_losses.append(policy_loss.item())
                alpha_losses.append(alpha_loss.item())


        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Alpha: {alpha:.4f}")

    return policy_net, episode_rewards, q1_losses, q2_losses, policy_losses, alpha_losses
# --- Main ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1") # Replace with "Pong-v0" for Pong, but adapt action space in GaussianPolicy
    state_size = env.observation_space.shape[0]
    action_size =  1 # Gaussian policy outputs single action

    trained_policy_net, episode_rewards, q1_losses, q2_losses, policy_losses, alpha_losses = train_sac(env)

    # --- Plotting ---
    plt.figure(figsize=(16, 8))

    # Reward
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("SAC Training on CartPole - Reward")

    # Q Losses
    plt.subplot(2, 2, 2)
    plt.plot(q1_losses, label="Q1 Loss")
    plt.plot(q2_losses, label="Q2 Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Q-Network Losses")
    plt.legend()

    # Policy Loss
    plt.subplot(2, 2, 3)
    plt.plot(policy_losses, label="Policy Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Policy Network Loss")
    plt.legend()

    # Alpha Loss
    plt.subplot(2, 2, 4)
    plt.plot(alpha_losses, label="Alpha Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Alpha Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


    # --- Visualize Policy ---
    num_states_to_visualize = 20
    states = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num_states_to_visualize)
    with torch.no_grad():
        means, log_stds = [], []
        for state in states:
            state_tensor = torch.tensor(np.array([state, 0, 0, 0]), dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # Adapt observation
            mean, log_std = trained_policy_net(state_tensor)
            means.append(mean.item())
            log_stds.append(log_std.item())

    plt.figure(figsize=(10, 6))
    plt.plot(states, means, label="Mean Action")
    plt.plot(states, np.array(means) + np.exp(np.array(log_stds)), label="Mean + Std")
    plt.plot(states, np.array(means) - np.exp(np.array(log_stds)), label="Mean - Std")
    plt.xlabel("State (Cart Position)")
    plt.ylabel("Action (Cart Acceleration)")
    plt.title("Learned Policy Visualization")
    plt.legend()
    plt.show()
    
    
# End of saving outputs as a file 
# Restore stdout and stderr to default (console)
log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("Experiment finished. All logs saved in '..\\outputs\\Extras_log.txt'.")