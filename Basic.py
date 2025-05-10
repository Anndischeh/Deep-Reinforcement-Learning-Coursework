import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import sys

# Save all outputs as a file
log_file = open('.\outputs\Basic_log.txt', "w")
sys.stdout = log_file
sys.stderr = log_file  # Redirect errors as well


# 1. Define the Environment and Problem (Grid World)
class GridWorld:
    def __init__(self, grid_size=5, goal_state=(4, 4), obstacle_states=None):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.obstacle_states = obstacle_states if obstacle_states else []  #List of tuples.
        self.state = (0, 0)  # Start state
        self.action_space = [0, 1, 2, 3]  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = grid_size * grid_size
        self.max_steps = 100  # Maximum steps per episode
        self.current_step = 0

    def reset(self):
        self.state = (0, 0)
        self.current_step = 0
        return self.state_to_index(self.state)


    def state_to_index(self, state):
        return state[0] * self.grid_size + state[1]

    def index_to_state(self, index):
        return (index // self.grid_size, index % self.grid_size)

    def step(self, action):
        self.current_step += 1
        row, col = self.state
        if action == 0:  # Up
            new_state = (max(row - 1, 0), col)
        elif action == 1:  # Down
            new_state = (min(row + 1, self.grid_size - 1), col)
        elif action == 2:  # Left
            new_state = (row, max(col - 1, 0))
        elif action == 3:  # Right
            new_state = (row, min(col + 1, self.grid_size - 1))
        else:
            raise ValueError("Invalid action")

        # Handle obstacles
        if new_state in self.obstacle_states:
            new_state = self.state  # Stay in the same state


        self.state = new_state
        reward = self.get_reward(new_state)
        done = self.is_done(new_state) or self.current_step >= self.max_steps
        return self.state_to_index(new_state), reward, done, {}

    def get_reward(self, state):
        if state == self.goal_state:
            return 10
        elif state in self.obstacle_states:
            return -10
        else:
            return -0.1  # Small negative reward for each step

    def is_done(self, state):
        return state == self.goal_state

    def render(self):  # Simple text-based rendering
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.state:
                    print("A", end=" ")  # Agent
                elif (i, j) == self.goal_state:
                    print("G", end=" ")  # Goal
                elif (i, j) in self.obstacle_states:
                    print("X", end=" ")  # Obstacle
                else:
                    print(".", end=" ")  # Empty cell
            print()
        print("-" * (self.grid_size * 2))



# 2. Q-learning Parameters and Policy
def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))


def choose_action(state, q_table, epsilon):
    if random.random() < epsilon:
        return random.choice(range(q_table.shape[1]))  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit


# 3. Q-learning Algorithm
def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000, epsilon_decay_rate=0.99, min_epsilon=0.01):
    q_table = initialize_q_table(env.observation_space, len(env.action_space))
    rewards_per_episode = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state, reward, done, _ = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(epsilon * epsilon_decay_rate, min_epsilon)  # Decay epsilon
        rewards_per_episode.append(total_reward)
        episode_lengths.append(steps)

    return q_table, rewards_per_episode, episode_lengths


# 4. Run Experiment and Represent Performance
def run_experiment(env, alpha, gamma, epsilon, num_episodes):
    q_table, rewards_per_episode, episode_lengths = q_learning(env, alpha, gamma, epsilon, num_episodes)
    return q_table, rewards_per_episode, episode_lengths


def plot_results(rewards_per_episode, episode_lengths, title="Q-learning Performance"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Rewards
    axs[0].plot(rewards_per_episode)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Cumulative Reward")
    axs[0].set_title("Cumulative Reward per Episode")
    axs[0].grid(True)

    # Plot Episode Lengths
    axs[1].plot(episode_lengths)
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Episode Length (Steps)")
    axs[1].set_title("Episode Length per Episode")
    axs[1].grid(True)

    fig.suptitle(title)
    plt.subplots_adjust(top=0.9)  # Adjust title spacing
    plt.show()


def visualize_policy(q_table, env):
    policy = np.argmax(q_table, axis=1)  # Get best action for each state
    grid_policy = policy.reshape((env.grid_size, env.grid_size)) # Convert policy into the shape of the grid world.

    plt.figure(figsize=(8, 6))
    sns.heatmap(grid_policy, annot=True, fmt="d", cmap="viridis", linewidths=.5, linecolor='black')
    plt.title("Learned Policy (Action IDs)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


# 5. Experiment with Different Parameters
def parameter_sweep(env, alphas, gammas, epsilons, num_episodes=500):
    results = []
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                print(f"Running experiment with alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                q_table, rewards_per_episode, episode_lengths = run_experiment(env, alpha, gamma, epsilon, num_episodes)
                results.append({
                    "alpha": alpha,
                    "gamma": gamma,
                    "epsilon": epsilon,
                    "rewards": rewards_per_episode,
                    "lengths": episode_lengths,
                    "q_table": q_table
                })
    return results

# 6. Analyze Results
def analyze_results(results):
    #Example: comparing based on the average reward over last 100 episodes.
    best_result = None
    best_avg_reward = float('-inf')

    for result in results:
        avg_reward = np.mean(result["rewards"][-100:])
        print(f"Alpha: {result['alpha']}, Gamma: {result['gamma']}, Epsilon: {result['epsilon']}, Avg Reward (Last 100): {avg_reward}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_result = result

    print("\nBest Performing Parameters:")
    print(f"Alpha: {best_result['alpha']}, Gamma: {best_result['gamma']}, Epsilon: {best_result['epsilon']}")
    plot_results(best_result["rewards"], best_result["lengths"], title="Best Performing Parameters")
    return best_result["q_table"]

# --- Main Execution ---
if __name__ == "__main__":
    # Define Environment
    env = GridWorld(grid_size=5, goal_state=(4, 4), obstacle_states=[(1, 1), (2, 3)])
    #Run the code for rendering the enviroment before the training.
    print("Initial Environment:")
    env.render()

    #Single Run (Baseline)
    print("\nRunning a single Q-learning experiment...")
    q_table, rewards_per_episode, episode_lengths = run_experiment(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=1000)
    plot_results(rewards_per_episode, episode_lengths, title="Q-learning Performance (Single Run)")
    visualize_policy(q_table, env)

    # Parameter Sweep
    print("\nRunning parameter sweep...")
    alphas = [0.1, 0.3]
    gammas = [0.7, 0.9]
    epsilons = [0.1, 0.3]
    results = parameter_sweep(env, alphas, gammas, epsilons, num_episodes=500)

    # Analyze Results
    print("\nAnalyzing results...")
    best_q_table = analyze_results(results)
    print("Visualizing the policy obtained from the best parameters")
    visualize_policy(best_q_table, env) #Display the policy using the best parameters

    print("\nDone!")
    
# End of saving outputs as a file 
# Restore stdout and stderr to default (console)
log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print("Experiment finished. All logs saved in '..\\outputs\\Basic_log.txt'.")
