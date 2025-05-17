# Deep Reinforcement Learning Coursework

This repository contains the coursework for the Deep Reinforcement Learning module. It is organized into three main sections:

## 1. Basic

In the Basic section, a 5x5 GridWorld environment is defined where an agent must navigate from the start state to a goal state while avoiding obstacles. The environment includes a deterministic state transition function and a reward function based on movement, collisions, and reaching the goal. The Q-learning algorithm is implemented with specified hyperparameters (α, γ, ε) and trained over multiple episodes. The agent’s performance is analyzed through reward and episode length trends. Parameter tuning is also conducted to identify the most effective learning configuration.

The results of this section are saved in the outputs folder.

## 2. Advanced

The Advanced section includes:

- **DQN Implementation**: A custom implementation of the Deep Q-Network algorithm.
- **RLlib Algorithm on Atari Environment**: Utilization of RLlib (a scalable RL library) to train agents in Atari environments.

The results and training logs for the **RLlib on Atari** experiments are logged and available on Weights & Biases at the following link:

👉 [WandB Project: Deep Reinforcement Learning](https://wandb.ai/anndischeh-univ-/Deep%20Reinforcement%20Learning)

## 3. Extras

Extra: Soft Actor-Critic (SAC) Training on CartPole

This section includes the implementation of a Soft Actor-Critic (SAC) algorithm for the classic CartPole-v1 environment from the Gym library. Unlike conventional approaches that use discrete algorithms for CartPole, this implementation uses a Gaussian policy with continuous outputs, which are then mapped to discrete actions.

This implementation includes the following components:

Definition of a Gaussian policy network with mean and log_std layers

Two independent Q-networks and their target networks for training stability

An experience replay buffer mechanism

Training using gradient descent and soft updates of target network weights

Automatic entropy tuning to balance exploration and exploitation

Plotting training rewards and losses after execution

✅ The results of this section are saved in: outputs

## Outputs

Outputs for all experiments (except for the RLlib Atari training which is stored on WandB) can be found in the `outputs/` directory of this repository.

## 📁 Project Structure
```
C:.
├── Advanced.py                         # Implementation of the advanced reinforcement learning algorithms and experiments.
├── Advanced_rllib.py                   # RLlib-based implementation and training on Atari environment.
├── Basic.py                           # Basic GridWorld environment and Q-learning implementation.
├── Extras.py                          # Extra experiments including Soft Actor-Critic (SAC) on CartPole.
├── requirements.txt                   # List of Python dependencies for the project.
└── outputs/                          # Folder containing all results and logs generated during training and evaluation.
    ├── Advanced_log.txt               # Training logs for Advanced experiments.
    ├── Advanced_Training Progress-Loss.png    # Training loss curve for Advanced experiments.
    ├── Advanced_Training Progress-Reward.png  # Training reward curve for Advanced experiments.
    ├── Basic_Best-Performing-Parameter.png    # Visualization of best parameters found in Basic experiments.
    ├── Basic_Learned-Policy-2.png     # Learned policy visualization for Basic experiments.
    ├── Basic_Learned-policy.png       # Another learned policy visualization for Basic experiments.
    ├── Basic_log.txt                  # Training logs for Basic experiments.
    ├── Basic_Q-Learning-Episode.png  # Q-learning episode performance plot.
    ├── episode_stats.png              # General episode statistics visualization.
    ├── Extras_log.txt                 # Training logs for Extras experiments.
    ├── Extra_Learned-Policy.png       # Learned policy visualization for Extras experiments.
    ├── Figure_1.png                   # Additional result figure.
    ├── Figure_2.png                   # Additional result figure.
    ├── Figure_3.png                   # Additional result figure.
    ├── Figure_4.png                   # Additional result figure.
    ├── training_results.png           # Summary plot of training results.
    └── Videos/                       # Folder containing recorded training episodes.
        ├── rl-video-episode-0.mp4    # Training episode video 0.
        ├── rl-video-episode-1.mp4    # Training episode video 1.
        └── rl-video-episode-2.mp4    # Training episode video 2.

```
