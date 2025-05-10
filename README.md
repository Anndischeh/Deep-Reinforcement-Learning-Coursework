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


