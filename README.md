# Deep Reinforcement Learning Coursework
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0-orange)
![Reinforcement Learning](https://img.shields.io/badge/field-Reinforcement%20Learning-purple)
![Gym](https://img.shields.io/badge/env-OpenAI%20Gym-yellow)
![RLlib](https://img.shields.io/badge/framework-RLlib-red)
![Weights & Biases](https://img.shields.io/badge/tracking-W%26B-black)
![License](https://img.shields.io/badge/license-MIT-green)


# Deep Reinforcement Learning Coursework

This repository contains my coursework for the **Deep Reinforcement Learning** module.
The project explores classical and deep RL algorithms across environments of increasing complexity,
from tabular GridWorld to high-dimensional visual Atari environments.

The implementation covers **Q-learning, DQN (with improvements), PPO, and SAC**, with
systematic evaluation and visualization of learning behavior.

---

## 🔍 Project Overview

The coursework is organized into three main sections:

### 1️⃣ Basic — Tabular Reinforcement Learning

- Implemented a **5×5 GridWorld** environment with obstacles
- Defined deterministic state transition and reward functions
- Trained an agent using **Q-learning**
- Tuned learning rate (α), discount factor (γ), and exploration rate (ε)
- Evaluated learning using:
  - Cumulative reward per episode
  - Episode length trends
  - Learned policy heatmaps

📊 Results and visualizations are stored in the `outputs/` directory.

---

### 2️⃣ Advanced — Deep Reinforcement Learning

This section extends learning to environments that require function approximation.

- **Deep Q-Network (DQN)**
  - Implemented neural-network-based Q-learning
  - Used experience replay for training stability

- **RLlib on Atari (Pong)**
  - Trained an agent on a high-dimensional pixel-based environment
  - Applied policy-gradient-based RL using RLlib
  - Tracked performance metrics such as episode return, length, loss, and KL divergence

📈 Training logs and metrics for Atari experiments are available on Weights & Biases:

👉 https://wandb.ai/anndischeh-univ-/Deep%20Reinforcement%20Learning

---

### 3️⃣ Extras — Soft Actor-Critic (SAC)

- Implemented **Soft Actor-Critic (SAC)** on CartPole-v1
- Used:
  - Gaussian policy network
  - Twin Q-networks with target networks
  - Experience replay buffer
  - Automatic entropy tuning
- Mapped continuous policy outputs to discrete actions
- Analyzed reward curves and loss evolution

✅ Results and plots are saved in `outputs/`.

---

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

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
### 2️⃣ Run Experiments
#### 🔹 Basic (Q-learning - GridWorld):

```bash
python basic.py
```
#### 🔹 Advanced (Deep Q-Network (DQN)):
```bash
python Advanced.py
```
#### 🔹Advanced (RLlib Atari Training):
This script supports configurable hardware resources and distributed training parameters.
```bash
python Advanced_rllib.py \
  --num-cpus=2 \
  --num-env-runners=1 \
  --num-learners=1 \
  --num-gpus-per-learner=1 \
  --framework=torch
```
Notes:
- Adjust CPU/GPU values based on your available hardware
- Uses RLlib for scalable reinforcement learning
- Results and training logs are tracked via Weights & Biases
  
### 🔹 Extras - Soft Actor-Critic on CarPole (SAC):
```bash
python Extras.py
```
------------
## 📌 Key Takeaways

- Implemented and compared tabular, value-based, and policy-gradient RL algorithms
- Gained hands-on experience with:
  - Exploration–exploitation trade-offs
  - Stability improvements in deep RL
  - High-dimensional visual environments
- Ensured experiment reproducibility and visualization using Weights & Biases

## 📚 References

- Sutton & Barto — Reinforcement Learning: An Introduction
- OpenAI Gym
- RLlib Documentation


*Note: The command uses `python`, not `!python`, unless you are executing within a Jupyter notebook.*

