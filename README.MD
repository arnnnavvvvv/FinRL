# 💰 FinRL: An Adaptive Financial Advisor using Reinforcement Learning (DQN)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Framework-Stable%20Baselines3-orange?logo=openai&logoColor=white">
  <img src="https://img.shields.io/badge/Type-Reinforcement%20Learning-green">
  <img src="https://img.shields.io/badge/Status-Research%20Project-success">
</p>

> **FinRL** is an intelligent Reinforcement Learning (RL) agent that learns how to manage personal finances — deciding *when to spend, save, or invest* — by interacting with a simulated financial environment.  
> Using **Deep Q-Learning (DQN)**, the model discovers balanced strategies that maximize long-term wealth while minimizing stress and maintaining satisfaction.

---

## 🧠 Project Overview

Traditional budgeting apps rely on static rules or manual user input. FinSage takes a step further — it **learns optimal financial behavior dynamically** through trial and feedback.

At every time step (month), the agent observes:
- 💵 **Balance, Income, Expenses**
- 📈 **Market Return**
- 📊 **Inflation**
- ⚖️ **Risk Tolerance**
- 🎯 **Goal Progress**

and decides between:
> **Spend (0)**, **Save (1)**, or **Invest (2)**  

---

## ⚙️ System Architecture
<p align="center">
  <img src="architecture.png" width="500px">
</p>

**Environment** simulates realistic financial fluctuations (income, inflation, returns).  
**Agent** learns via Deep Q-Learning with experience replay and target networks.  
**Reward function** balances long-term growth and psychological satisfaction.

---

## 📈 Reward Function

$$
R_t = \alpha G_t + \beta S_t - \gamma F_t
$$

Where:
- \( G_t \): Portfolio Growth  
- \( S_t \): User Satisfaction  
- \( F_t \): Financial Stress  

Default weights:
\[
\alpha = 0.6, \quad \beta = 0.3, \quad \gamma = 0.1
\]

These can be **personalized** per user to simulate different financial personalities.

---

## 🧩 Key Components

| Component | Description |
|------------|--------------|
| `envs/finance_env.py` | Custom Gym environment (financial simulator) |
| `train.py` | Trains the DQN agent |
| `eval.py` | Evaluates trained policy and generates plots |
| `plot_training_rewards.py` | Visualizes reward progression |
| `plot_action_distribution.py` | Analyzes action frequency |
| `plot_multiple_episodes.py` | Plots multi-run wealth trajectories |
| `results/` | Contains training plots and evaluation figures |
| `models/` | Contains saved model checkpoints |
| `Paper/` | IEEE-format conference paper |

---

## 🧪 Training Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Start training
python train.py

# Evaluate trained agent
python eval.py
