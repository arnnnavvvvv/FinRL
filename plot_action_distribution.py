import numpy as np
from stable_baselines3 import DQN
from envs.finance_env import FinanceEnv
import matplotlib.pyplot as plt
from collections import Counter

model = DQN.load("models/dqn_finance_final.zip")
env = FinanceEnv()
action_counts = Counter()

for ep in range(50):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        action_counts[int(action)] += 1

labels = ["Spend", "Save", "Invest"]
values = [action_counts[i] for i in range(3)]

plt.figure(figsize=(6,6))
plt.bar(labels, values)
plt.title("Agent Action Distribution (after training)")
plt.ylabel("Count")
plt.savefig("results/action_distribution.png")
print("✅ Saved: results/action_distribution.png")
