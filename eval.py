import numpy as np
from stable_baselines3 import DQN
from envs.finance_env import FinanceEnv
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

model = DQN.load("models/dqn_finance_final.zip")

balances, assets = [], []
for ep in range(10):
    env = FinanceEnv()
    obs, _ = env.reset()
    done = False
    episode_assets = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        episode_assets.append(env.total_assets)
    assets.append(episode_assets)
    balances.append(env.balance)

mean_curve = np.mean([np.pad(a, (0, max(map(len, assets))-len(a)), constant_values=a[-1]) for a in assets], axis=0)

plt.plot(mean_curve)
plt.xlabel("Month")
plt.ylabel("Average Total Assets")
plt.title("Agent’s Portfolio Growth Over Time")
plt.savefig("results/portfolio_curve.png")
print("✅ Evaluation complete! See 'results/portfolio_curve.png'")
