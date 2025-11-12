import numpy as np
from stable_baselines3 import DQN
from envs.finance_env import FinanceEnv
import matplotlib.pyplot as plt

model = DQN.load("models/dqn_finance_final.zip")
episodes = 5
plt.figure(figsize=(10,6))

for ep in range(episodes):
    env = FinanceEnv()
    obs, _ = env.reset()
    done = False
    assets = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        assets.append(env.total_assets)
    plt.plot(assets, label=f"Episode {ep+1}")

plt.xlabel("Month")
plt.ylabel("Total Assets")
plt.title("Total Asset Growth Across Multiple Episodes")
plt.legend()
plt.grid(True)
plt.savefig("results/multi_episode_growth.png")
print("✅ Saved: results/multi_episode_growth.png")
