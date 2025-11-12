import numpy as np
import matplotlib.pyplot as plt
import os

rewards_path = "results/episode_rewards.npy"

if os.path.exists(rewards_path):
    rewards = np.load(rewards_path, allow_pickle=True)
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Progression")
    plt.grid(True)
    plt.savefig("results/training_rewards.png")
    print("✅ Saved: results/training_rewards.png")
else:
    print("❌ No rewards file found. Run train.py again with RewardLogger.")
