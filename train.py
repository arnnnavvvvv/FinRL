import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from envs.finance_env import FinanceEnv


# -----------------------------
# Callback: Logs episode rewards
# -----------------------------
class RewardLogger(BaseCallback):
    def __init__(self, log_dir="results/", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self):
        rewards_path = os.path.join(self.log_dir, "episode_rewards.npy")
        np.save(rewards_path, np.array(self.episode_rewards))
        print(f"✅ Saved episode rewards to {rewards_path}")


# -----------------------------
# Environment factory
# -----------------------------
def make_env():
    def _init():
        return FinanceEnv(episode_months=48, init_balance=10000.0)
    return _init


# -----------------------------
# Main training routine
# -----------------------------
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    env = DummyVecEnv([make_env() for _ in range(4)])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        verbose=1
    )

    checkpoint_cb = CheckpointCallback(save_freq=40000, save_path="./models/", name_prefix="dqn_finance")
    reward_logger = RewardLogger()

    print("\n🚀 Training started...\n")

    # Training loop — same rhythm as before (40000 steps intervals)
    total_timesteps = 200000
    intervals = [40000, 80000, 120000, 160000, 200000]

    for step in intervals:
        print(f"\n📈 Training up to {step} timesteps...\n")
        model.learn(total_timesteps=step, callback=[checkpoint_cb, reward_logger], reset_num_timesteps=False)
        model.save(f"models/dqn_finance_{step}")
        print(f"💾 Model checkpoint saved at {step} timesteps.")

    print("\n✅ Training complete!\n")

    # Save final model and reward data
    model.save("models/dqn_finance_final")
    np.save("results/episode_rewards.npy", np.array(reward_logger.episode_rewards))
    print("""
    ===========================================
    🎯 TRAINING FINISHED SUCCESSFULLY
    Outputs:
      ✔ models/dqn_finance_final.zip
      ✔ results/episode_rewards.npy
      ✔ models/dqn_finance_40000.zip
      ✔ models/dqn_finance_80000.zip
      ✔ models/dqn_finance_120000.zip
      ✔ models/dqn_finance_160000.zip
      ✔ models/dqn_finance_200000.zip
    Next steps:
      1. python eval.py
      2. python plot_training_rewards.py
      3. python plot_action_distribution.py
      4. python plot_multiple_episodes.py
    ===========================================
    """)

