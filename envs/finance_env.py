import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FinanceEnv(gym.Env):
    """
    Custom RL environment for financial decision-making.
    Actions: 0=Spend, 1=Save, 2=Invest
    """

    def __init__(self,
                 episode_months=36,
                 init_balance=5000.0,
                 salary_mean=2000.0,
                 expense_mean=1500.0,
                 risk_tolerance=0.5,
                 goal_amount=50000.0,
                 alpha=0.6, beta=0.3, gamma=0.1,
                 seed=None):
        super().__init__()
        self.episode_months = episode_months
        self.init_balance = init_balance
        self.salary_mean = salary_mean
        self.expense_mean = expense_mean
        self.risk_tolerance = risk_tolerance
        self.goal_amount = goal_amount
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._rng = np.random.default_rng(seed)

        # Observation: 7 features
        low = np.array([0., 0., 0., -1.0, -0.1, 0.0, 0.0], dtype=np.float32)
        high = np.array([1e6, 1e5, 1e5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.month = 0
        self.balance = self.init_balance
        self.invested = 0.0
        self.total_assets = self.balance
        self.prev_total = self.total_assets
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        market_return = self._rng.normal(0.005, 0.03)
        inflation = self._rng.normal(0.002, 0.003)
        income = self.salary_mean * (1.0 + self._rng.normal(0, 0.05))
        expenses = self.expense_mean * (1.0 + self._rng.normal(0, 0.05))

        if action == 0: spend, save, invest = 0.8, 0.1, 0.1
        elif action == 1: spend, save, invest = 0.6, 0.3, 0.1
        else: spend, save, invest = 0.6, 0.1, 0.3 * self.risk_tolerance

        available = self.balance + income - expenses
        if available < 0:
            liquidation = min(-available, self.invested)
            self.invested -= liquidation
            available += liquidation

        spend_amt = available * spend
        save_amt = available * save
        invest_amt = available * invest

        self.balance = self.balance + income - expenses - spend_amt - invest_amt + save_amt
        self.invested += invest_amt
        self.invested *= (1 + market_return)
        self.balance *= (1 - inflation)
        self.total_assets = self.balance + self.invested

        # Reward function
        portfolio_growth = self.total_assets - self.prev_total
        goal_progress = min(1.0, self.total_assets / self.goal_amount)
        satisfaction = 0.5 * (spend_amt / (1 + spend_amt)) + 0.5 * goal_progress
        stress = (1 - self.balance / max(1, self.total_assets)) * (1 - self.risk_tolerance)
        reward = self.alpha * portfolio_growth + self.beta * satisfaction - self.gamma * stress

        # Normalize reward to keep training stable
        reward = np.clip(reward / 1000.0, -10.0, 10.0)

        self.prev_total = self.total_assets
        self.month += 1
        done = self.month >= self.episode_months

        # ✅ Define info dict (this is what was missing before)
        info = {
            "total_assets": self.total_assets,
            "balance": self.balance,
            "invested": self.invested,
            "month": self.month
        }

        # ✅ Add episode info so SB3 callbacks log it properly
        if done:
            info["episode"] = {"r": float(reward), "l": self.month}

        obs = self._get_obs()
        return obs, reward, done, False, info

    def _get_obs(self):
        return np.array([
            self.balance, self.salary_mean, self.expense_mean,
            0.0, 0.0, self.risk_tolerance,
            min(1.0, self.total_assets / self.goal_amount)
        ], dtype=np.float32)

    def render(self):
        print(f"Month {self.month}: Total assets = {self.total_assets:.2f}")
