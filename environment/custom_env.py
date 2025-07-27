import gymnasium as gym
import numpy as np
from gymnasium import spaces
from implementation.rendering import Renderer

class TradingEnv(gym.Env):
    def __init__(self):
        super(TradingEnv, self).__init__()
        self.num_assets = 5
        self.initial_cash = 10000
        self.max_steps = 100
        self.max_shares = 100

        # State: [cash, shares (5), prices (5)]
        state_size = 1 + self.num_assets + self.num_assets
        self.observation_space = spaces.Box(
            low=np.array([0] + [0] * self.num_assets + [0] * self.num_assets),
            high=np.array([np.inf] + [self.max_shares] * self.num_assets + [np.inf] * self.num_assets),
            dtype=np.float32
        )
        # Actions: 3 per asset (hold, buy, sell) + no-action = 16
        self.action_space = spaces.Discrete(self.num_assets * 3 + 1)
        self.renderer = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = np.ones(self.num_assets) * 50  # Start with 50 shares per asset
        self.prices = np.random.uniform(50, 150, self.num_assets)
        self.portfolio_value = self.cash + np.sum(self.shares * self.prices)
        self.dividends = np.zeros(self.num_assets)  # For rendering
        self.moving_avg = self.prices.copy()  # For rendering
        self.compliance_flag = 1  # Dummy for rendering
        return self._get_state(), {}

    def _get_state(self):
        return np.concatenate([
            [self.cash], self.shares, self.prices
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        prev_portfolio_value = self.portfolio_value
        reward = 0.0
        self.dividends = np.zeros(self.num_assets)  # Reset for rendering

        # Simulate stock price movement
        self.prices += np.random.normal(0.6, 0.5, self.num_assets)
        self.prices = np.maximum(0, self.prices)
        self.moving_avg = 0.9 * self.moving_avg + 0.1 * self.prices  # For rendering

        # Dividends every 5 steps
        if self.step_count % 5 == 0:
            self.dividends = np.random.uniform(0, 1, self.num_assets) * self.shares
            self.cash += np.sum(self.dividends)
            reward += np.sum(self.dividends) / self.initial_cash

        # Process action
        if action < self.num_assets * 3:
            asset_idx = action // 3
            action_type = action % 3
            if action_type == 0:  # Hold
                reward += 0.01
            elif action_type == 1:  # Buy
                if self.cash >= self.prices[asset_idx]:
                    self.shares[asset_idx] += 1
                    self.cash -= self.prices[asset_idx]
            elif action_type == 2:  # Sell
                if self.shares[asset_idx] > 0:
                    self.shares[asset_idx] -= 1
                    self.cash += self.prices[asset_idx]
        elif action == self.num_assets * 3:  # No-action
            reward += 0.01

        # Update portfolio value
        self.portfolio_value = self.cash + np.sum(self.shares * self.prices)
        reward += (self.portfolio_value - prev_portfolio_value) / self.initial_cash

        # Termination conditions
        done = self.step_count >= self.max_steps or self.portfolio_value < 0.1 * self.initial_cash
        truncated = False
        info = {}
        return self._get_state(), reward, done, truncated, info

    def render(self, mode='human'):
        if self.renderer is None:
            self.renderer = Renderer()
        self.renderer.render(self)