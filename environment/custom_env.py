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

        # State: [cash, shares (5), prices (5), dividends (5), compliance_flag, moving_avg (5)]
        state_size = 1 + self.num_assets * 3 + 1 + self.num_assets
        self.observation_space = spaces.Box(
            low=np.array([0] + [0] * self.num_assets + [0] * self.num_assets + [0] * self.num_assets + [0] + [0] * self.num_assets),
            high=np.array([np.inf] + [self.max_shares] * self.num_assets + [np.inf] * self.num_assets + [np.inf] * self.num_assets + [1] + [np.inf] * self.num_assets),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_assets * 5 + 1)
        self.renderer = None
        self.reset()

        # Actions: 0=hold, 1=buy, 2=sell, 3=set stop-loss, 4=reinvest dividends per asset, plus 5=no action
        self.action_space = spaces.Discrete(self.num_assets * 5 + 1)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = np.zeros(self.num_assets)
        self.prices = np.random.uniform(50, 150, self.num_assets)
        self.dividends = np.zeros(self.num_assets)
        self.compliance_flag = 1
        self.moving_avg = self.prices.copy()
        self.stop_loss = np.zeros(self.num_assets)
        self.portfolio_value = self.cash + np.sum(self.shares * self.prices)
        return self._get_state(), {}

    def _get_state(self):
        return np.concatenate([
            [self.cash], self.shares, self.prices, self.dividends,
            [self.compliance_flag], self.moving_avg
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        prev_portfolio_value = self.portfolio_value
        reward = 0.0
        self.dividends = np.zeros(self.num_assets)

        # Simulate stock price movement (random walk)
        self.prices += np.random.normal(0, 1.0, self.num_assets)
        self.prices = np.maximum(0, self.prices)
        self.moving_avg = 0.9 * self.moving_avg + 0.1 * self.prices

        # Random dividends every 10 steps
        if self.step_count % 10 == 0:
            self.dividends = np.random.uniform(0, 1, self.num_assets) * self.shares
            self.cash += np.sum(self.dividends)
            reward += np.sum(self.dividends) / self.initial_cash

        # Process action
        if action < self.num_assets * 5 and self.compliance_flag == 1:
            asset_idx = action // 5
            action_type = action % 5
            if action_type == 0:  # Hold
                pass
            elif action_type == 1:  # Buy
                if self.cash >= self.prices[asset_idx]:
                    self.shares[asset_idx] += 1
                    self.cash -= self.prices[asset_idx]
                    reward -= 0.01  # Transaction cost
                else:
                    reward -= 0.1
                    self.compliance_flag = 0
            elif action_type == 2:  # Sell
                if self.shares[asset_idx] > 0:
                    self.shares[asset_idx] -= 1
                    self.cash += self.prices[asset_idx]
                    reward -= 0.01  # Transaction cost
                else:
                    reward -= 0.1
                    self.compliance_flag = 0
            elif action_type == 3:  # Set stop-loss
                self.stop_loss[asset_idx] = self.prices[asset_idx] * 0.95  # 5% below current price
            elif action_type == 4:  # Reinvest dividends
                if self.dividends[asset_idx] > 0:
                    shares_to_buy = self.dividends[asset_idx] // self.prices[asset_idx]
                    self.shares[asset_idx] += shares_to_buy
                    self.cash -= shares_to_buy * self.prices[asset_idx]
                    reward += 0.05 

        # Check stop-loss triggers
        for i in range(self.num_assets):
            if self.stop_loss[i] > 0 and self.prices[i] < self.stop_loss[i]:
                self.cash += self.shares[i] * self.prices[i]
                reward += 0.1 if self.prices[i] < self.moving_avg[i] else -0.1
                self.shares[i] = 0
                self.stop_loss[i] = 0

        # Update portfolio value
        self.portfolio_value = self.cash + np.sum(self.shares * self.prices)
        reward += (self.portfolio_value - prev_portfolio_value) / self.initial_cash

        # Random compliance check 
        if np.random.random() < 0.05:
            self.compliance_flag = 0
            reward -= 0.5
        else:
            self.compliance_flag = 1

        # Termination conditions
        done = self.step_count >= self.max_steps or self.portfolio_value < 0.1 * self.initial_cash
        truncated = False
        info = {}
        return self._get_state(), reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render the environment using the Renderer class."""
        if self.renderer is None:
            self.renderer = Renderer()
        self.renderer.render(self)