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
        self.base_transaction_cost = 0.001
        self.volatility_factor = 0.0005
        self.max_concentration = 0.4
        
        # State: [cash, shares (5), prices (5), moving_avg (5), event_flag]
        self.observation_space = spaces.Box(
            low=np.array([0] + [0]*self.num_assets + [0]*self.num_assets + [0]*self.num_assets + [0]),
            high=np.array([np.inf] + [self.max_shares]*self.num_assets + [np.inf]*self.num_assets + [np.inf]*self.num_assets + [1]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.num_assets * 3 + 1)  # 3 actions per asset + no-op
        self.renderer = None
        self.episode_total_reward = 0.0  # Added for rendering
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.cash = self.initial_cash
        self.shares = np.ones(self.num_assets) * 50
        self.prices = np.random.uniform(50, 150, self.num_assets)
        self.price_history = [self.prices.copy()]
        self.moving_avg = self.prices.copy()
        self.event_flag = 0
        self.dividends = np.zeros(self.num_assets)
        self.compliance_flag = 1
        self.episode_total_reward = 0.0  # Reset cumulative reward
        
        # Initialize correlation structure
        self.correlation_matrix = np.array([
            [1.0, 0.7, 0.3, -0.2, 0.1],
            [0.7, 1.0, 0.5, -0.1, 0.2],
            [0.3, 0.5, 1.0, 0.4, 0.3],
            [-0.2, -0.1, 0.4, 1.0, 0.1],
            [0.1, 0.2, 0.3, 0.1, 1.0]
        ])
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        
        return self._get_state(), {}

    def _get_state(self):
        return np.concatenate([
            [self.cash], 
            self.shares, 
            self.prices,
            self.moving_avg,
            [self.event_flag]
        ])

    def _apply_market_event(self):
        event_prob = 0.01
        if np.random.random() < event_prob:
            self.event_flag = 1
            event_type = np.random.choice(['earnings', 'regulation', 'macro'])
            impacts = {
                'earnings': (0.1, -0.05),
                'regulation': (-0.15, 0),
                'macro': (0.07, -0.07)
            }
            impact = np.random.uniform(*impacts[event_type], size=self.num_assets)
            self.prices *= (1 + impact)
        else:
            self.event_flag = 0
            
    def step(self, action):
        self.step_count += 1
        prev_value = self.portfolio_value
        
        # 1. Apply correlated price movements
        noise = np.random.normal(0.6, 0.5, self.num_assets)
        correlated_noise = self.cholesky @ noise
        self.prices += correlated_noise
        self.prices = np.maximum(1, self.prices)
        self.price_history.append(self.prices.copy())
        self.moving_avg = 0.9*self.moving_avg + 0.1*self.prices
        
        # 2. Apply market events
        self._apply_market_event()
        
        # 3. Process action with dynamic costs
        current_vol = np.std([p[-1] for p in self.price_history[-20:]]) if len(self.price_history) > 20 else 0
        dynamic_cost = self.base_transaction_cost + current_vol * self.volatility_factor
        
        reward = 0
        if action < self.num_assets * 3:
            asset_idx = action // 3
            action_type = action % 3
            
            if action_type == 1:  # Buy
                cost = self.prices[asset_idx] * (1 + dynamic_cost)
                if self.cash >= cost:
                    self.shares[asset_idx] += 1
                    self.cash -= cost
                    reward -= dynamic_cost
            elif action_type == 2:  # Sell
                revenue = self.prices[asset_idx] * (1 - dynamic_cost)
                if self.shares[asset_idx] > 0:
                    self.shares[asset_idx] -= 1
                    self.cash += revenue
                    reward -= dynamic_cost
        
        # 4. Apply concentration penalty
        port_values = self.shares * self.prices
        weights = port_values / (self.portfolio_value + 1e-8)
        if np.any(weights > self.max_concentration):
            excess = np.sum(weights[weights > self.max_concentration] - self.max_concentration)
            reward -= excess * 0.1
            
        # 5. Calculate returns
        reward += (self.portfolio_value - prev_value) / self.initial_cash
        
        # 6. Update cumulative reward
        self.episode_total_reward += reward
        
        # 7. Termination
        done = (self.step_count >= self.max_steps) or (self.portfolio_value < 0.1*self.initial_cash)
        return self._get_state(), reward, done, False, {}

    @property
    def portfolio_value(self):
        return self.cash + np.sum(self.shares * self.prices)

    def render(self, mode='human'):
        if self.renderer is None:
            self.renderer = Renderer()
        self.renderer.render(self)