import os
import sys
import random
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import TradingEnv
import numpy as np

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Initialize environment
env = TradingEnv()
env = Monitor(env)

# Load existing model
model_path = "models/dqn/dqn_trading_env"
if os.path.exists(model_path + ".zip"):
    print("Loading existing model...")
    model = DQN.load(model_path, env=env)
    # Update hyperparameters
    model.learning_rate = 2e-5
    model.buffer_size = 250000
    model.learning_starts = 5000
    model.batch_size = 64
    model.tau = 0.005
    model.gamma = 0.95
    model.target_update_interval = 100
    model.exploration_fraction = 0.05
    model.exploration_final_eps = 0.01
else:
    print("Creating new model...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-5,
        buffer_size=250000,
        learning_starts=5000,
        batch_size=64,
        tau=0.005,
        gamma=0.95,
        train_freq=4,
        target_update_interval=100,
        exploration_fraction=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )

# Train
model.learn(total_timesteps=1000000, log_interval=100, reset_num_timesteps=False)

# Save
os.makedirs("models/dqn", exist_ok=True)
model.save("models/dqn/dqn_trading_env")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print(f"DQN Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")