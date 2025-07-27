import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import TradingEnv
import numpy as np

np.random.seed(42)
env = TradingEnv()
env = Monitor(env)

# Train REINFORCE
reinforce_model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.0,
    verbose=1
)
reinforce_model.learn(total_timesteps=100000, log_interval=100)
os.makedirs("models/pg", exist_ok=True)
reinforce_model.save("models/pg/reinforce_trading_env")
mean_reward, std_reward = evaluate_policy(reinforce_model, env, n_eval_episodes=10)
print(f"REINFORCE Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train PPO
ppo_model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)
ppo_model.learn(total_timesteps=100000, log_interval=100)
ppo_model.save("models/pg/ppo_trading_env")
mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=10)
print(f"PPO Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train A2C
a2c_model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    verbose=1
)
a2c_model.learn(total_timesteps=100000, log_interval=100)
a2c_model.save("models/pg/a2c_trading_env")
mean_reward, std_reward = evaluate_policy(a2c_model, env, n_eval_episodes=10)
print(f"A2C Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")