from environment.custom_env import TradingEnv
from stable_baselines3.common.monitor import Monitor
env = Monitor(TradingEnv())
total_reward = 0
episodes = 50
for _ in range(episodes):
    obs = env.reset()[0]
    done = False
    while not done:
        action = 15
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
baseline_reward = total_reward / episodes
print(f"Baseline Mean Reward: {baseline_reward:.2f}")