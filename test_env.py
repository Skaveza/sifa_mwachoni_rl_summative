from environment.custom_env import TradingEnv
env = TradingEnv()
obs = env.reset()
print("Initial shares:", env.shares)