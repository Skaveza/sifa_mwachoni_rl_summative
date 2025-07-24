import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import TradingEnv
from implementation.rendering import save_gif

env = TradingEnv()
save_gif(env, filename='trading_env_random.gif')