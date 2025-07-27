import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import TradingEnv
from stable_baselines3 import DQN
from implementation.rendering import Renderer
import imageio

def save_dqn_gif(env, model, filename="dqn_trading_new.gif", max_steps=100, fps=10):
    renderer = Renderer()
    frames = []
    state, _ = env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, truncated, _ = env.step(action)
        renderer.render(env)
        frames.append(renderer.get_frame())
        if done or truncated:
            break
    renderer.close()
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved as {filename}")

if __name__ == "__main__":
    env = TradingEnv()
    model_path = "models/dqn/dqn_trading_env"
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}.zip")
    model = DQN.load(model_path, env=env)
    save_dqn_gif(env, model, filename="dqn_trading_new.gif", max_steps=100, fps=10)