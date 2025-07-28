import os
import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import TradingEnv

class DQNTrainer:
    def __init__(self):
        self.results = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
    def train(self):
        env = Monitor(TradingEnv())
        os.makedirs("models/dqn", exist_ok=True)
        
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-5,
            buffer_size=500000,
            learning_starts=10000,
            batch_size=128,
            tau=0.005,
            gamma=0.95,
            train_freq=4,
            gradient_steps=2,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            verbose=1
        )
        
        print("\n=== Training DQN ===")
        model.learn(total_timesteps=500000, log_interval=100)
        model.save("models/dqn/dqn_trading_env")
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        self.results.append(("DQN", mean_reward, std_reward))
        
        print(f"\nDQN Evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return model

if __name__ == "__main__":
    trainer = DQNTrainer()
    trainer.train()