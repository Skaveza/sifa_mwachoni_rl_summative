import os
import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import TradingEnv

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.q_losses = []
        self.current_rewards = []
    
    def _on_step(self) -> bool:
        self.current_rewards.append(self.locals["rewards"])
        if self.locals["dones"]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        if "loss" in self.locals:
            self.q_losses.append(self.locals["loss"])
        return True

class DQNTrainer:
    def __init__(self):
        self.results = []
        
    def train(self):
        env = Monitor(TradingEnv())
        os.makedirs("models/dqn", exist_ok=True)
        
        callback = TrainingCallback()
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
        model.learn(total_timesteps=500000, log_interval=100, callback=callback)
        model.save("models/dqn/dqn_trading_env")
        
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
        self.results.append(("DQN", mean_reward, std_reward))
        
        print(f"\nDQN Evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return model, callback.episode_rewards, callback.q_losses

if __name__ == "__main__":
    trainer = DQNTrainer()
    model, episode_rewards, q_losses = trainer.train()
    np.save("results/dqn_episode_rewards.npy", episode_rewards)
    np.save("results/dqn_q_losses.npy", q_losses)