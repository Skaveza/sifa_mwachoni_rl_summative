import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.custom_env import TradingEnv
from implementation.render_pg import REINFORCE

np.random.seed(42)
torch.manual_seed(42)

class PGCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PGCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.entropies = []
        self.current_rewards = []
    
    def _on_step(self) -> bool:
        self.current_rewards.append(self.locals["rewards"])
        # Collect entropy for PPO and A2C
        if "policy_loss" in self.locals:
            entropy = self.locals.get("entropy_loss", 0)
            self.entropies.append(entropy)
        if self.locals["dones"]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True

class PolicyGradientTrainer:
    def __init__(self):
        self.results = []
        
    def train(self):
        env = Monitor(TradingEnv())
        os.makedirs("models/pg", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # REINFORCE
        print("\n=== Training REINFORCE Agent ===")
        reinforce = REINFORCE(env, learning_rate=3e-4, gamma=0.99)
        episode_rewards, entropies = reinforce.learn(total_timesteps=300000)
        reinforce.save("models/pg/reinforce_trading_env")
        mean_reward = np.mean(episode_rewards[-50:])
        std_reward = np.std(episode_rewards[-50:])
        self.results.append(("REINFORCE", mean_reward, std_reward))
        np.save("results/reinforce_episode_rewards.npy", episode_rewards)
        np.save("results/reinforce_entropies.npy", entropies)
        
        # PPO
        print("\n=== Training PPO Agent ===")
        ppo_callback = PGCallback()
        ppo = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.8,
            verbose=1
        )
        ppo.learn(total_timesteps=300000, callback=ppo_callback)
        ppo.save("models/pg/ppo_trading_env")
        mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=50)
        self.results.append(("PPO", mean_reward, std_reward))
        np.save("results/ppo_episode_rewards.npy", ppo_callback.episode_rewards)
        np.save("results/ppo_entropies.npy", ppo_callback.entropies)
        
        # A2C
        print("\n=== Training A2C Agent ===")
        a2c_callback = PGCallback()
        a2c = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=7e-4,
            n_steps=10,
            gamma=0.99,
            gae_lambda=1.0,
            use_rms_prop=True,
            verbose=1
        )
        a2c.learn(total_timesteps=300000, callback=a2c_callback)
        a2c.save("models/pg/a2c_trading_env")
        mean_reward, std_reward = evaluate_policy(a2c, env, n_eval_episodes=50)
        self.results.append(("A2C", mean_reward, std_reward))
        np.save("results/a2c_episode_rewards.npy", a2c_callback.episode_rewards)
        np.save("results/a2c_entropies.npy", a2c_callback.entropies)
        
        self.print_results()
    
    def print_results(self):
        print("\n=== Final Evaluation Results ===")
        print(f"{'Model':<12} | {'Mean Reward':<12} | {'Std Reward':<12}")
        print("-" * 40)
        for name, mean_reward, std_reward in self.results:
            print(f"{name:<12} | {mean_reward:>11.2f} | {std_reward:>11.2f}")

if __name__ == "__main__":
    trainer = PolicyGradientTrainer()
    trainer.train()
