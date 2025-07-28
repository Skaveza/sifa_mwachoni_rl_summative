import os
import sys
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import TradingEnv

class PolicyGradientTrainer:
    def __init__(self):
        self.results = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
    def train(self):
        env = Monitor(TradingEnv())
        os.makedirs("models/pg", exist_ok=True)
        
        # REINFORCE (A2C with n_steps=1)
        reinforce = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1,
            use_rms_prop=False,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=1
        )
        reinforce.learn(total_timesteps=300000)
        reinforce.save("models/pg/reinforce_trading_env")
        
        # PPO
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
        ppo.learn(total_timesteps=300000)
        ppo.save("models/pg/ppo_trading_env")
        
        # A2C
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
        a2c.learn(total_timesteps=300000)
        a2c.save("models/pg/a2c_trading_env")
        
        # Evaluation
        print("\n=== Final Evaluation ===")
        for name, model in [("REINFORCE", reinforce), ("PPO", ppo), ("A2C", a2c)]:
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
            self.results.append((name, mean_reward, std_reward))
            print(f"{name}: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    trainer = PolicyGradientTrainer()
    trainer.train()