import os
import sys
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class PolicyGradientTrainer:
    def __init__(self):
        self.results = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        from environment.custom_env import TradingEnv
        self.TradingEnv = TradingEnv
        
    def setup_environment(self):
        np.random.seed(42)
        env = self.TradingEnv()
        return Monitor(env)
    
    def train_and_evaluate(self, model, model_name, env):
        try:
            print(f"\n=== Training {model_name} ===")
            model.learn(total_timesteps=100000, log_interval=100)
            model.save(f"models/pg/{model_name.lower()}_trading_env")
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
            self.results.append((model_name, mean_reward, std_reward))
            return True
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return False
    
    def print_final_results(self):
        print("\n=== Final Evaluation Results ===")
        print("{:<12} {:<15} {:<15}".format("Model", "Mean Reward", "Std Reward"))
        print("-" * 45)
        for name, mean, std in self.results:
            print("{:<12} {:<15.2f} {:<15.2f}".format(name, mean, std))
        print("-" * 45)
    
    def run(self):
        env = self.setup_environment()
        os.makedirs("models/pg", exist_ok=True)
        
        # REINFORCE
        reinforce_model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1,
            use_rms_prop=False,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=1
        )
        self.train_and_evaluate(reinforce_model, "REINFORCE", env)
        
        # PPO
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
        self.train_and_evaluate(ppo_model, "PPO", env)
        
        # A2C
        a2c_model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=1
        )
        self.train_and_evaluate(a2c_model, "A2C", env)
        
        self.print_final_results()

if __name__ == "__main__":
    trainer = PolicyGradientTrainer()
    trainer.run()