import os
import sys
import numpy as np
from typing import List, Optional, Dict
from stable_baselines3 import PPO

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from environment.custom_env import TradingEnv
from implementation.rendering import Renderer, save_gif, record_agent_performance
from training.dqn_training import DQNTrainer
from training.pg_training import PolicyGradientTrainer

def setup_paths():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def initialize_environment():
    env = TradingEnv()
    return env

def generate_demonstrations(env) -> Dict[str, List[Dict]]:
    os.makedirs("results", exist_ok=True)
    performance_data = {}
    
    print("\nRecording random agent...")
    save_gif(
        env,
        filename="results/random_agent.gif",
        max_steps=200,
        fps=15
    )
    
    # Record only the best-performing agent (PPO)
    model_name = "PPO"
    model_class = PPO
    path = "models/pg/ppo_trading_env"
    print(f"\nRecording {model_name} agent...")
    if not os.path.exists(path + ".zip"):
        print(f"Model not found at {path}.zip")
    else:
        model = model_class.load(path, env=env)
        data = record_agent_performance(
            env=env,
            model=model,
            filename=f"results/{model_name.lower()}_performance.mp4",
            episodes=3,
            max_steps=100,
            fps=30
        )
        if data:
            performance_data[model_name] = data
    
    return performance_data

def main():
    setup_paths()
    try:
        print("Running DQN training...")
        dqn_trainer = DQNTrainer()
        dqn_trainer.train()
        
        print("Running policy gradient training...")
        pg_trainer = PolicyGradientTrainer()
        pg_trainer.train()
        
        env = initialize_environment()
        performance_data = generate_demonstrations(env)
        
        print("\nRunning metric analysis...")
        from analysis import main as analysis_main
        analysis_main()
        
        print("\nOperation completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()