import os
import sys
from environment.custom_env import TradingEnv
from implementation.rendering import save_gif

def main():
    # Initialize environment
    env = TradingEnv()
    
    # Create GIF of random actions
    print("Creating random action demonstration...")
    save_gif(
        env, 
        filename="trading_env_random.gif",
        max_steps=100,
        fps=10
    )
    
    # Add your other main functionality here
    print("Demo complete. Run training scripts separately.")

if __name__ == "__main__":
    # Fix imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    
    main()