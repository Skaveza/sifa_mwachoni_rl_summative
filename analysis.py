import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from environment.custom_env import TradingEnv
from implementation.reinforce import REINFORCE
import torch
from stable_baselines3 import DQN, PPO, A2C
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def load_training_data():
    """Load training data from saved .npy files."""
    models = ["DQN", "REINFORCE", "PPO", "A2C"]
    data = {}
    
    for model in models:
        reward_file = f"results/{model.lower()}_episode_rewards.npy"
        if os.path.exists(reward_file):
            rewards = np.load(reward_file)
            data[model] = {
                "rewards": rewards.cumsum(),
                "raw_rewards": rewards
            }
        
        if model == "DQN":
            loss_file = f"results/dqn_q_losses.npy"
            if os.path.exists(loss_file):
                data[model]["q_loss"] = np.load(loss_file)
        else:
            entropy_file = f"results/{model.lower()}_entropies.npy"
            if os.path.exists(entropy_file):
                data[model]["entropy"] = np.load(entropy_file)
    
    return data

def plot_cumulative_rewards(data: Dict[str, Dict], output_dir="results"):
    """Generate individual and combined cumulative reward plots."""
    os.makedirs(output_dir, exist_ok=True)
    models = ["DQN", "REINFORCE", "PPO", "A2C"]
    
    for model in models:
        if model in data and "rewards" in data[model]:
            plt.figure(figsize=(8, 6))
            plt.plot(data[model]["rewards"], label=f"{model} Cumulative Reward", color="blue")
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(f"{model} Cumulative Reward Over {len(data[model]['rewards'])} Episodes")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{model.lower()}_cumulative_reward.png"))
            plt.close()
    
    plt.figure(figsize=(10, 6))
    for model in models:
        if model in data and "rewards" in data[model]:
            plt.plot(data[model]["rewards"], label=model)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Comparison Across Models")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "combined_cumulative_reward.png"))
    plt.close()

def plot_objective_functions(data: Dict[str, Dict], output_dir="results"):
    """Plot Q-loss for DQN and policy entropy for PG methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    if "DQN" in data and "q_loss" in data["DQN"]:
        plt.figure(figsize=(8, 6))
        plt.plot(data["DQN"]["q_loss"], label="DQN Q-Loss", color="red")
        plt.xlabel("Training Step")
        plt.ylabel("Q-Loss")
        plt.title("DQN Objective Function (Q-Loss) Over Training")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "dqn_q_loss.png"))
        plt.close()
    
    plt.figure(figsize=(10, 6))
    for model in ["REINFORCE", "PPO", "A2C"]:
        if model in data and "entropy" in data[model]:
            plt.plot(data[model]["entropy"], label=f"{model} Entropy")
    plt.xlabel("Training Step")
    plt.ylabel("Policy Entropy")
    plt.title("Policy Entropy for Policy Gradient Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pg_policy_entropy.png"))
    plt.close()

def test_unseen_states(env, models: Dict, episodes=10) -> Dict[str, List[Dict]]:
    """Test trained models on unseen initial states."""
    performance_data = {}
    
    def modified_reset(self, seed=None):
        super(TradingEnv, self).reset(seed=seed)
        self.prices = np.random.uniform(100, 200, self.num_assets)
        self.shares = np.random.randint(20, 80, self.num_assets)
        self.price_history = [self.prices.copy()]
        self.moving_avg = self.prices.copy()
        self.correlation_matrix = np.array([
            [1.0, 0.2, 0.1, -0.3, 0.0],
            [0.2, 1.0, 0.4, -0.2, 0.1],
            [0.1, 0.4, 1.0, 0.3, 0.2],
            [-0.3, -0.2, 0.3, 1.0, 0.0],
            [0.0, 0.1, 0.2, 0.0, 1.0]
        ])
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        self.episode_total_reward = 0.0
        return self._get_state(), {}
    
    env.reset = modified_reset.__get__(env, TradingEnv)
    
    for model_name, model in models.items():
        episode_data = []
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_rewards = []
            done = False
            for _ in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                if done or truncated:
                    break
            episode_data.append({
                "episode": episode + 1,
                "total_reward": sum(episode_rewards),
                "final_value": env.portfolio_value
            })
        performance_data[model_name] = episode_data
    
    return performance_data

def analyze_stability(data: Dict[str, Dict], threshold=0.05, window=5):
    """Analyze episodes required for stable performance."""
    stability_info = {}
    for model in data:
        if "raw_rewards" not in data[model]:
            continue
        rewards = data[model]["raw_rewards"]
        final_mean = np.mean(rewards[-5:])
        for i in range(len(rewards) - window + 1):
            window_mean = np.mean(rewards[i:i+window])
            if abs(window_mean - final_mean) / final_mean < threshold:
                stability_info[model] = {
                    "stable_episode": i + window,
                    "mean_reward": window_mean,
                    "std_reward": np.std(rewards[i:i+window])
                }
                break
        else:
            stability_info[model] = {
                "stable_episode": len(rewards),
                "mean_reward": final_mean,
                "std_reward": np.std(rewards[-window:])
            }
    return stability_info

def generate_analysis_report(training_data: Dict, unseen_data: Dict, stability_info: Dict, output_dir="results"):
    """Generate markdown report with plots and stability analysis."""
    os.makedirs(output_dir, exist_ok=True)
    report = """# RL Trading Agent Metric Analysis

## Cumulative Reward Analysis
Below are the cumulative reward plots for each model over training episodes. The random agent is excluded as it does not learn, and its visualization (`random_agent.gif`) omits cumulative reward in the Portfolio Panel to reflect pre-training behavior.

"""
    
    for model in ["DQN", "REINFORCE", "PPO", "A2C"]:
        if model in training_data and "rewards" in training_data[model]:
            report += f"![{model} Cumulative Reward]({model.lower()}_cumulative_reward.png)\n"
    
    report += "\n### Combined Cumulative Reward\n"
    report += "![Combined Cumulative Reward](combined_cumulative_reward.png)\n"
    
    report += "\n## Objective Function and Policy Entropy\n"
    if "DQN" in training_data and "q_loss" in training_data["DQN"]:
        report += "![DQN Q-Loss](dqn_q_loss.png)\n"
    report += "![Policy Entropy](pg_policy_entropy.png)\n"
    
    report += "\n## Performance on Unseen Initial States\n"
    report += "Models were tested on 10 episodes with unseen initial conditions (prices in [100, 200], shares in [20, 80], altered correlation matrix).\n"
    report += "| Model | Avg Reward | Avg Growth (%) |\n"
    report += "|-------|------------|----------------|\n"
    for model, episodes in unseen_data.items():
        avg_reward = np.mean([ep["total_reward"] for ep in episodes])
        avg_growth = np.mean([(ep["final_value"] / 10000 - 1) * 100 for ep in episodes])
        report += f"| {model} | {avg_reward:.2f} | {avg_growth:.1f}% |\n"
    
    report += "\n## Episodes to Stable Performance\n"
    report += "Stability is defined as the mean reward over 5 consecutive episodes being within 5% of the final mean reward (average of last 5 episodes).\n\n"
    report += "| Model | Stable Episode | Mean Reward | Std Reward |\n"
    report += "|-------|----------------|-------------|------------|\n"
    for model, info in stability_info.items():
        report += f"| {model} | {info['stable_episode']} | {info['mean_reward']:.2f} | {info['std_reward']:.2f} |\n"
    
    with open(os.path.join(output_dir, "analysis_report.md"), "w") as f:
        f.write(report)
    print(f"Generated analysis report at {output_dir}/analysis_report.md")

def main():
    """Run metric analysis and generate report."""
    os.makedirs("results", exist_ok=True)
    
    training_data = load_training_data()
    plot_cumulative_rewards(training_data)
    plot_objective_functions(training_data)
    
    env = TradingEnv()
    models = {}
    try:
        reinforce_path = "models/pg/reinforce_trading_env"
        if os.path.exists(reinforce_path + ".pth"):
            models["REINFORCE"] = REINFORCE(env).load(reinforce_path)
        
        for model_name, path in [
            ("DQN", "models/dqn/dqn_trading_env"),
            ("PPO", "models/pg/ppo_trading_env"),
            ("A2C", "models/pg/a2c_trading_env")
        ]:
            if os.path.exists(path + ".zip"):
                model_class = globals()[model_name]
                models[model_name] = model_class.load(path, env=env)
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return
    
    unseen_data = test_unseen_states(env, models)
    stability_info = analyze_stability(training_data)
    generate_analysis_report(training_data, unseen_data, stability_info)

if __name__ == "__main__":
    main()