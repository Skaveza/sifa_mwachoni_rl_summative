import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from typing import Dict, List

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from environment.custom_env import TradingEnv
from implementation.render_pg import REINFORCE
from stable_baselines3 import DQN, PPO, A2C

def load_training_data():
    models = ["DQN", "REINFORCE", "PPO", "A2C"]
    data = {}
    
    for model in models:
        reward_file = f"results/{model.lower()}_episode_rewards.npy"
        if os.path.exists(reward_file):
            rewards = np.load(reward_file)
            # Flatten the array to ensure it's 1D
            rewards = rewards.flatten()
            if len(rewards) > 0:
                data[model] = {
                    "rewards": rewards.cumsum(),
                    "raw_rewards": rewards
                }
                print(f"Loaded {model} rewards: shape {rewards.shape}, length {len(rewards)}")
        
        if model == "DQN":
            loss_file = f"results/dqn_q_losses.npy"
            if os.path.exists(loss_file):
                losses = np.load(loss_file)
                # Flatten the array to ensure it's 1D
                losses = losses.flatten()
                if len(losses) > 0:
                    data[model]["q_loss"] = losses
                    print(f"Loaded {model} Q-losses: shape {losses.shape}, length {len(losses)}")
        else:
            entropy_file = f"results/{model.lower()}_entropies.npy"
            if os.path.exists(entropy_file):
                entropies = np.load(entropy_file)
                # Flatten the array to ensure it's 1D
                entropies = entropies.flatten()
                if len(entropies) > 0:
                    data[model]["entropy"] = entropies
                    print(f"Loaded {model} entropies: shape {entropies.shape}, length {len(entropies)}")
    
    return data

def plot_enhanced_reward_analysis(data: Dict[str, Dict], output_dir="results"):
    """
    Create comprehensive reward analysis with multiple perspectives
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = ["DQN", "REINFORCE", "PPO", "A2C"]
    colors = ['blue', 'orange', 'green', 'red']
    
    # Create a comprehensive figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Reward Analysis', fontsize=16, fontweight='bold')
    
    # 1. Episode-wise rewards (not cumulative)
    for i, model in enumerate(models):
        if model in data and "raw_rewards" in data[model]:
            episode_rewards = data[model]["raw_rewards"]
            # Ensure it's 1D and convert to pandas Series safely
            episode_rewards = np.array(episode_rewards).flatten()
            # Smooth the episode rewards for better visibility
            smoothed_rewards = pd.Series(episode_rewards).rolling(window=50, min_periods=1).mean()
            axes[0,0].plot(smoothed_rewards, label=model, color=colors[i], linewidth=2)
    
    axes[0,0].set_title('Episode Rewards (Smoothed, window=50)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward per Episode')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Cumulative rewards (your existing plot)
    for i, model in enumerate(models):
        if model in data and "rewards" in data[model]:
            axes[0,1].plot(data[model]["rewards"], label=model, color=colors[i], linewidth=2)
    
    axes[0,1].set_title('Cumulative Rewards')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Cumulative Reward')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Learning efficiency (cumulative reward / episode)
    for i, model in enumerate(models):
        if model in data and "rewards" in data[model]:
            cumulative = data[model]["rewards"]
            episodes = np.arange(1, len(cumulative) + 1)
            efficiency = cumulative / episodes
            axes[0,2].plot(efficiency, label=model, color=colors[i], linewidth=2)
    
    axes[0,2].set_title('Learning Efficiency (Cumulative Reward / Episode)')
    axes[0,2].set_xlabel('Episode')
    axes[0,2].set_ylabel('Average Reward per Episode')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Performance improvement rate (derivative of cumulative rewards)
    for i, model in enumerate(models):
        if model in data and "rewards" in data[model]:
            cumulative = data[model]["rewards"]
            if len(cumulative) > 1:
                improvement_rate = np.diff(cumulative)
                # Smooth the improvement rate
                smoothed_rate = pd.Series(improvement_rate).rolling(window=100, min_periods=1).mean()
                axes[1,0].plot(smoothed_rate, label=model, color=colors[i], linewidth=2)
    
    axes[1,0].set_title('Learning Rate (Change in Cumulative Reward)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Reward Improvement per Episode')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Moving average performance window
    window_size = 100
    for i, model in enumerate(models):
        if model in data and "raw_rewards" in data[model]:
            episode_rewards = np.array(data[model]["raw_rewards"]).flatten()
            if len(episode_rewards) >= window_size:
                moving_avg = pd.Series(episode_rewards).rolling(window=window_size).mean()
                axes[1,1].plot(moving_avg, label=f'{model} (MA-{window_size})', 
                              color=colors[i], linewidth=2)
    
    axes[1,1].set_title(f'Moving Average Performance (window={window_size})')
    axes[1,1].set_xlabel('Episode')
    axes[1,1].set_ylabel('Average Reward')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Performance stability (rolling standard deviation)
    for i, model in enumerate(models):
        if model in data and "raw_rewards" in data[model]:
            episode_rewards = np.array(data[model]["raw_rewards"]).flatten()
            if len(episode_rewards) >= window_size:
                rolling_std = pd.Series(episode_rewards).rolling(window=window_size).std()
                axes[1,2].plot(rolling_std, label=f'{model}', color=colors[i], linewidth=2)
    
    axes[1,2].set_title(f'Performance Stability (Rolling Std, window={window_size})')
    axes[1,2].set_xlabel('Episode')
    axes[1,2].set_ylabel('Reward Standard Deviation')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "enhanced_reward_analysis.png"), 
                dpi=150, bbox_inches='tight')
    print(f"Saved enhanced reward analysis to {output_dir}/enhanced_reward_analysis.png")
    plt.close()

def plot_convergence_analysis(data: Dict[str, Dict], output_dir="results"):
    """
    Analyze when each model converges to stable performance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    
    models = ["DQN", "REINFORCE", "PPO", "A2C"]
    colors = ['blue', 'orange', 'green', 'red']
    
    convergence_info = {}
    
    for i, model in enumerate(models):
        if model in data and "raw_rewards" in data[model]:
            rewards = np.array(data[model]["raw_rewards"]).flatten()
            
            # Calculate convergence metrics
            window = min(100, len(rewards) // 10)  # Adaptive window size
            final_performance = np.mean(rewards[-window:])
            convergence_threshold = 0.1  # 10% of final performance
            
            # Find convergence point
            convergence_episode = None
            if len(rewards) >= window * 2:  # Need enough data
                for j in range(window, len(rewards)):
                    recent_performance = np.mean(rewards[j-window:j])
                    if abs(final_performance) > 1e-6:  # Avoid division by zero
                        if abs(recent_performance - final_performance) / abs(final_performance) < convergence_threshold:
                            convergence_episode = j
                            break
            
            convergence_info[model] = {
                'episode': convergence_episode or len(rewards),
                'final_performance': final_performance,
                'performance_at_convergence': recent_performance if convergence_episode else final_performance
            }
            
            # Plot 1: Convergence visualization
            axes[0,0].plot(rewards, alpha=0.3, color=colors[i], linewidth=0.5)
            smoothed = pd.Series(rewards).rolling(window=50, min_periods=1).mean()
            axes[0,0].plot(smoothed, label=model, color=colors[i], linewidth=2)
            
            if convergence_episode:
                axes[0,0].axvline(x=convergence_episode, color=colors[i], 
                                 linestyle='--', alpha=0.7, linewidth=1)
                axes[0,0].axhline(y=final_performance, color=colors[i], 
                                 linestyle=':', alpha=0.5, linewidth=1)
    
    axes[0,0].set_title('Convergence Points (Dashed Lines)')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Reward per Episode')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Sample efficiency comparison
    for i, model in enumerate(models):
        if model in data and "rewards" in data[model]:
            cumulative = data[model]["rewards"]
            # Normalize by final performance for fair comparison
            final_cumulative = cumulative[-1]
            if final_cumulative != 0:
                normalized_cumulative = cumulative / final_cumulative
                axes[0,1].plot(normalized_cumulative, label=model, color=colors[i], linewidth=2)
    
    axes[0,1].set_title('Sample Efficiency (Normalized Cumulative Rewards)')
    axes[0,1].set_xlabel('Episode')
    axes[0,1].set_ylabel('Normalized Cumulative Reward')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Learning curve comparison (percentage of final performance reached)
    for i, model in enumerate(models):
        if model in data and "raw_rewards" in data[model]:
            rewards = np.array(data[model]["raw_rewards"]).flatten()
            final_perf = convergence_info[model]['final_performance']
            
            if abs(final_perf) > 1e-6:  # Avoid division by zero
                # Calculate percentage of final performance reached over time
                smoothed_rewards = pd.Series(rewards).rolling(window=50, min_periods=1).mean()
                percentage_reached = (smoothed_rewards / final_perf) * 100
                
                axes[1,0].plot(percentage_reached, label=model, color=colors[i], linewidth=2)
    
    axes[1,0].axhline(y=90, color='black', linestyle='--', alpha=0.5, label='90% threshold')
    axes[1,0].set_title('Learning Progress (% of Final Performance)')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('% of Final Performance')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Convergence summary bar chart
    model_names = []
    convergence_episodes = []
    final_performances = []
    
    for model in models:
        if model in convergence_info:
            model_names.append(model)
            convergence_episodes.append(convergence_info[model]['episode'])
            final_performances.append(convergence_info[model]['final_performance'])
    
    if model_names:
        x_pos = np.arange(len(model_names))
        bars = axes[1,1].bar(x_pos, convergence_episodes, 
                            color=[colors[models.index(m)] for m in model_names])
        axes[1,1].set_title('Episodes to Convergence')
        axes[1,1].set_xlabel('Model')
        axes[1,1].set_ylabel('Episodes to Convergence')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(model_names)
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, episodes in zip(bars, convergence_episodes):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + max(convergence_episodes)*0.01,
                          f'{episodes}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_analysis.png"), 
                dpi=150, bbox_inches='tight')
    print(f"Saved convergence analysis to {output_dir}/convergence_analysis.png")
    plt.close()
    
    return convergence_info

def generate_performance_summary(data: Dict[str, Dict], convergence_info: Dict, output_dir="results"):
    """
    Generate a comprehensive performance summary table
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    
    for model in ["DQN", "REINFORCE", "PPO", "A2C"]:
        if model in data and "raw_rewards" in data[model]:
            rewards = np.array(data[model]["raw_rewards"]).flatten()
            cumulative = data[model]["rewards"]
            
            # Calculate various metrics
            final_cumulative = cumulative[-1]
            window = min(100, len(rewards) // 5)  # Adaptive window
            final_avg_reward = np.mean(rewards[-window:])
            best_episode_reward = np.max(rewards)
            worst_episode_reward = np.min(rewards)
            reward_std = np.std(rewards[-window:])
            
            # Sample efficiency: episodes to reach 90% of final performance
            target_performance = final_avg_reward * 0.9
            smoothed_rewards = pd.Series(rewards).rolling(window=50, min_periods=1).mean()
            episodes_to_90pct = None
            for i, reward in enumerate(smoothed_rewards):
                if not pd.isna(reward) and reward >= target_performance:
                    episodes_to_90pct = i
                    break
            
            convergence_episode = convergence_info.get(model, {}).get('episode', len(rewards))
            
            summary_data.append({
                'Model': model,
                'Final Cumulative Reward': f"{final_cumulative:.0f}",
                'Final Avg Reward': f"{final_avg_reward:.2f}",
                'Best Episode Reward': f"{best_episode_reward:.2f}",
                'Worst Episode Reward': f"{worst_episode_reward:.2f}",
                'Reward Stability (Std)': f"{reward_std:.2f}",
                'Episodes to 90% Performance': episodes_to_90pct or "N/A",
                'Episodes to Convergence': convergence_episode,
                'Sample Efficiency Rank': 0  # Will be filled later
            })
    
    # Calculate sample efficiency ranking
    valid_90pct = [s for s in summary_data if s['Episodes to 90% Performance'] != "N/A"]
    valid_90pct.sort(key=lambda x: x['Episodes to 90% Performance'])
    
    for i, entry in enumerate(valid_90pct):
        entry['Sample Efficiency Rank'] = i + 1
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    summary_df.to_csv(os.path.join(output_dir, "performance_summary.csv"), index=False)
    
    # Create a formatted markdown table
    try:
        markdown_table = summary_df.to_markdown(index=False)
    except AttributeError:
        # Fallback if to_markdown is not available
        markdown_table = summary_df.to_string(index=False)
    
    with open(os.path.join(output_dir, "performance_summary.md"), "w") as f:
        f.write("# Model Performance Summary\n\n")
        f.write(markdown_table)
        f.write("\n\n## Interpretation:\n")
        f.write("- **Final Cumulative Reward**: Total reward accumulated over all training episodes\n")
        f.write("- **Final Avg Reward**: Average reward per episode in the last episodes\n")
        f.write("- **Reward Stability**: Standard deviation of rewards in the final episodes (lower is more stable)\n")
        f.write("- **Episodes to 90% Performance**: How quickly the model reaches 90% of its final performance\n")
        f.write("- **Episodes to Convergence**: When the model's performance stabilizes within 10% of final performance\n")
        f.write("- **Sample Efficiency Rank**: Ranking based on episodes to reach 90% performance (1 = most efficient)\n")
    
    print(f"Generated performance summary at {output_dir}/performance_summary.csv and .md")
    return summary_df

def main():
    """
    Main function to run the enhanced analysis
    """
    print("Starting enhanced RL analysis...")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    training_data = load_training_data()
    
    if not training_data:
        print("No training data found! Make sure your .npy files are in the results/ directory.")
        return
    
    print(f"Found data for models: {list(training_data.keys())}")
    
    # Generate enhanced visualizations
    print("Generating enhanced reward analysis...")
    plot_enhanced_reward_analysis(training_data)
    
    print("Generating convergence analysis...")
    convergence_info = plot_convergence_analysis(training_data)
    
    print("Generating performance summary...")
    performance_summary = generate_performance_summary(training_data, convergence_info)
    
    print("\n" + "="*50)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("="*50)
    print("\nFiles generated:")
    print("- enhanced_reward_analysis.png")
    print("- convergence_analysis.png") 
    print("- performance_summary.csv")
    print("- performance_summary.md")
    
    print("\nQuick Performance Overview:")
    if not performance_summary.empty:
        print(performance_summary[['Model', 'Final Avg Reward', 'Episodes to 90% Performance', 'Sample Efficiency Rank']].to_string(index=False))
    
    print("\nCheck the results/ directory for all generated files!")

if __name__ == "__main__":
    main()