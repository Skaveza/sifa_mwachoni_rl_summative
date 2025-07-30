RL Summative: Multi-Asset Trading Environment
This project implements and compares reinforcement learning (RL) agents—Deep Q-Network (DQN), REINFORCE, Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C)—in a custom multi-asset trading environment. The agents manage a portfolio of five assets, optimizing trades under dynamic 0.1% transaction costs and 40% concentration limits. The project includes training scripts, visual demonstrations (GIF and MP4), and a performance report, built using Gymnasium, Stable Baselines3, PyTorch, Pygame, and OpenCV.
Project Overview
The trading environment simulates a market with five correlated assets (prices $50–$150), a 1% chance of market events, and a 0.6/step price trend. Agents observe a 16D state (cash, 5 shares, 5 prices, 5 moving averages, event flag) and select from 16 discrete actions (buy, sell, hold for each asset, plus no-op). The reward combines portfolio return, concentration penalties, and transaction costs. The goal is to maximize portfolio value over 100-step episodes, starting with $10,000 and 50 shares per asset.
Key Outputs:

Models: Trained models for DQN, REINFORCE, PPO, and A2C.
Visuals:
results/random_agent.gif: 200-step random agent demonstration (price chart width halved to 640 pixels, legend on right showing asset colors).
results/reinforce_performance.mp4: 3-minute video of REINFORCE agent (3 episodes, ~100 steps each, 30 fps, ~1.97 cumulative reward).


Report: results/report.md and results/report.pdf with performance metrics (50 episodes):
DQN: 1.34 ± 0.14
REINFORCE: 1.97 ± 0.22
PPO: 1.95 ± 0.17
A2C: 1.91 ± 0.18



Prerequisites

Python 3.8+
macOS/Linux (for brew and ffmpeg)
Git (for cloning the repository)

Setup

Clone the Repository:
git clone <repository-url>
cd sifa_mwachoni_rl_summative


Create and Activate Virtual Environment:
python -m venv venv
source venv/bin/activate


Install Dependencies:
pip install -r requirements.txt
brew install ffmpeg
brew install pandoc

Verify ffmpeg installation:
ffmpeg -version

Required Python packages (in requirements.txt):
gymnasium
stable-baselines3
torch
numpy
pygame
opencv-python
imageio
imageio-ffmpeg



Project Structure
sifa_mwachoni_rl_summative/
├── environment/
│   ├── custom_env.py        # Trading environment (16D state, 16 actions)
├── implementation/
│   ├── rendering.py         # Renders GIF and MP4 (price chart: 640px wide, legend on right)
│   ├── render_pg.py         # REINFORCE model for video rendering
├── training/
│   ├── pg_training.py       # Trains REINFORCE, PPO, A2C
│   ├── dqn_training.py      # Trains DQN
├── models/
│   ├── pg/
│   │   ├── reinforce_trading_env.pth
│   │   ├── ppo_trading_env.zip
│   │   ├── a2c_trading_env.zip
│   ├── dqn/
│   │   ├── dqn_trading_env.zip
├── results/
│   ├── random_agent.gif     # Random agent demo
│   ├── reinforce_performance.mp4  # REINFORCE agent video
│   ├── performance_data.json  # Raw performance data
├── main.py                  # Main script for training and output generation
├── requirements.txt         # Dependencies
├── README.md                # This file

Usage
Run the main script to train models, generate visuals, and produce the report:
python main.py

Expected Output

Console:Running DQN training...
Running policy gradient training...
Recording random agent...
Saved GIF to results/random_agent.gif
Recording REINFORCE agent...
Saved performance video to results/reinforce_performance.mp4
Generated report at results/report.md
Operation completed successfully!

Visual Outputs

Random Agent GIF (results/random_agent.gif):
Shows a 200-step episode of a random agent.
Price chart (640 pixels wide, x=50 to 690) displays 5 asset price histories (Red: Asset 1, Green: Asset 2, Blue: Asset 3, Yellow: Asset 4, Magenta: Asset 5).
Legend on the right (x=710, y=290) lists asset colors.
Displays portfolio metrics (total value, cash, return, step, cumulative reward).

REINFORCE Video (results/reinforce_performance.mp4):
3-minute video (3 episodes, ~100 steps each, 30 fps).
Shows REINFORCE agent trading (~1.97 cumulative reward).
Same chart and legend format as the GIF.


Troubleshooting

ModuleNotFoundError:
Ensure rendering.py, render_pg.py in implementation/, custom_env.py in environment/.
Run with explicit path:export PYTHONPATH=$PYTHONPATH:./implementation:./environment
python main.py


REINFORCE Model Not Found:
Verify models/pg/reinforce_trading_env.pth.
Rerun training:cd training
python pg_training.py


GIF/Video Fails:
Confirm dependencies:ffmpeg -version
pip show opencv-python imageio imageio-ffmpeg


Run standalone REINFORCE video:cd implementation
python -c "from rendering import record_agent_performance; from environment.custom_env import TradingEnv; from render_pg import REINFORCE; env = TradingEnv(); model = REINFORCE(env).load('../models/pg/reinforce_trading_env'); record_agent_performance(env, model, '../results/reinforce_performance.mp4', episodes=3, max_steps=100, fps=30)"


Cumulative Reward is 0:
Add debug print in rendering.py’s _draw_portfolio_metrics:# Edit implementation/rendering.py
def _draw_portfolio_metrics(self, env):
    print(f"Debug: episode_total_reward = {getattr(env, 'episode_total_reward', 0)}")
    ...


Verify custom_env.py updates episode_total_reward in step.


Legend/Chart Issues:
Check rendering.py’s _draw_price_chart for w=640, legend_x=710, legend_y=290.
Adjust legend_x or legend_y if overlap occurs:# Edit implementation/rendering.py
legend_x, legend_y = x + w + 20, y + 10  # Try legend_x = x + w + 30





Notes

The REINFORCE agent (1.97 ± 0.22) outperforms other RL models, due to transaction costs and concentration penalties.
The random agent GIF includes a halved-width price chart (640 pixels) with a legend on the right for clarity.
