# RL Summative: Multi-Asset Trading Environment

This project implements and compares reinforcement learning (RL) agents—Deep Q-Network (DQN), REINFORCE, Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C)—in a custom multi-asset trading environment. The agents manage a portfolio of five assets, optimizing trades under dynamic 0.1% transaction costs and 40% concentration limits.

## Project Overview

The trading environment simulates a market with five correlated assets (prices $50–$150), a 1% chance of market events, and a 0.6/step price trend. Agents observe a 16D state (cash, 5 shares, 5 prices, 5 moving averages, event flag) and select from 16 discrete actions (buy, sell, hold for each asset, plus no-op). The reward combines portfolio return, concentration penalties, and transaction costs. The goal is to maximize portfolio value over 100-step episodes, starting with $10,000 and 50 shares per asset.

### Key Outputs

- **Models**: Trained models for DQN, REINFORCE, PPO, and A2C
- **Visuals**:
  - `results/random_agent.gif`: 200-step random agent demonstration (price chart width halved to 640 pixels, legend on right showing asset colors)
  - `results/reinforce_performance.mp4`: 3-minute video of REINFORCE agent (3 episodes, ~100 steps each, 30 fps, ~1.97 cumulative reward)
- **Report**: `results/report.md` and `results/report.pdf` with performance metrics (50 episodes):
  - DQN: 1.34 ± 0.14
  - REINFORCE: 1.97 ± 0.22
  - PPO: 1.95 ± 0.17
  - A2C: 1.91 ± 0.18

## Prerequisites

- Python 3.8+
- macOS/Linux (for brew and ffmpeg)
- Git (for cloning the repository)

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd sifa_mwachoni_rl_summative
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install ffmpeg
brew install pandoc

# Verify ffmpeg installation
ffmpeg -version
```

### Required Python Packages

The following packages are included in `requirements.txt`:

```
gymnasium
stable-baselines3
torch
numpy
pygame
opencv-python
imageio
imageio-ffmpeg
```

## Project Structure

```
sifa_mwachoni_rl_summative/
├── environment/
│   └── custom_env.py        # Trading environment (16D state, 16 actions)
├── implementation/
│   ├── rendering.py         # Renders GIF and MP4 (price chart: 640px wide, legend on right)
│   └── render_pg.py         # REINFORCE model for video rendering
├── training/
│   ├── pg_training.py       # Trains REINFORCE, PPO, A2C
│   └── dqn_training.py      # Trains DQN
├── models/
│   ├── pg/
│   │   ├── reinforce_trading_env.pth
│   │   ├── ppo_trading_env.zip
│   │   └── a2c_trading_env.zip
│   └── dqn/
│       └── dqn_trading_env.zip
├── results/
│   ├── random_agent.gif     # Random agent demo
│   ├── reinforce_performance.mp4  # REINFORCE agent video
│   └── performance_data.json  # Raw performance data
├── main.py                  # Main script for training and output generation
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Usage

Run the main script to train models, generate visuals, and produce the report:

```bash
python main.py
```

### Expected Output

```
Running DQN training...
Running policy gradient training...
Recording random agent...
Saved GIF to results/random_agent.gif
Recording REINFORCE agent...
Saved performance video to results/reinforce_performance.mp4
Generated report at results/report.md
Operation completed successfully!
```

## Visual Outputs

### Random Agent GIF (`results/random_agent.gif`)
- Shows a 200-step episode of a random agent
- Price chart (640 pixels wide, x=50 to 690) displays 5 asset price histories:
  - Red: Asset 1
  - Green: Asset 2
  - Blue: Asset 3
  - Yellow: Asset 4
  - Magenta: Asset 5
- Legend on the right (x=710, y=290) lists asset colors
- Displays portfolio metrics (total value, cash, return, step, cumulative reward)

### REINFORCE Video (`results/reinforce_performance.mp4`)
- 3-minute video (3 episodes, ~100 steps each, 30 fps)
- Shows REINFORCE agent trading (~1.97 cumulative reward)
- Same chart and legend format as the GIF

## Troubleshooting

### ModuleNotFoundError

Ensure proper module paths and run with explicit PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:./implementation:./environment
python main.py
```

### REINFORCE Model Not Found

Verify the model file exists and retrain if necessary:

```bash
# Check if model exists
ls -la models/pg/reinforce_trading_env.pth

# Retrain if needed
cd training
python pg_training.py
```

### GIF/Video Generation Fails

Confirm all dependencies are installed:

```bash
# Check ffmpeg
ffmpeg -version

# Check Python packages
pip show opencv-python imageio imageio-ffmpeg
```

### Run Standalone REINFORCE Video

If the main script fails, try generating the video directly:

```bash
cd implementation
python -c "
from rendering import record_agent_performance
from environment.custom_env import TradingEnv
from render_pg import REINFORCE

env = TradingEnv()
model = REINFORCE(env).load('../models/pg/reinforce_trading_env')
record_agent_performance(env, model, '../results/reinforce_performance.mp4', episodes=3, max_steps=100, fps=30)
"
```

### Debug Cumulative Reward Issues

Add debug prints to verify reward tracking:

```bash
# Edit implementation/rendering.py
# Add to _draw_portfolio_metrics method:
# print(f"Debug: episode_total_reward = {getattr(env, 'episode_total_reward', 0)}")
```

Verify that `custom_env.py` properly updates `episode_total_reward` in the `step` method.

### Legend/Chart Display Issues

If legend overlaps with chart, adjust positioning in `rendering.py`:

```python
# Edit implementation/rendering.py
# In _draw_price_chart method:
legend_x, legend_y = x + w + 30, y + 10  # Increase spacing
```

## Performance Notes

The REINFORCE agent (1.97 ± 0.22) outperforms other RL models in this environment, likely due to better handling of transaction costs and concentration penalties. The random agent GIF includes a halved-width price chart (640 pixels) with a legend on the right for improved clarity.

## Built With

- **Gymnasium** - Environment framework
- **Stable Baselines3** - RL algorithms
- **PyTorch** - Deep learning framework
- **Pygame** - Visualization rendering
- **OpenCV** - Video processing
