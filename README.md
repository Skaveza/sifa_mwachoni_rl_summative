Sifa Mwachoni RL Summative
Overview
This project implements a reinforcement learning (RL) environment for a simulated investment platform, inspired by the capstone project to provide Kenyan investors with access to global financial markets (NYSE stocks and ETFs) through a secure, compliance-first platform with automated dividend reinvestment (Auto-DRIP) and active trading tools. The goal is to compare RL methods (DQN, REINFORCE, PPO) in this custom environment, as part of a summative assignment.
Repository Structure
sifa_mwachoni_rl_summative/
├── environment/
│   ├── custom_env.py       # Custom TradingEnv implementation
│   ├── __pycache__/
├── implementation/
│   ├── rendering.py        # Pygame visualization and GIF generation
│   ├── __pycache__/
├── training/
│   ├── dqn_training.py     # DQN training script
│   ├── pg_training.py      # PPO/REINFORCE training script
├── models/
│   ├── dqn/                # Saved DQN models
│   ├── pg/                 # Saved policy gradient models
├── main.py                 # Script to generate random action GIF
├── requirements.txt        # Project dependencies
├── README.md
├── trading_env_random.gif  # Visualization of random actions
├── venv/                   # Virtual environment

Implemented Components
Custom Environment (environment/custom_env.py)
The TradingEnv class simulates a portfolio management scenario for Kenyan investors accessing NYSE stocks and ETFs. Key features:

State Space: Continuous, including:
Cash balance
Shares held for 5 assets
Asset prices
Dividend payouts
KYC compliance flag (1 = compliant, 0 = non-compliant)
7-day moving averages for each asset


Action Space: Discrete, 26 actions:
Per asset (5 assets): Hold, Buy, Sell, Set Stop-Loss, Reinvest Dividends
No action (entire portfolio)


Rewards:
Portfolio value increase ((new - prev) / initial_cash)
Dividend income (sum(dividends) / initial_cash)
Auto-DRIP bonus (+0.05)
Stop-loss bonus/penalty (±0.1 based on price vs. moving average)
Transaction cost (-0.01 for buy/sell)
Invalid action penalty (-0.1)
Non-compliance penalty (-0.5)


Dynamics:
Random walk for asset prices
Random dividends every 10 steps
Random KYC compliance checks (5% chance of non-compliance)
Stop-loss triggers at 95% of asset price


Alignment with Capstone: Supports global market access (5 assets), Auto-DRIP, active trading (stop-loss), and KYC compliance.

Visualization (implementation/rendering.py)
The Renderer class uses Pygame to visualize the environment. Run main.py to generate a GIF of random actions (trading_env_random.gif).
Displayed Components:

Portfolio value and cash (text)
KYC status (green/red circle)
Per asset: Price, moving average (text), shares (bar), dividends (text when non-zero)


Setup Instructions

Clone the Repository:
git clone <repository-url>
cd sifa_mwachoni_rl_summative


Set Up Virtual Environment:
python -m venv venv
source venv/bin/activate  # On macOS/Linux


Install Dependencies:
pip install -r requirements.txt

requirements.txt includes:
gymnasium
numpy
pygame
imageio



Next Steps

Implement Training Scripts:
dqn_training.py: Train a DQN agent using Stable-Baselines3.
pg_training.py: Train PPO/REINFORCE agents using Stable-Baselines3.


Save Models: Store trained models in models/dqn/ and models/pg/.
Enhancements:
Add historical NYSE/ETF data for realistic price movements.
Include action labels in the visualization.
Expand state space with additional technical indicators (e.g., RSI).


Compare RL Methods: Train and evaluate DQN, REINFORCE, and PPO, comparing performance in the TradingEnv.

Troubleshooting

Import Errors: Ensure imports in custom_env.py and main.py use implementation.rendering. If issues persist, add the project root to PYTHONPATH:export PYTHONPATH=$PYTHONPATH:/path/to/sifa_mwachoni_rl_summative


Pygame Issues: Verify display environment is set up (macOS typically handles this automatically).
Dependencies: Reinstall with pip install -r requirements.txt if errors occur.
