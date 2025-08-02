# Model Performance Summary

| Model     |   Final Cumulative Reward |   Final Avg Reward |   Best Episode Reward |   Worst Episode Reward |   Reward Stability (Std) | Episodes to 90% Performance   |   Episodes to Convergence |   Sample Efficiency Rank |
|:----------|--------------------------:|-------------------:|----------------------:|-----------------------:|-------------------------:|:------------------------------|--------------------------:|-------------------------:|
| DQN       |                      6778 |               1.3  |                  2.17 |                   0.3  |                     0.17 | N/A                           |                       662 |                        0 |
| REINFORCE |                      4639 |               1.54 |                  2.15 |                   0.65 |                     0.14 | N/A                           |                       100 |                        0 |
| PPO       |                      5450 |               1.85 |                  2.45 |                   0.87 |                     0.21 | 1                             |                       100 |                        1 |
| A2C       |                      5807 |               1.94 |                  2.59 |                   0.89 |                     0.18 | N/A                           |                       100 |                        0 |

## Interpretation:
- **Final Cumulative Reward**: Total reward accumulated over all training episodes
- **Final Avg Reward**: Average reward per episode in the last episodes
- **Reward Stability**: Standard deviation of rewards in the final episodes (lower is more stable)
- **Episodes to 90% Performance**: How quickly the model reaches 90% of its final performance
- **Episodes to Convergence**: When the model's performance stabilizes within 10% of final performance
- **Sample Efficiency Rank**: Ranking based on episodes to reach 90% performance (1 = most efficient)
