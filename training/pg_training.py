import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.custom_env import TradingEnv

np.random.seed(42)
torch.manual_seed(42)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class REINFORCE:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropies = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        self.entropies.append(entropy.item())
        return action.item(), log_prob

    def learn(self, total_timesteps=300000):
        episode_rewards = []
        for _ in range(total_timesteps // self.env.max_steps):
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                state, reward, done, truncated, _ = self.env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                if done or truncated:
                    break
            
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
            policy_loss = torch.stack(policy_loss).sum()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            episode_rewards.append(sum(rewards))
        return episode_rewards, self.entropies

    def save(self, path):
        torch.save(self.policy.state_dict(), path + ".pth")

    def predict(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state)
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action, None

class PGCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PGCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.entropies = []
        self.current_rewards = []
    
    def _on_step(self) -> bool:
        self.current_rewards.append(self.locals["rewards"])
        if "entropies" in self.locals:
            self.entropies.append(self.locals["entropies"])
        if self.locals["dones"]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []
        return True

class PolicyGradientTrainer:
    def __init__(self):
        self.results = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sys.path.append(project_root)
        
    def train(self):
        env = Monitor(TradingEnv())
        os.makedirs("models/pg", exist_ok=True)
        
        # REINFORCE
        print("\n=== Training REINFORCE Agent ===")
        reinforce = REINFORCE(env, learning_rate=3e-4, gamma=0.99)
        episode_rewards, entropies = reinforce.learn(total_timesteps=300000)
        reinforce.save("models/pg/reinforce_trading_env")
        mean_reward = np.mean(episode_rewards[-50:])
        std_reward = np.std(episode_rewards[-50:])
        self.results.append(("REINFORCE", mean_reward, std_reward))
        np.save("results/reinforce_episode_rewards.npy", episode_rewards)
        np.save("results/reinforce_entropies.npy", entropies)
        
        # PPO
        print("\n=== Training PPO Agent ===")
        ppo_callback = PGCallback()
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
        ppo.learn(total_timesteps=300000, callback=ppo_callback)
        ppo.save("models/pg/ppo_trading_env")
        mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=50)
        self.results.append(("PPO", mean_reward, std_reward))
        np.save("results/ppo_episode_rewards.npy", ppo_callback.episode_rewards)
        np.save("results/ppo_entropies.npy", ppo_callback.entropies)
        
        # A2C
        print("\n=== Training A2C Agent ===")
        a2c_callback = PGCallback()
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
        a2c.learn(total_timesteps=300000, callback=a2c_callback)
        a2c.save("models/pg/a2c_trading_env")
        mean_reward, std_reward = evaluate_policy(a2c, env, n_eval_episodes=50)
        self.results.append(("A2C", mean_reward, std_reward))
        np.save("results/a2c_episode_rewards.npy", a2c_callback.episode_rewards)
        np.save("results/a2c_entropies.npy", a2c_callback.entropies)
        
        self.print_results()
    
    def print_results(self):
        print("\n=== Final Evaluation Results ===")
        print(f"{'Model':<12} | {'Mean Reward':<12} | {'Std Reward':<12}")
        print("-" * 40)
        for name, mean_reward, std_reward in self.results:
            print(f"{name:<12} | {mean_reward:>11.2f} | {std_reward:>11.2f}")

if __name__ == "__main__":
    trainer = PolicyGradientTrainer()
    trainer.train()